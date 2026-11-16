"""운영 데이터 읽기.

데이터는 한 곳(운영시트)에 모은다. 엑셀 통합문서(.xlsx) 한 개이거나,
시트 이름과 같은 CSV 파일들이 든 폴더다. 그로우즈·결제선생·AR에서
내보낸 파일은 열 이름이 달라도 SCHEMA의 별칭으로 맞춰 읽는다.
"""

from __future__ import annotations

import csv
import re
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

# 표 → (시트 이름, {내부 필드: [열 이름 별칭들]}, 필수 필드)
SCHEMA: dict[str, dict] = {
    "students": {
        "sheet": "학생",
        "fields": {
            "name": ["이름", "학생명", "학생이름", "성명", "학생"],
            "grade": ["학년"],
            "course": ["과정", "반", "수강과정"],
            "days": ["등원요일", "요일", "수업요일"],
            "time": ["등원시간", "시간", "수업시간"],
            "enrolled": ["등록일", "입학일", "입원일"],
            "status": ["상태", "재원상태"],
            "left": ["퇴원일"],
            "phone": ["학부모연락처", "연락처", "보호자연락처", "학부모 연락처"],
            "level": ["현재레벨", "레벨", "AR레벨", "북레벨"],
            "bill_day": ["결제일", "납부일(매월)", "결제일(매월)"],
            "fee": ["원비", "교습비", "수강료"],
            "memo": ["메모", "비고", "특이사항"],
        },
        "required": ["name"],
    },
    "attendance": {
        "sheet": "출결",
        "fields": {
            "date": ["날짜", "일자", "출결일"],
            "name": ["이름", "학생명", "학생이름", "성명"],
            "status": ["상태", "출결", "출결상태"],
            "reason": ["사유", "결석사유", "메모", "비고"],
        },
        "required": ["date", "name", "status"],
    },
    "reading": {
        "sheet": "독서기록",
        "fields": {
            "date": ["날짜", "퀴즈일", "응시일", "Date Taken", "Quiz Date", "Date"],
            "name": ["이름", "학생명", "학생이름", "Student", "Student Name"],
            "title": ["책제목", "제목", "도서명", "Title", "Book Title"],
            "bl": ["북레벨", "BL", "레벨", "Book Level", "ATOS Book Level"],
            "pct": ["정답률", "정답률(%)", "% Correct", "Percent Correct", "점수"],
            "words": ["단어수", "Word Count", "Words"],
        },
        "required": ["date", "name"],
    },
    "sr": {
        "sheet": "레벨진단",
        "fields": {
            "date": ["날짜", "응시일", "테스트일", "Test Date", "Date"],
            "name": ["이름", "학생명", "학생이름", "Student"],
            "ge": ["SR점수", "SR", "GE", "Grade Equivalent", "점수"],
        },
        "required": ["date", "name"],
    },
    "payments": {
        "sheet": "수납",
        "fields": {
            "month": ["청구월", "수납월", "해당월"],
            "name": ["이름", "학생명", "학생이름", "성명"],
            "amount": ["금액", "청구금액", "청구액"],
            "due": ["납부기한", "결제기한", "기한"],
            "paid": ["납부일", "결제일", "수납일", "결제완료일"],
        },
        "required": ["name", "due"],
    },
    "contacts": {
        "sheet": "상담연락",
        "fields": {
            "date": ["날짜", "상담일", "일자"],
            "name": ["이름", "학생명", "학생이름", "성명"],
            "kind": ["구분", "유형", "상담구분"],
            "method": ["방법", "수단"],
            "note": ["내용", "상담내용", "메모"],
            "follow_up": ["후속조치일", "후속일", "다음연락일"],
            "follow_done": ["후속완료", "완료"],
            "by": ["담당", "담당자", "작성자"],
        },
        "required": ["date", "name"],
    },
    "leads": {
        "sheet": "문의",
        "fields": {
            "date": ["문의일", "날짜", "접수일"],
            "name": ["학생이름", "이름", "학생명"],
            "grade": ["학년"],
            "phone": ["연락처", "학부모연락처"],
            "source": ["경로", "유입경로"],
            "status": ["상태", "진행상태"],
            "test_date": ["진단일", "진단평가일", "레벨테스트일", "방문일"],
            "last_contact": ["최근연락일", "마지막연락일"],
            "memo": ["메모", "비고"],
        },
        "required": ["date", "name"],
    },
}

WEEKDAYS = "월화수목금토일"
ACTIVE = "재원"
PRESENT = {"출석", "지각", "보강"}
CONTACT_KINDS = [
    "정기상담", "결석연락", "수납연락", "해피콜", "전환상담",
    "레벨안내", "칭찬", "클레임", "기타",
]
LEAD_STATES = ["신규문의", "회신완료", "진단예약", "진단완료", "등록", "종료"]


@dataclass
class Student:
    name: str
    grade: str = ""
    course: str = "도서관"
    days: frozenset[int] = frozenset()
    time: str = ""
    enrolled: date | None = None
    status: str = ACTIVE
    left: date | None = None
    phone: str = ""
    level: float | None = None
    bill_day: int | None = None
    fee: int = 0
    memo: str = ""

    @property
    def active(self) -> bool:
        return self.status == ACTIVE

    @property
    def days_label(self) -> str:
        return "".join(WEEKDAYS[d] for d in sorted(self.days))


@dataclass
class Issue:
    """관리자가 고쳐야 할 데이터 입력 문제."""

    table: str
    row: int
    message: str


@dataclass
class Data:
    students: dict[str, Student] = field(default_factory=dict)
    attendance: list[dict] = field(default_factory=list)
    reading: list[dict] = field(default_factory=list)
    sr: list[dict] = field(default_factory=list)
    payments: list[dict] = field(default_factory=list)
    contacts: list[dict] = field(default_factory=list)
    leads: list[dict] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def active_students(self) -> list[Student]:
        return sorted((s for s in self.students.values() if s.active), key=lambda s: s.name)

    def of(self, table: str, name: str) -> list[dict]:
        """한 학생의 기록을 날짜순으로."""
        rows = [r for r in getattr(self, table) if r["name"] == name]
        return sorted(rows, key=lambda r: r.get("date") or r.get("due") or date.min)


# ── 값 해석 ──────────────────────────────────────────────────────────

def parse_date(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)) and 20000 < value < 80000:  # 엑셀 날짜 일련번호
        return date(1899, 12, 30) + timedelta(days=int(value))
    text = str(value).strip()
    m = re.match(r"^(\d{4})\D+(\d{1,2})\D+(\d{1,2})", text)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", text)  # AR 내보내기: MM/DD/YYYY
    if m:
        try:
            return date(int(m[3]), int(m[1]), int(m[2]))
        except ValueError:
            return None
    return None


def parse_num(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = re.sub(r"[,\s원%]", "", str(value))
    try:
        return float(text)
    except ValueError:
        return None


def parse_pct(value) -> float | None:
    """정답률. 0.85와 85, '85%'를 모두 85로 읽는다."""
    num = parse_num(value)
    if num is None:
        return None
    if isinstance(value, str) and "%" in value:
        return num
    return num * 100 if 0 < num <= 1 else num


def parse_days(value) -> frozenset[int]:
    return frozenset(WEEKDAYS.index(ch) for ch in str(value or "") if ch in WEEKDAYS)


def parse_time(value) -> str:
    """'3시', '3:30', '15:00', 시간 셀 → 'HH:MM'. 1~9시는 오후로 본다."""
    if value is None or value == "":
        return ""
    if hasattr(value, "hour"):
        hour, minute = value.hour, value.minute
    else:
        m = re.search(r"(\d{1,2})\s*(?:[:시]\s*(\d{1,2}))?", str(value))
        if not m:
            return ""
        hour, minute = int(m[1]), int(m[2] or 0)
    if 1 <= hour <= 9:
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def parse_month(value) -> str:
    """청구월 → 'YYYY-MM'."""
    if isinstance(value, (date, datetime)):
        return f"{value.year}-{value.month:02d}"
    m = re.match(r"^(\d{4})\D+(\d{1,2})", str(value or "").strip())
    return f"{m[1]}-{int(m[2]):02d}" if m else ""


def _text(value) -> str:
    return "" if value is None else str(value).strip()


# ── 파일 읽기 ────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[list]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.reader(f)]


def _read_tables(source: Path) -> dict[str, list[list]]:
    """{시트 이름: 행들(첫 행이 머리글)}"""
    if source.is_dir():
        return {p.stem: _read_csv(p) for p in sorted(source.glob("*.csv"))}
    if source.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook

        book = load_workbook(source, data_only=True, read_only=True)
        return {ws.title: [list(row) for row in ws.iter_rows(values_only=True)] for ws in book}
    raise ValueError(f"읽을 수 없는 데이터 위치입니다: {source} (.xlsx 파일이나 CSV 폴더를 지정하세요)")


def _load_aliases(path: Path | None) -> None:
    """config/columns.toml의 별칭을 SCHEMA에 더한다."""
    if not path or not path.exists():
        return
    extra = tomllib.loads(path.read_text(encoding="utf-8"))
    for table, fields in extra.items():
        for fname, aliases in fields.items():
            if table in SCHEMA and fname in SCHEMA[table]["fields"]:
                known = SCHEMA[table]["fields"][fname]
                known.extend(a for a in aliases if a not in known)


def _rows(tables: dict[str, list[list]], table: str, issues: list[Issue]) -> list[tuple[int, dict]]:
    spec = SCHEMA[table]
    raw = tables.get(spec["sheet"]) or tables.get(table)
    if not raw:
        return []
    header = [_text(h) for h in raw[0]]
    index: dict[str, int] = {}
    for fname, aliases in spec["fields"].items():
        for alias in aliases:
            if alias in header:
                index[fname] = header.index(alias)
                break
    missing = [spec["fields"][f][0] for f in spec["required"] if f not in index]
    if missing:
        issues.append(Issue(spec["sheet"], 1, f"필수 열이 없습니다: {', '.join(missing)}"))
        return []
    out = []
    for n, row in enumerate(raw[1:], start=2):
        if not any(_text(c) for c in row):
            continue
        out.append((n, {f: (row[i] if i < len(row) else None) for f, i in index.items()}))
    return out


def load(source: str | Path, columns: Path | None = None) -> Data:
    source = Path(source)
    _load_aliases(columns)
    tables = _read_tables(source)
    data = Data()
    issues = data.issues

    if SCHEMA["students"]["sheet"] not in tables and "students" not in tables:
        issues.append(Issue("학생", 1, "학생 시트를 찾을 수 없습니다."))

    for n, r in _rows(tables, "students", issues):
        name = _text(r.get("name"))
        if not name:
            continue
        if name in data.students:
            issues.append(Issue("학생", n, f"이름이 겹칩니다: {name} (예: '{name}A', '{name}B'로 구분)"))
            continue
        level = parse_num(r.get("level"))
        bill_day = parse_num(r.get("bill_day"))
        data.students[name] = Student(
            name=name,
            grade=_text(r.get("grade")),
            course=_text(r.get("course")) or "도서관",
            days=parse_days(r.get("days")),
            time=parse_time(r.get("time")),
            enrolled=parse_date(r.get("enrolled")),
            status=_text(r.get("status")) or ACTIVE,
            left=parse_date(r.get("left")),
            phone=_text(r.get("phone")),
            level=level,
            bill_day=int(bill_day) if bill_day and 1 <= bill_day <= 31 else None,
            fee=int(parse_num(r.get("fee")) or 0),
            memo=_text(r.get("memo")),
        )

    def known(table: str, n: int, name: str) -> bool:
        if name in data.students:
            return True
        issues.append(Issue(SCHEMA[table]["sheet"], n, f"학생 시트에 없는 이름입니다: {name}"))
        return False

    def dated(table: str, n: int, value, label: str = "날짜") -> date | None:
        d = parse_date(value)
        if d is None:
            issues.append(Issue(SCHEMA[table]["sheet"], n, f"{label}를 읽을 수 없습니다: {value!r}"))
        return d

    for n, r in _rows(tables, "attendance", issues):
        name, d = _text(r["name"]), dated("attendance", n, r["date"])
        if d and known("attendance", n, name):
            data.attendance.append(
                {"date": d, "name": name, "status": _text(r["status"]), "reason": _text(r.get("reason"))}
            )

    for n, r in _rows(tables, "reading", issues):
        name, d = _text(r["name"]), dated("reading", n, r["date"])
        if d and known("reading", n, name):
            data.reading.append({
                "date": d, "name": name, "title": _text(r.get("title")),
                "bl": parse_num(r.get("bl")), "pct": parse_pct(r.get("pct")),
                "words": int(parse_num(r.get("words")) or 0),
            })

    for n, r in _rows(tables, "sr", issues):
        name, d = _text(r["name"]), dated("sr", n, r["date"])
        if d and known("sr", n, name):
            data.sr.append({"date": d, "name": name, "ge": parse_num(r.get("ge"))})

    for n, r in _rows(tables, "payments", issues):
        name, due = _text(r["name"]), dated("payments", n, r["due"], "납부기한")
        if due and known("payments", n, name):
            data.payments.append({
                "name": name, "due": due, "paid": parse_date(r.get("paid")),
                "amount": int(parse_num(r.get("amount")) or 0),
                "month": parse_month(r.get("month")) or f"{due.year}-{due.month:02d}",
            })

    for n, r in _rows(tables, "contacts", issues):
        name, d = _text(r["name"]), dated("contacts", n, r["date"])
        if d and known("contacts", n, name):
            kind = _text(r.get("kind")) or "기타"
            if kind not in CONTACT_KINDS:
                issues.append(Issue("상담연락", n, f"구분을 목록에서 골라 주세요: {kind!r}"))
            data.contacts.append({
                "date": d, "name": name, "kind": kind, "method": _text(r.get("method")),
                "note": _text(r.get("note")), "follow_up": parse_date(r.get("follow_up")),
                "follow_done": _text(r.get("follow_done")).upper() in {"Y", "O", "완료", "TRUE", "예"},
                "by": _text(r.get("by")),
            })

    for n, r in _rows(tables, "leads", issues):
        name, d = _text(r["name"]), dated("leads", n, r["date"], "문의일")
        if not d or not name:
            continue
        status = _text(r.get("status")) or "신규문의"
        if status not in LEAD_STATES:
            issues.append(Issue("문의", n, f"상태를 목록에서 골라 주세요: {status!r}"))
        data.leads.append({
            "date": d, "name": name, "grade": _text(r.get("grade")), "phone": _text(r.get("phone")),
            "source": _text(r.get("source")), "status": status,
            "test_date": parse_date(r.get("test_date")),
            "last_contact": parse_date(r.get("last_contact")), "memo": _text(r.get("memo")),
        })

    return data


def load_config(config_dir: str | Path) -> tuple[dict, dict]:
    """(규칙, 문자 템플릿)"""
    config_dir = Path(config_dir)
    rules = tomllib.loads((config_dir / "rules.toml").read_text(encoding="utf-8"))
    templates = tomllib.loads((config_dir / "templates.toml").read_text(encoding="utf-8"))
    return rules, templates
