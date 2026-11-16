"""운영시트(.xlsx) 만들기. 빈 템플릿과 데모 데이터가 같은 틀을 쓴다."""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from .store import CONTACT_KINDS, LEAD_STATES, SCHEMA

ORDER = ["students", "attendance", "reading", "sr", "payments", "contacts", "leads"]

# (표, 필드) → 드롭다운 선택지
CHOICES = {
    ("students", "course"): ["도서관", "입시"],
    ("students", "status"): ["재원", "휴원", "퇴원"],
    ("attendance", "status"): ["출석", "결석", "지각", "보강"],
    ("contacts", "kind"): CONTACT_KINDS,
    ("contacts", "method"): ["전화", "문자", "대면"],
    ("contacts", "follow_done"): ["Y"],
    ("leads", "status"): LEAD_STATES,
}

GUIDE = [
    ("탐스 운영시트", ""),
    ("", ""),
    ("이 파일 하나가 학원 운영 데이터의 원본입니다. 관리자는 매일 여기에만 입력합니다.", ""),
    ("", ""),
    ("시트", "언제, 무엇을 적나"),
    ("학생", "등록·퇴원·레벨 변경이 있을 때. 등원요일은 '월수금', 등원시간은 '15:00'처럼 적습니다."),
    ("출결", "매일 마감 전. 한 학생 한 줄 (출석/결석/지각/보강). 미리 연락받은 결석은 사유를 적습니다."),
    ("독서기록", "매일 마감 전. AR 퀴즈 결과를 붙여 넣습니다 (책제목, 북레벨, 정답률, 단어수)."),
    ("레벨진단", "SR 테스트를 본 날."),
    ("수납", "결제선생에서 청구서를 보낸 날 한 줄 추가, 입금되면 납부일을 채웁니다."),
    ("상담연락", "학부모에게 전화·문자를 한 직후. 구분을 정확히 골라야 브리핑의 할 일이 사라집니다."),
    ("문의", "신규 문의가 온 즉시. 진행될 때마다 상태를 바꿉니다."),
    ("", ""),
    ("규칙", ""),
    ("1", "이름은 학생 시트와 똑같이 적습니다. 동명이인은 '김민준A', '김민준B'로 구분합니다."),
    ("2", "날짜는 2026-11-16 형식으로 적습니다."),
    ("3", "지나간 줄은 지우지 않습니다. 기록이 쌓여야 리포트가 만들어집니다."),
    ("4", "머리글(첫 줄)의 이름은 바꾸지 않습니다."),
]


def write_workbook(path: str | Path, rows: dict[str, list[dict]] | None = None) -> Path:
    """rows: {표 이름: [{필드: 값}]}. 비우면 빈 템플릿."""
    rows = rows or {}
    book = Workbook()
    guide = book.active
    guide.title = "안내"
    for line in GUIDE:
        guide.append(line)
    guide["A1"].font = Font(bold=True, size=16)
    for cell in (guide["A5"], guide["B5"], guide["A14"]):
        cell.font = Font(bold=True)
    guide.column_dimensions["A"].width = 14
    guide.column_dimensions["B"].width = 96

    head_fill = PatternFill("solid", fgColor="1F2A44")
    for table in ORDER:
        spec = SCHEMA[table]
        ws = book.create_sheet(spec["sheet"])
        fields = list(spec["fields"])
        ws.append([spec["fields"][f][0] for f in fields])
        for col, fname in enumerate(fields, start=1):
            cell = ws.cell(row=1, column=col)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = head_fill
            cell.alignment = Alignment(horizontal="center")
            letter = get_column_letter(col)
            ws.column_dimensions[letter].width = 34 if fname in {"note", "memo", "title"} else 14
            choices = CHOICES.get((table, fname))
            if choices:
                rule = DataValidation(type="list", formula1='"' + ",".join(choices) + '"', allow_blank=True)
                rule.error, rule.errorTitle = "목록에서 골라 주세요.", "입력 확인"
                ws.add_data_validation(rule)
                rule.add(f"{letter}2:{letter}5000")
        ws.freeze_panes = "A2"
        for row in rows.get(table, []):
            ws.append([row.get(f) for f in fields])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    return path
