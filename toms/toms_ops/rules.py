"""규칙 엔진: 데이터 + 원장의 판단 기준(rules.toml) → 오늘 할 일.

할 일은 상태를 따로 저장하지 않는다. 데이터에서 매번 다시 계산하므로,
관리자가 운영시트에 처리 내용을 적으면 다음 브리핑에서 저절로 사라진다.
`since`(일이 생긴 날)도 데이터에서 나오므로, 며칠째 밀렸는지 속일 수 없다.
"""

from __future__ import annotations

import calendar
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from .metrics import brief_lines, mean, student_stats
from .store import WEEKDAYS, Data, Issue, Student

URGENT, TODAY, WEEK = 0, 1, 2
PRIORITY_LABEL = {URGENT: "긴급", TODAY: "오늘", WEEK: "이번 주"}
OWNER_RULES = {"A3", "R3", "P4", "X1", "T1"}


@dataclass
class Task:
    rule: str
    category: str
    priority: int
    assignee: str
    student: str
    title: str
    reason: str
    action: str
    since: date
    sla: int
    message: str = ""
    done_when: str = ""
    brief: tuple[str, ...] = ()
    open_days: frozenset[int] = frozenset(range(5))

    def age(self, today: date) -> int:
        """일이 생긴 뒤 지난 운영일 수. 주말에 쌓인 날은 세지 않는다."""
        start = self.since
        while start.weekday() not in self.open_days and start < today:
            start += timedelta(days=1)  # 쉬는 날 생긴 일은 다음 운영일부터 센다
        days = (today - start).days
        return sum(1 for n in range(1, days + 1) if (start + timedelta(days=n)).weekday() in self.open_days)

    def overdue(self, today: date) -> bool:
        return self.age(today) > self.sla


class Engine:
    def __init__(self, data: Data, cfg: dict, templates: dict, today: date):
        self.data, self.cfg, self.tpl, self.today = data, cfg, templates, today
        self.manager = cfg["center"]["manager"]
        self.owner = cfg["center"]["owner"]
        self.open_days = frozenset(cfg["center"]["open_days"])
        self.tasks: list[Task] = []
        self._stats: dict[str, dict] = {}

    # ── 도우미 ───────────────────────────────────────────────────────

    def stats(self, s: Student) -> dict:
        if s.name not in self._stats:
            self._stats[s.name] = student_stats(self.data, s, self.today, self.cfg)
        return self._stats[s.name]

    def msg(self, key: str, **kw) -> str:
        values = defaultdict(str, greeting=self.cfg["center"]["greeting"], **kw)
        return self.tpl[key].format_map(values).strip()

    def contacted(self, name: str, since: date, kinds=None, method: str = "", by: str = "") -> bool:
        for c in self.data.contacts:
            if c["name"] != name or c["date"] < since:
                continue
            if kinds and c["kind"] not in kinds:
                continue
            if method and method not in c["method"]:
                continue
            if by and by not in c["by"]:
                continue
            return True
        return False

    def owner_touched(self, name: str, since: date) -> bool:
        return self.contacted(name, since, by=self.owner)

    def add(self, rule, category, priority, assignee, student, title, reason, action, since, sla, **kw):
        self.tasks.append(Task(
            rule, category, priority, assignee, student, title, reason, action, since, sla,
            open_days=self.open_days, **kw,
        ))

    def brief(self, s: Student) -> tuple[str, ...]:
        return tuple(brief_lines(s, self.stats(s)))

    # ── 출결 ─────────────────────────────────────────────────────────

    def attendance(self):
        c, today = self.cfg["attendance"], self.today
        for s in self.data.active_students():
            recs = [r for r in self.data.of("attendance", s.name) if r["date"] <= today]
            if not recs:
                continue
            streak = 0
            for r in reversed(recs):
                if r["status"] != "결석":
                    break
                streak += 1
            st = self.stats(s)
            absences = [r for r in recs if r["status"] == "결석"]
            last_abs = absences[-1] if absences else None
            high_rate = st["att_records"] >= c["min_records"] and st["absence_rate"] >= c["escalate_absence_rate"]
            recent_abs = last_abs and (today - last_abs["date"]).days <= c["window_days"]

            if recent_abs and (streak >= c["escalate_after_consecutive"] or high_rate):
                if not self.owner_touched(s.name, last_abs["date"]):
                    why = f"연속 결석 {streak}회" if streak >= c["escalate_after_consecutive"] else (
                        f"최근 4주 결석 {st['absences']}/{st['att_records']}회"
                    )
                    self.add(
                        "A3", "출결", TODAY, self.owner, s.name, "결석 누적 — 원장 통화",
                        f"{why} (마지막 결석 {last_abs['date']:%m/%d})",
                        "원장이 직접 학부모와 통화합니다. 관리자는 이 내용을 원장에게 전달만 합니다.",
                        last_abs["date"], 2,
                        done_when=f"상담연락 시트에 담당={self.owner}으로 통화 기록",
                        brief=self.brief(s),
                    )
            elif streak >= c["call_after_consecutive"]:
                if not self.contacted(s.name, last_abs["date"], ["결석연락"], method="전화"):
                    self.add(
                        "A2", "출결", URGENT, self.manager, s.name, f"연속 결석 {streak}회 — 전화",
                        f"{recs[-streak]['date']:%m/%d}부터 {streak}회 연속 결석",
                        "학부모에게 전화해 사유와 다음 등원일을 확인합니다. 문자로 대신하지 않습니다.",
                        last_abs["date"], 1,
                        message=self.msg("absent_repeat", name=s.name, count=streak),
                        done_when="상담연락 시트에 구분=결석연락, 방법=전화로 기록",
                        brief=self.brief(s),
                    )
            elif streak == 1 and not last_abs["reason"] and (today - last_abs["date"]).days <= 7:
                if not self.contacted(s.name, last_abs["date"], ["결석연락"]):
                    self.add(
                        "A1", "출결", URGENT, self.manager, s.name, "무단결석 — 당일 연락",
                        f"{last_abs['date']:%m/%d}({WEEKDAYS[last_abs['date'].weekday()]}) 사유 없이 결석",
                        "학부모에게 문자를 보내고, 1시간 안에 답이 없으면 전화합니다.",
                        last_abs["date"], c["unexcused_sla"],
                        message=self.msg("absent", name=s.name),
                        done_when="상담연락 시트에 구분=결석연락으로 기록",
                    )

            if c["makeup_enabled"] and streak < c["call_after_consecutive"]:
                start = today - timedelta(days=c["makeup_window_days"])
                window_abs = [r for r in absences if r["date"] > start]
                if window_abs:
                    makeups = sum(1 for r in recs if r["status"] == "보강" and r["date"] > window_abs[0]["date"])
                    if len(window_abs) > makeups:
                        target = window_abs[makeups]
                        self.add(
                            "A4", "출결", WEEK, self.manager, s.name, "보강 예약",
                            f"{target['date']:%m/%d} 결석분 보강이 아직 없습니다",
                            "학부모와 보강 가능한 회차를 정해 예약합니다. 보강 온 날 출결을 '보강'으로 적습니다.",
                            target["date"], 7,
                            message=self.msg("makeup", name=s.name, date=f"{target['date']:%m/%d}"),
                            done_when="출결 시트에 상태=보강 기록이 생기면 사라집니다",
                        )

    def missing_attendance(self):
        """직전 운영일에 등원 예정이었는데 출결이 비어 있는 학생."""
        day = self.today - timedelta(days=1)
        open_days = self.cfg["center"]["open_days"]
        for _ in range(7):
            if day.weekday() in open_days:
                break
            day -= timedelta(days=1)
        recorded = {r["name"] for r in self.data.attendance if r["date"] == day}
        expected = [
            s.name for s in self.data.active_students()
            if day.weekday() in s.days and (not s.enrolled or s.enrolled <= day)
        ]
        missing = [n for n in expected if n not in recorded]
        if not missing:
            return
        label = f"{day:%m/%d}({WEEKDAYS[day.weekday()]})"
        if not recorded:
            reason = f"{label} 출결이 한 건도 없습니다 (휴원일이었다면 넘어가세요)"
        else:
            reason = f"{label} 출결 미입력 {len(missing)}명: {', '.join(missing)}"
        self.add(
            "D1", "데이터", TODAY, self.manager, "", "어제 출결 입력", reason,
            "출결 시트에 출석/결석을 채웁니다. 비어 있으면 결석 연락·리포트가 모두 틀어집니다.",
            day, 1, done_when="출결 시트에 해당 날짜 기록이 생기면 사라집니다",
        )

    # ── 독서 ─────────────────────────────────────────────────────────

    def reading(self):
        c, today = self.cfg["reading"], self.today
        for s in self.data.active_students():
            if "도서관" not in s.course:
                continue
            st = self.stats(s)
            quizzes = [r for r in self.data.of("reading", s.name) if r["date"] <= today]
            scored = [r for r in quizzes if r["pct"] is not None]
            # 레벨안내 기록 = 레벨을 바꾼 날. 그 뒤에 읽은 책만으로 다시 판단한다.
            changes = [x["date"] for x in self.data.contacts if x["name"] == s.name and x["kind"] == "레벨안내"]
            since_change = [r for r in scored if not changes or r["date"] > max(changes)]

            if s.level is not None:
                at_level = [r for r in since_change if r["bl"] is not None and r["bl"] >= s.level - c["near_level"]]
                recent = at_level[-c["recent_books"]:]
                avg = mean(r["pct"] for r in recent)
                if len(at_level) >= c["min_books_at_level"] and len(recent) == c["recent_books"] and avg >= c["levelup_min_avg"]:
                    new_level = round(s.level + c["levelup_step"], 1)
                    self.add(
                        "R1", "독서", TODAY, self.manager, s.name, f"레벨업 {s.level:.1f} → {new_level:.1f}",
                        f"현재 레벨 책 {len(at_level)}권, 최근 {len(recent)}권 평균 {avg:.0f}% (기준 {c['levelup_min_avg']}%)",
                        f"학생 시트의 현재레벨을 {new_level:.1f}로 고치고, 다음 등원일에 새 레벨 책장에서 첫 책을 같이 고릅니다. "
                        "학부모에게 축하 문자를 보냅니다.",
                        recent[-1]["date"], 7,
                        message=self.msg(
                            "levelup", name=s.name, count=len(recent), avg=f"{avg:.0f}",
                            level=f"{s.level:.1f}", new_level=f"{new_level:.1f}",
                        ),
                        done_when="현재레벨 수정 + 상담연락 시트에 구분=레벨안내로 기록",
                    )

            low = scored[-c["escalate_books"]:]
            if len(low) == c["escalate_books"] and mean(r["pct"] for r in low) < c["struggle_max_avg"]:
                if not self.owner_touched(s.name, today - timedelta(days=14)):
                    self.add(
                        "R3", "독서", TODAY, self.owner, s.name, "독해 부진 지속 — 원장 진단",
                        f"최근 {len(low)}권 평균 {mean(r['pct'] for r in low):.0f}% (기준 {c['struggle_max_avg']}% 미만)",
                        "레벨 조정으로 해결되지 않았습니다. 원장이 학생을 직접 보고 원인(어휘·집중·레벨)을 진단합니다.",
                        low[-1]["date"], 7,
                        done_when=f"상담연락 시트에 담당={self.owner}으로 기록",
                        brief=self.brief(s),
                    )
            else:
                recent = since_change[-c["struggle_books"]:]
                avg = mean(r["pct"] for r in recent)
                if len(recent) == c["struggle_books"] and avg < c["struggle_max_avg"]:
                    level = f"{s.level:.1f}" if s.level is not None else "?"
                    new = f"{s.level - c['leveldown_step']:.1f}" if s.level is not None else "한 단계 아래"
                    self.add(
                        "R2", "독서", TODAY, self.manager, s.name, f"레벨 조정 {level} → {new}",
                        f"최근 {len(recent)}권 평균 {avg:.0f}% (기준 {c['struggle_max_avg']}% 미만)",
                        f"학생 시트의 현재레벨을 {new}로 낮추고, 다음 등원일에 쉬운 책을 같이 고릅니다. "
                        "혼내지 않고 '정확하게 읽는 연습'이라고 설명합니다.",
                        recent[-1]["date"], 5,
                        message=self.msg("leveldown", name=s.name, avg=f"{avg:.0f}"),
                        done_when="현재레벨 수정 + 상담연락 시트에 구분=레벨안내로 기록",
                    )

            settled = s.enrolled and (today - s.enrolled).days >= 14
            if settled and st["present_14d"] >= 2 and st["quizzes_14d"] < c["min_quizzes_14d"]:
                self.add(
                    "R4", "독서", WEEK, self.manager, s.name, "독서량 부족 — 책 고르기 도움",
                    f"최근 2주 출석 {st['present_14d']}회인데 퀴즈 {st['quizzes_14d']}권 (기준 {c['min_quizzes_14d']}권)",
                    "다음 등원일에 옆에서 책을 같이 고르고, 한 권을 끝까지 읽고 퀴즈까지 치는지 확인합니다.",
                    st["last_quiz"] or s.enrolled, 21,
                    message=self.msg("low_volume", name=s.name, count=st["quizzes_14d"]),
                    done_when=f"최근 2주 퀴즈가 {c['min_quizzes_14d']}권 이상이 되면 사라집니다",
                )

            last_sr = st["sr_last"]["date"] if st["sr_last"] else None
            due = (last_sr + timedelta(days=c["sr_retest_days"])) if last_sr else (
                s.enrolled + timedelta(days=7) if s.enrolled else None
            )
            if due and due <= today:
                self.add(
                    "R5", "독서", WEEK, self.manager, s.name, "SR 레벨 진단",
                    f"마지막 SR {last_sr:%Y-%m-%d} (재진단 주기 {c['sr_retest_days']}일)" if last_sr
                    else "SR 진단 기록이 없습니다",
                    "다음 등원일 첫 20분에 SR 테스트를 보게 하고, 결과를 레벨진단 시트에 적습니다.",
                    due, 14,
                    message=self.msg("sr_retest", name=s.name),
                    done_when="레벨진단 시트에 새 기록이 생기면 사라집니다",
                )

            self._milestones(s, quizzes)

    def _milestones(self, s: Student, quizzes: list[dict]):
        c, today = self.cfg["reading"], self.today
        window = timedelta(days=c["milestone_window_days"])
        hit: dict[str, tuple[int, date]] = {}
        words = 0
        for n, r in enumerate(quizzes, start=1):
            before, words = words, words + r["words"]
            if n in c["milestone_books"] and today - r["date"] <= window:
                hit["books"] = (n, r["date"])
            for m in c["milestone_words"]:
                if before < m <= words and today - r["date"] <= window:
                    hit["words"] = (m, r["date"])
        for kind, (value, when) in hit.items():
            if self.contacted(s.name, when, ["칭찬"]):
                continue
            label = f"누적 {value}권" if kind == "books" else f"누적 {value:,} 단어"
            self.add(
                "G1", "칭찬", TODAY, self.manager, s.name, f"{label} 달성 축하",
                f"{when:%m/%d}에 달성",
                "게시판 명예의 전당에 이름을 올리고, 학부모에게 축하 문자를 보냅니다.",
                when, 3,
                message=self.msg(f"milestone_{kind}", name=s.name, count=value, words=f"{value:,}"),
                done_when="상담연락 시트에 구분=칭찬으로 기록",
            )

    # ── 수납 ─────────────────────────────────────────────────────────

    def payments(self):
        c, today = self.cfg["payment"], self.today
        for s in self.data.active_students():
            if not s.bill_day:
                continue
            due = _next_due(today, s.bill_day)
            month = f"{due.year}-{due.month:02d}"
            if (due - today).days <= c["notice_days_before"]:
                if not any(p["name"] == s.name and p["month"] == month for p in self.data.payments):
                    self.add(
                        "P1", "수납", TODAY, self.manager, s.name, f"{due.month}월 청구서 발송",
                        f"결제일 {due:%m/%d} ({(due - today).days}일 전)",
                        "결제선생에서 청구서를 보내고, 수납 시트에 청구 한 줄을 추가합니다.",
                        due - timedelta(days=c["notice_days_before"]), 2,
                        message=self.msg(
                            "payment_notice", name=s.name, month=f"{due.month}월",
                            amount=f"{s.fee:,}" if s.fee else "", due=f"{due:%m/%d}",
                        ),
                        done_when="수납 시트에 해당 월 청구가 생기면 사라집니다",
                    )

        for p in self.data.payments:
            if p["paid"] or p["due"] >= today:
                continue
            late = (today - p["due"]).days
            name, amount = p["name"], f"{p['amount']:,}"
            reason = f"{p['month']} 교습비 {amount}원, 납부기한 {p['due']:%m/%d} ({late}일 경과)"
            if late >= c["escalate_after"]:
                since = p["due"] + timedelta(days=c["escalate_after"])
                if not self.owner_touched(name, since):
                    self.add(
                        "P4", "수납", TODAY, self.owner, name, "장기 미납 — 원장 통화", reason,
                        "문자·전화 안내가 끝났습니다. 원장이 직접 통화해 납부 일정을 정합니다.",
                        since, 3, done_when=f"납부일 입력, 또는 상담연락 시트에 담당={self.owner} 기록",
                    )
            elif late >= c["call_after"]:
                since = p["due"] + timedelta(days=c["call_after"])
                if not self.contacted(name, since, ["수납연락"], method="전화"):
                    self.add(
                        "P3", "수납", TODAY, self.manager, name, "미납 — 전화 안내", reason,
                        "전화로 정중히 확인합니다. '결제선생 청구서가 잘 갔는지 확인차 연락드렸다'로 시작합니다.",
                        since, 2, done_when="납부일 입력, 또는 상담연락 시트에 구분=수납연락, 방법=전화 기록",
                    )
            elif late >= c["remind_after"]:
                since = p["due"] + timedelta(days=c["remind_after"])
                if not self.contacted(name, since, ["수납연락"]):
                    self.add(
                        "P2", "수납", TODAY, self.manager, name, "미납 — 리마인드 문자", reason,
                        "결제선생에서 청구서를 다시 보내고 문자를 함께 보냅니다.",
                        since, 2,
                        message=self.msg("payment_remind", name=name, amount=amount, due=f"{p['due']:%m/%d}"),
                        done_when="납부일 입력, 또는 상담연락 시트에 구분=수납연락 기록",
                    )

    # ── 상담 ─────────────────────────────────────────────────────────

    def consults(self):
        c, today = self.cfg["consult"], self.today
        for s in self.data.active_students():
            st = self.stats(s)
            base = st["last_consult"] or s.enrolled
            if base and (today - base).days > c["interval_days"]:
                since = base + timedelta(days=c["interval_days"])
                self.add(
                    "C1", "상담", WEEK, self.manager, s.name, "정기 상담 전화 (2개월 룰)",
                    f"마지막 상담 {base:%Y-%m-%d} ({(today - base).days}일 전)" if st["last_consult"]
                    else f"등록({base:%Y-%m-%d}) 이후 상담 기록이 없습니다",
                    "아래 브리프의 숫자로 근황을 전하고, 가정에서 느끼는 점과 요청 사항을 듣습니다. "
                    "불만·퇴원 언급이 나오면 약속하지 말고 '원장님이 다시 연락드리겠다'고 한 뒤 후속조치일을 적습니다.",
                    since, c["sla"],
                    done_when="상담연락 시트에 구분=정기상담으로 기록",
                    brief=self.brief(s),
                )
            if s.enrolled:
                calls = sum(
                    1 for x in self.data.contacts
                    if x["name"] == s.name and x["kind"] == "해피콜" and x["date"] >= s.enrolled
                )
                for i, d in enumerate(c["onboarding_days"]):
                    due = s.enrolled + timedelta(days=d)
                    if due <= today <= due + timedelta(days=14) and calls <= i:
                        self.add(
                            "C2", "상담", TODAY, self.manager, s.name, f"신입 해피콜 (등록 {d}일째)",
                            f"등록일 {s.enrolled:%m/%d}",
                            "적응 상태를 전하고 불편한 점을 묻습니다. 첫 달 이탈을 막는 가장 중요한 전화입니다.",
                            due, 3,
                            message=self.msg("onboarding", name=s.name, days=d),
                            done_when="상담연락 시트에 구분=해피콜로 기록",
                            brief=self.brief(s),
                        )
                        break

        for x in self.data.contacts:
            if x["follow_up"] and x["follow_up"] <= today and not x["follow_done"]:
                who = self.owner if self.owner in x["by"] else self.manager
                self.add(
                    "C3", "상담", TODAY, who, x["name"], "약속한 후속 조치",
                    f"{x['date']:%m/%d} {x['kind']}: {x['note'] or '(내용 없음)'}",
                    "약속한 내용을 처리하고 학부모에게 결과를 알립니다.",
                    x["follow_up"], 1, done_when="해당 줄의 후속완료에 Y 입력",
                )

    # ── 신규 문의 ────────────────────────────────────────────────────

    def leads(self):
        c, today = self.cfg["leads"], self.today
        for l in self.data.leads:
            name, status = l["name"], l["status"]
            who = f"{name}({l['grade']})" if l["grade"] else name
            touched = l["last_contact"] or l["date"]
            if status == "신규문의":
                self.add(
                    "L1", "문의", URGENT, self.manager, who, "신규 문의 회신",
                    f"{l['date']:%m/%d} 접수" + (f" · {l['source']}" if l["source"] else ""),
                    "전화로 회신해 진단 평가 날짜를 잡습니다. 학생은 정시, 학부모는 40분 뒤 도착으로 안내합니다.",
                    l["date"], c["reply_sla"],
                    message=self.msg("lead_reply", name=name),
                    done_when="문의 시트 상태를 '회신완료' 또는 '진단예약'으로 변경",
                )
            elif status == "회신완료":
                if (today - touched).days >= c["nudge_after"]:
                    self.add(
                        "L2", "문의", WEEK, self.manager, who, "진단 예약 재연락",
                        f"마지막 연락 {touched:%m/%d}, 진단 예약 전",
                        "부담 없이 한 번 더 연락합니다. 답이 없으면 상태를 '종료'로 바꿉니다.",
                        touched + timedelta(days=c["nudge_after"]), 3,
                        message=self.msg("lead_reply", name=name),
                        done_when="최근연락일 갱신 또는 상태 변경",
                    )
            elif status == "진단예약":
                test = l["test_date"]
                if test is None:
                    self.data.issues.append(Issue("문의", 0, f"{name}: 진단예약인데 진단일이 비어 있습니다"))
                elif test == today:
                    self.add(
                        "L3", "문의", URGENT, self.manager, who, "오늘 진단 평가",
                        "오늘 방문 예정",
                        "진단 평가지·채점 진단표를 출력하고 자리를 준비합니다. 끝나면 채점 → 진단 리포트 → 상담. "
                        "상태를 '진단완료'로 바꿉니다.",
                        today, 0, done_when="문의 시트 상태를 '진단완료'로 변경",
                    )
                elif test < today:
                    self.add(
                        "L5", "문의", TODAY, self.manager, who, "진단 결과 입력",
                        f"진단일 {test:%m/%d}이 지났는데 상태가 '진단예약'입니다",
                        "방문했다면 '진단완료', 오지 않았다면 다시 연락해 날짜를 고칩니다.",
                        test, 1, done_when="문의 시트 상태 또는 진단일 변경",
                    )
                elif (test - today).days == 1 and touched < today:
                    self.add(
                        "L4", "문의", TODAY, self.manager, who, "내일 진단 리마인드 문자",
                        f"진단일 {test:%m/%d}",
                        "리마인드 문자를 보내고 최근연락일을 오늘로 적습니다.",
                        today, 0,
                        message=self.msg("lead_reminder", name=name, date=f"{test:%m/%d}", time="예약 시간"),
                        done_when="최근연락일을 오늘로 입력",
                    )
            elif status == "진단완료":
                base = l["test_date"] or l["date"]
                waited = (today - base).days
                if waited >= c["close_after"]:
                    self.add(
                        "L8", "문의", WEEK, self.manager, who, "문의 종료 처리",
                        f"진단 후 {waited}일 경과, 등록 없음",
                        "상태를 '종료'로 바꾸고 메모에 이유를 적습니다. 다음 모집 때 다시 연락할 명단이 됩니다.",
                        base + timedelta(days=c["close_after"]), 7, done_when="문의 시트 상태를 '종료'로 변경",
                    )
                elif waited >= c["last_try_after"] and touched < base + timedelta(days=c["last_try_after"]):
                    self.add(
                        "L7", "문의", TODAY, self.manager, who, "등록 여부 마지막 확인",
                        f"진단 후 {waited}일 경과",
                        "마지막으로 한 번 연락합니다. 재촉하지 않고 '자리 안내' 정도로만 전합니다.",
                        base + timedelta(days=c["last_try_after"]), 2,
                        message=self.msg("lead_followup", name=name, date=f"{base:%m/%d}"),
                        done_when="최근연락일 갱신 또는 상태 변경",
                    )
                elif waited >= c["followup_after"] and touched < base + timedelta(days=c["followup_after"]):
                    self.add(
                        "L6", "문의", TODAY, self.manager, who, "진단 후 등록 확인 전화",
                        f"진단 후 {waited}일 경과",
                        "진단 리포트에 대한 질문을 받고 등록 의사를 확인합니다.",
                        base + timedelta(days=c["followup_after"]), 2,
                        message=self.msg("lead_followup", name=name, date=f"{base:%m/%d}"),
                        done_when="최근연락일 갱신 또는 상태 변경",
                    )
            elif status == "등록" and name not in self.data.students:
                self.data.issues.append(Issue("문의", 0, f"{name}: 상태가 '등록'인데 학생 시트에 없습니다"))

    # ── 도서관 → 입시반 전환 ─────────────────────────────────────────

    def transition(self):
        c, today = self.cfg["transition"], self.today
        if today.month not in c["months"]:
            return
        first = c["months"][0]
        season = date(today.year if today.month >= first else today.year - 1, first, 1)
        for s in self.data.active_students():
            if s.grade != c["grade"] or c["course"] not in s.course:
                continue
            if self.contacted(s.name, season, ["전환상담"]):
                continue
            self.add(
                "T1", "전환", WEEK, self.owner, s.name, "입시반 전환 상담",
                f"{s.grade} · 2월 개강 전 전환 상담 대상",
                "도서관에서 쌓은 읽기 실력을 브리프의 숫자로 보여주고, 초5 강의형 과정과 학습 진단 평가를 안내합니다.",
                season, c["sla"],
                done_when="상담연락 시트에 구분=전환상담으로 기록",
                brief=self.brief(s),
            )

    # ── 이탈 위험 ────────────────────────────────────────────────────

    def churn_score(self, s: Student) -> tuple[int, list[str]]:
        c, st, today = self.cfg["churn"], self.stats(s), self.today
        score, why = 0.0, []
        if st["att_records"] >= self.cfg["attendance"]["min_records"] and st["absences"]:
            score += c["w_absence"] * min(1.0, st["absence_rate"] / 0.5)
            why.append(f"결석 {st['absences']}/{st['att_records']}회")
        recent, prior = st["avg_recent"], st["avg_prior"]
        if recent is not None and recent < self.cfg["reading"]["struggle_max_avg"]:
            score += c["w_score_drop"]
            why.append(f"정답률 {recent:.0f}%")
        elif recent is not None and prior is not None and prior - recent >= 10:
            score += c["w_score_drop"] * min(1.0, (prior - recent) / 25)
            why.append(f"정답률 {prior:.0f}→{recent:.0f}%")
        need = self.cfg["reading"]["min_quizzes_14d"]
        if "도서관" in s.course and st["quizzes_14d"] < need and s.enrolled and (today - s.enrolled).days >= 14:
            score += c["w_low_volume"] * (1 - st["quizzes_14d"] / need)
            why.append(f"2주 독서 {st['quizzes_14d']}권")
        base = st["last_consult"] or s.enrolled
        if base and (today - base).days > self.cfg["consult"]["interval_days"]:
            score += c["w_no_consult"]
            why.append(f"상담 공백 {(today - base).days}일")
        grace = self.cfg["payment"]["remind_after"]
        if any(p["name"] == s.name and not p["paid"] and (today - p["due"]).days >= grace for p in self.data.payments):
            score += c["w_overdue"]
            why.append("교습비 미납")
        return round(score), why

    def churn(self):
        c, today = self.cfg["churn"], self.today
        for s in self.data.active_students():
            score, why = self.churn_score(s)
            if score < c["threshold"]:
                continue
            if self.owner_touched(s.name, today - timedelta(days=c["recheck_days"])):
                continue
            self.add(
                "X1", "이탈위험", TODAY, self.owner, s.name, f"이탈 위험 {score}점 — 원장 통화",
                " · ".join(why),
                "여러 신호가 겹쳤습니다. 원장이 직접 통화해 가정의 분위기를 확인합니다.",
                today, 7,
                done_when=f"상담연락 시트에 담당={self.owner}으로 기록 (구분=정기상담)",
                brief=self.brief(s),
            )

    # ── 실행 ─────────────────────────────────────────────────────────

    def run(self) -> list[Task]:
        self.tasks = []
        for step in (
            self.missing_attendance, self.attendance, self.reading, self.payments,
            self.consults, self.leads, self.transition, self.churn,
        ):
            step()
        return self._dedupe()

    def _dedupe(self) -> list[Task]:
        """한 학생에게 같은 날 전화가 두 번 가지 않게 정리한다."""
        owner_calls = defaultdict(set)
        for t in self.tasks:
            if t.rule in OWNER_RULES:
                owner_calls[t.student].add(t.rule)
        out = []
        for t in self.tasks:
            rules = owner_calls.get(t.student, set())
            if t.rule == "X1" and rules & {"A3", "R3", "P4"}:
                continue  # 더 구체적인 원장 건이 이미 있다
            if t.rule == "C1" and rules & {"A3", "R3", "X1", "T1"}:
                continue  # 원장 통화가 정기 상담을 겸한다
            out.append(t)
        out.sort(key=lambda t: (t.priority, -t.age(self.today), t.category, t.student))
        return out


def _next_due(today: date, bill_day: int) -> date:
    def on(year: int, month: int) -> date:
        return date(year, month, min(bill_day, calendar.monthrange(year, month)[1]))

    due = on(today.year, today.month)
    if due < today:
        due = on(today.year + today.month // 12, today.month % 12 + 1)
    return due


def build_tasks(data: Data, cfg: dict, templates: dict, today: date) -> list[Task]:
    return Engine(data, cfg, templates, today).run()


def todays_roster(data: Data, tasks: list[Task], today: date) -> list[tuple[str, list[tuple[Student, list[str]]]]]:
    """오늘 등원 예정 학생을 시간대별로. 각 학생에 눈여겨볼 점을 붙인다."""
    notes: dict[str, list[str]] = defaultdict(list)
    short = {
        "R1": "레벨업 대상", "R2": "레벨 조정", "R3": "독해 부진", "R4": "책 고르기 도움",
        "R5": "SR 진단", "G1": "축하", "A1": "결석 후 첫 등원", "A2": "결석 후 첫 등원",
        "A4": "보강 예약 확인", "C2": "신입",
    }
    for t in tasks:
        if t.rule in short and short[t.rule] not in notes[t.student]:
            notes[t.student].append(short[t.rule])
    slots: dict[str, list] = defaultdict(list)
    for s in data.active_students():
        if today.weekday() in s.days and (not s.enrolled or s.enrolled <= today):
            slots[s.time or "시간 미정"].append((s, notes.get(s.name, [])))
    return sorted(slots.items())
