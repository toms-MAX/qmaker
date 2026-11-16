"""실행: toms 폴더에서  python -m unittest"""

import csv
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from toms_ops import metrics, render, sample
from toms_ops.rules import Engine, Task, _next_due, todays_roster
from toms_ops.store import load, load_config, parse_date, parse_days, parse_pct, parse_time
from toms_ops.workbook import write_workbook

ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 11, 16)  # 월요일
CFG, TPL = load_config(ROOT / "config")


def run(rows, today=TODAY):
    with tempfile.TemporaryDirectory() as tmp:
        data = load(write_workbook(Path(tmp) / "book.xlsx", rows))
    engine = Engine(data, CFG, TPL, today)
    return data, engine, engine.run()


def has(tasks, rule, student=None):
    return any(t.rule == rule and (student is None or t.student == student) for t in tasks)


class Parsing(unittest.TestCase):
    def test_dates(self):
        for text in ("2026-11-16", "2026.11.16", "2026/11/16", "2026년 11월 16일", "11/16/2026"):
            self.assertEqual(parse_date(text), TODAY, text)
        self.assertEqual(parse_date(46342), TODAY)  # 엑셀 일련번호
        self.assertIsNone(parse_date("내일"))

    def test_percent_time_days(self):
        self.assertEqual(parse_pct("85%"), 85)
        self.assertEqual(parse_pct(0.85), 85)
        self.assertEqual(parse_pct(85), 85)
        self.assertEqual(parse_time("3시"), "15:00")
        self.assertEqual(parse_time("3:30"), "15:30")
        self.assertEqual(parse_time("14:10"), "14:10")
        self.assertEqual(parse_days("월수금"), frozenset({0, 2, 4}))

    def test_next_due_clamps_to_month_end(self):
        self.assertEqual(_next_due(date(2027, 2, 10), 31), date(2027, 2, 28))
        self.assertEqual(_next_due(date(2026, 12, 20), 5), date(2027, 1, 5))


class Aging(unittest.TestCase):
    def task(self, since):
        return Task("A1", "출결", 0, "관리자", "x", "t", "r", "a", since, 1)

    def test_weekend_is_not_counted(self):
        self.assertEqual(self.task(date(2026, 11, 13)).age(TODAY), 1)  # 금 → 월
        self.assertEqual(self.task(date(2026, 11, 15)).age(TODAY), 0)  # 일요일에 생긴 일
        self.assertFalse(self.task(date(2026, 11, 13)).overdue(TODAY))
        self.assertTrue(self.task(date(2026, 11, 12)).overdue(TODAY))


class Demo(unittest.TestCase):
    """데모 데이터에는 규칙마다 한 명씩 상황을 심어 두었다."""

    @classmethod
    def setUpClass(cls):
        cls.rows = sample.build(TODAY)
        cls.data, cls.engine, cls.tasks = run(cls.rows)

    def test_every_rule_fires_for_its_planted_student(self):
        expected = {
            "A1": "서가온", "A2": "윤나래", "A3": "배다솜", "A4": "서가온",
            "R1": "임라온", "R2": "고마루", "R3": "문바다", "R4": "손사랑", "R5": "양아람", "G1": "권자람",
            "P2": "홍하늘", "P3": "전한결", "P4": "남보람",
            "C1": "심슬기", "C2": "노이든", "C3": "하누리", "T1": "곽도담", "X1": "차여울",
        }
        for rule, student in expected.items():
            self.assertTrue(has(self.tasks, rule, student), f"{rule} {student}")
        for rule in ("D1", "P1", "L1", "L2", "L3", "L4", "L6", "L8"):
            self.assertTrue(has(self.tasks, rule), rule)

    def test_owner_only_gets_escalations(self):
        owner = {t.rule for t in self.tasks if t.assignee == "원장"}
        self.assertEqual(owner, {"A3", "R3", "P4", "X1", "T1"})

    def test_one_call_per_family(self):
        self.assertFalse(has(self.tasks, "C1", "차여울"))  # 원장 통화가 정기 상담을 겸한다
        self.assertFalse(has(self.tasks, "A1", "배다솜"))  # 원장에게 넘어간 결석은 관리자가 연락하지 않는다

    def test_typo_name_is_reported(self):
        self.assertTrue(any("서가은" in i.message for i in self.data.issues))

    def test_left_student_is_ignored(self):
        self.assertFalse(any(t.student == "탁한별" for t in self.tasks))

    def test_roster_lists_only_todays_students(self):
        names = [s.name for _, kids in todays_roster(self.data, self.tasks, TODAY) for s, _ in kids]
        self.assertIn("서가온", names)      # 월수금
        self.assertNotIn("홍하늘", names)   # 화목

    def test_pages_render(self):
        roster = todays_roster(self.data, self.tasks, TODAY)
        brief = render.briefing(self.data, self.tasks, roster, TODAY, CFG)
        self.assertIn("무단결석", brief)
        weekly = render.weekly_report(metrics.weekly(self.data, TODAY, CFG), self.tasks, [], TODAY, CFG)
        self.assertIn("주간 리포트", weekly)
        reports = [metrics.monthly(self.data, s, 2026, 11) for s in self.data.active_students()]
        monthly = render.monthly_report(reports, {"서가온": "직접 쓴 코멘트"}, CFG, TODAY)
        self.assertIn("직접 쓴 코멘트", monthly)


class Resolution(unittest.TestCase):
    """관리자가 운영시트에 기록하면 할 일이 사라진다."""

    def after(self, change):
        rows = sample.build(TODAY)
        change(rows)
        return run(rows)[2]

    def contact(self, name, kind, method="전화", by="관리자"):
        return {"date": TODAY, "name": name, "kind": kind, "method": method, "note": "", "by": by}

    def test_absence_contact(self):
        tasks = self.after(lambda r: r["contacts"].append(self.contact("서가온", "결석연락", "문자")))
        self.assertFalse(has(tasks, "A1", "서가온"))

    def test_repeat_absence_needs_a_phone_call(self):
        tasks = self.after(lambda r: r["contacts"].append(self.contact("윤나래", "결석연락", "문자")))
        self.assertTrue(has(tasks, "A2", "윤나래"))
        tasks = self.after(lambda r: r["contacts"].append(self.contact("윤나래", "결석연락", "전화")))
        self.assertFalse(has(tasks, "A2", "윤나래"))

    def test_owner_call_clears_escalations(self):
        def change(r):
            for name in ("배다솜", "차여울", "문바다", "남보람"):
                r["contacts"].append(self.contact(name, "정기상담", by="원장"))
        tasks = self.after(change)
        for rule, name in (("A3", "배다솜"), ("X1", "차여울"), ("R3", "문바다"), ("P4", "남보람")):
            self.assertFalse(has(tasks, rule, name), rule)

    def test_manager_contact_does_not_clear_owner_task(self):
        tasks = self.after(lambda r: r["contacts"].append(self.contact("배다솜", "결석연락")))
        self.assertTrue(has(tasks, "A3", "배다솜"))

    def test_level_up_recorded(self):
        def change(r):
            next(s for s in r["students"] if s["name"] == "임라온")["level"] = 2.9
            r["contacts"].append(self.contact("임라온", "레벨안내", "문자"))
        self.assertFalse(has(self.after(change), "R1", "임라온"))

    def test_payment_received(self):
        def change(r):
            for p in r["payments"]:
                if p["name"] == "홍하늘":
                    p["paid"] = TODAY
        self.assertFalse(has(self.after(change), "P2", "홍하늘"))

    def test_regular_consult_and_follow_up(self):
        def change(r):
            r["contacts"].append(self.contact("심슬기", "정기상담"))
            for c in r["contacts"]:
                if c["name"] == "하누리" and c.get("follow_up"):
                    c["follow_done"] = "Y"
        tasks = self.after(change)
        self.assertFalse(has(tasks, "C1", "심슬기"))
        self.assertFalse(has(tasks, "C3", "하누리"))

    def test_makeup_attended(self):
        tasks = self.after(lambda r: r["attendance"].append(
            {"date": TODAY - timedelta(days=1), "name": "서가온", "status": "보강", "reason": ""}))
        self.assertFalse(has(tasks, "A4", "서가온"))

    def test_lead_status_change(self):
        def change(r):
            r["leads"][0]["status"] = "회신완료"
            r["leads"][0]["last_contact"] = TODAY
        self.assertFalse(has(self.after(change), "L1"))

    def test_transition_is_seasonal(self):
        rows = sample.build(date(2026, 6, 15))
        self.assertFalse(has(run(rows, date(2026, 6, 15))[2], "T1"))


class CsvExports(unittest.TestCase):
    def test_folder_with_export_style_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            def write(name, header, *rows):
                with open(Path(tmp) / f"{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
                    csv.writer(f).writerows([header, *rows])
            write("학생", ["학생명", "학년", "요일", "시간", "레벨"], ["김하나", "초2", "월수금", "3시", "2.1"])
            write("독서기록", ["Student", "Date Taken", "Title", "Book Level", "% Correct", "Word Count"],
                  ["김하나", "11/13/2026", "Fly Guy", "1.5", "90%", "1,200"], ["없는학생", "11/13/2026", "x", "1", "80", "1"])
            data = load(tmp)
        s = data.students["김하나"]
        self.assertEqual((s.time, s.level, s.days), ("15:00", 2.1, frozenset({0, 2, 4})))
        self.assertEqual(data.reading[0]["pct"], 90)
        self.assertEqual(data.reading[0]["words"], 1200)
        self.assertEqual(data.reading[0]["date"], date(2026, 11, 13))
        self.assertEqual(len(data.issues), 1)

    def test_html_is_escaped(self):
        rows = sample.build(TODAY)
        rows["leads"][0]["name"] = "<script>alert(1)</script>"
        data, _, tasks = run(rows)
        html = render.briefing(data, tasks, todays_roster(data, tasks, TODAY), TODAY, CFG)
        self.assertNotIn("<script>alert(1)", html)


if __name__ == "__main__":
    unittest.main()
