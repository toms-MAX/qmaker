"""숫자 계산: 학생별 현황, 주간 지표, 월간 독서 리포트 재료."""

from __future__ import annotations

from datetime import date, timedelta

from .store import PRESENT, Data, Student


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def last_regular_consult(data: Data, name: str, kinds: list[str]) -> date | None:
    dates = [c["date"] for c in data.contacts if c["name"] == name and c["kind"] in kinds]
    return max(dates) if dates else None


def student_stats(data: Data, s: Student, today: date, cfg: dict) -> dict:
    """상담 브리프·이탈 점수·리포트가 함께 쓰는 학생 한 명의 현황."""
    window = cfg["attendance"]["window_days"]
    att = [r for r in data.of("attendance", s.name) if today - timedelta(days=window) < r["date"] <= today]
    absences = sum(1 for r in att if r["status"] == "결석")
    quizzes = [r for r in data.of("reading", s.name) if r["date"] <= today]
    scored = [r for r in quizzes if r["pct"] is not None]
    recent, prior = scored[-5:], scored[-10:-5]
    sr = [r for r in data.of("sr", s.name) if r["ge"] is not None and r["date"] <= today]
    return {
        "att_records": len(att),
        "absences": absences,
        "absence_rate": absences / len(att) if att else 0.0,
        "quizzes_14d": sum(1 for r in quizzes if r["date"] > today - timedelta(days=14)),
        "present_14d": sum(
            1 for r in att if r["status"] in PRESENT and r["date"] > today - timedelta(days=14)
        ),
        "books_30d": sum(1 for r in quizzes if r["date"] > today - timedelta(days=30)),
        "avg_recent": mean(r["pct"] for r in recent),
        "avg_prior": mean(r["pct"] for r in prior) if len(prior) >= 3 else None,
        "cum_books": len(quizzes),
        "cum_words": sum(r["words"] for r in quizzes),
        "last_quiz": quizzes[-1]["date"] if quizzes else None,
        "sr_first": sr[0] if sr else None,
        "sr_last": sr[-1] if sr else None,
        "last_consult": last_regular_consult(data, s.name, cfg["consult"]["regular_kinds"]),
    }


def brief_lines(s: Student, st: dict) -> list[str]:
    """전화하기 전에 읽는 한 장 요약. 관리자가 원장 없이 상담할 수 있게 한다."""
    lines = []
    if st["att_records"]:
        lines.append(f"최근 4주 출석 {st['att_records'] - st['absences']}/{st['att_records']}회")
    if st["cum_books"]:
        line = f"최근 30일 {st['books_30d']}권 · 누적 {st['cum_books']}권"
        if st["cum_words"]:
            line += f" · {st['cum_words']:,} 단어"
        lines.append(line)
    if st["avg_recent"] is not None:
        line = f"최근 5권 평균 정답률 {st['avg_recent']:.0f}%"
        if st["avg_prior"] is not None:
            diff = st["avg_recent"] - st["avg_prior"]
            line += f" (직전 5권 대비 {diff:+.0f}%p)"
        lines.append(line)
    if s.level is not None:
        lines.append(f"현재 리딩 레벨 {s.level:.1f}")
    if st["sr_last"]:
        line = f"SR {st['sr_last']['ge']:.1f} ({st['sr_last']['date']:%m/%d})"
        if st["sr_first"] and st["sr_first"] is not st["sr_last"]:
            line += f" ← 처음 {st['sr_first']['ge']:.1f}"
        lines.append(line)
    if s.memo:
        lines.append(f"메모: {s.memo}")
    return lines


# ── 원장 주간 리포트 ─────────────────────────────────────────────────

def _period(rows: list[dict], start: date, end: date, key: str = "date") -> list[dict]:
    return [r for r in rows if r.get(key) and start <= r[key] <= end]


def _att_rate(rows: list[dict]) -> float | None:
    return sum(1 for r in rows if r["status"] in PRESENT) / len(rows) if rows else None


def weekly(data: Data, today: date, cfg: dict) -> dict:
    """어제까지의 7일과 그 앞 7일을 비교한다. 오늘은 아직 기록이 없으므로 뺀다."""
    end = today - timedelta(days=1)
    start = end - timedelta(days=6)
    p_end, p_start = start - timedelta(days=1), start - timedelta(days=7)
    active = data.active_students()
    students = data.students.values()

    att, p_att = _period(data.attendance, start, end), _period(data.attendance, p_start, p_end)
    rd, p_rd = _period(data.reading, start, end), _period(data.reading, p_start, p_end)

    month = f"{today.year}-{today.month:02d}"
    bills = [p for p in data.payments if p["month"] == month]
    billed = sum(p["amount"] for p in bills)
    paid = sum(p["amount"] for p in bills if p["paid"])
    overdue = [p for p in data.payments if not p["paid"] and p["due"] < today]

    interval = cfg["consult"]["interval_days"]
    kinds = cfg["consult"]["regular_kinds"]
    covered = 0
    for s in active:
        base = last_regular_consult(data, s.name, kinds) or s.enrolled
        if base and (today - base).days <= interval:
            covered += 1

    leads30 = _period(data.leads, today - timedelta(days=29), today)
    tested = [l for l in leads30 if l["status"] in {"진단완료", "등록"} or (l["test_date"] and l["test_date"] <= today)]
    enrolled30 = [l for l in leads30 if l["status"] == "등록"]

    contacts = _period(data.contacts, start, end)
    by: dict[str, int] = {}
    for c in contacts:
        by[c["by"] or "미기재"] = by.get(c["by"] or "미기재", 0) + 1

    t = cfg["transition"]
    pipeline = [s for s in active if s.grade == t["grade"] and t["course"] in s.course]

    return {
        "start": start, "end": end,
        "active": len(active),
        "by_course": _count(s.course for s in active),
        "new": sorted(s.name for s in students if s.enrolled and start <= s.enrolled <= end),
        "left": sorted(s.name for s in students if s.left and start <= s.left <= end),
        "monthly_revenue": sum(s.fee for s in active),
        "att_rate": _att_rate(att), "att_rate_prev": _att_rate(p_att),
        "quizzes": len(rd), "quizzes_prev": len(p_rd),
        "avg_pct": mean(r["pct"] for r in rd), "avg_pct_prev": mean(r["pct"] for r in p_rd),
        "words": sum(r["words"] for r in rd),
        "billed": billed, "paid": paid,
        "collect_rate": paid / billed if billed else None,
        "overdue_amount": sum(p["amount"] for p in overdue), "overdue_count": len(overdue),
        "consult_cover": covered / len(active) if active else None,
        "leads": len(leads30), "leads_tested": len(tested), "leads_enrolled": len(enrolled30),
        "contacts": len(contacts), "contacts_by": by,
        "pipeline": len(pipeline),
    }


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


# ── 학부모 월간 독서 리포트 ──────────────────────────────────────────

def monthly(data: Data, s: Student, year: int, month: int) -> dict:
    start = date(year, month, 1)
    end = (date(year + month // 12, month % 12 + 1, 1)) - timedelta(days=1)
    books = _period(data.of("reading", s.name), start, end)
    all_books = [r for r in data.of("reading", s.name) if r["date"] <= end]
    att = _period(data.of("attendance", s.name), start, end)
    sr = [r for r in data.of("sr", s.name) if r["ge"] is not None and r["date"] <= end]
    prev_end = start - timedelta(days=1)
    prev = _period(data.of("reading", s.name), date(prev_end.year, prev_end.month, 1), prev_end)
    return {
        "student": s, "year": year, "month": month,
        "books": books,
        "count": len(books), "count_prev": len(prev),
        "words": sum(r["words"] for r in books),
        "avg_pct": mean(r["pct"] for r in books),
        "avg_pct_prev": mean(r["pct"] for r in prev),
        "avg_bl": mean(r["bl"] for r in books),
        "cum_books": len(all_books), "cum_words": sum(r["words"] for r in all_books),
        "present": sum(1 for r in att if r["status"] in PRESENT), "sessions": len(att),
        "sr_first": sr[0] if sr else None, "sr_last": sr[-1] if sr else None,
    }


def default_comment(m: dict) -> str:
    """규칙으로 쓰는 기본 코멘트. Claude가 /toms-monthly에서 더 다듬는다."""
    s, parts = m["student"], []
    if m["count"] == 0:
        return f"{s.name} 학생은 이번 달 퀴즈 기록이 없습니다. 다음 달에는 책 고르기부터 함께 하겠습니다."
    parts.append(f"{s.name} 학생은 이번 달 원서 {m['count']}권을 읽었습니다.")
    if m["count_prev"] and m["count"] > m["count_prev"]:
        parts.append(f"지난달보다 {m['count'] - m['count_prev']}권 더 읽었습니다.")
    avg = m["avg_pct"]
    if avg is not None:
        if avg >= 85:
            parts.append(f"평균 정답률 {avg:.0f}%로 내용을 정확하게 이해하며 읽고 있습니다.")
        elif avg >= 70:
            parts.append(f"평균 정답률 {avg:.0f}%로 안정적으로 읽고 있습니다.")
        else:
            parts.append(
                f"평균 정답률이 {avg:.0f}%여서, 다음 달에는 조금 편한 레벨의 책으로 정확하게 읽는 연습을 늘리겠습니다."
            )
    first, last = m["sr_first"], m["sr_last"]
    if first and last and last["ge"] > first["ge"]:
        parts.append(f"리딩 레벨(SR)은 처음 {first['ge']:.1f}에서 {last['ge']:.1f}로 올랐습니다.")
    return " ".join(parts)
