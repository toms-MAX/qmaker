"""데모 데이터. 모두 지어낸 이름이며, 규칙마다 한 명씩 걸리도록 상황을 심었다."""

from __future__ import annotations

import random
from datetime import date, timedelta

OPEN = {0, 1, 2, 3, 4}
D = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4}

# 이름, 학년, 과정, 요일, 시간, 레벨, 결제일, 원비
ROSTER = [
    ("서가온", "초2", "도서관", "월수금", "14:10", 1.8, 1, 180000),
    ("윤나래", "초3", "도서관", "월수금", "14:10", 2.4, 1, 180000),
    ("배다솜", "초1", "도서관", "월수", "14:10", 1.2, 5, 140000),
    ("임라온", "초3", "도서관", "월수금", "15:00", 2.6, 5, 180000),
    ("고마루", "초2", "도서관", "월수금", "15:00", 2.2, 5, 180000),
    ("문바다", "초3", "도서관", "월금", "15:00", 2.8, 10, 140000),
    ("손사랑", "초2", "도서관", "월수금", "15:30", 1.6, 10, 180000),
    ("양아람", "초4", "도서관", "월수금", "15:30", 3.4, 10, 180000),
    ("권자람", "초4", "도서관", "월수금", "15:30", 3.1, 15, 180000),
    ("홍하늘", "초1", "도서관", "화목", "14:10", 1.0, 12, 140000),
    ("전한결", "초2", "도서관", "화목", "14:10", 1.9, 7, 140000),
    ("남보람", "초3", "도서관", "화목", "15:00", 2.5, 1, 140000),
    ("심슬기", "초2", "도서관", "화목", "15:00", 2.0, 20, 140000),
    ("노이든", "초1", "도서관", "월수", "15:00", 0.8, 8, 140000),
    ("하누리", "초3", "도서관", "화목금", "15:30", 2.7, 20, 180000),
    ("곽도담", "초4", "도서관", "화목", "15:30", 3.6, 20, 140000),
    ("성새론", "초4", "도서관", "월수", "15:30", 3.3, 25, 140000),
    ("차여울", "초3", "도서관", "월수금", "14:10", 2.3, 25, 180000),
    ("주온유", "초1", "도서관", "화목", "14:10", 1.1, 25, 140000),
    ("우소담", "초2", "도서관", "월수금", "15:00", 1.7, 15, 180000),
    ("구다온", "초5", "입시", "월목", "16:30", None, 15, 280000),
    ("민해솔", "초6", "입시", "월목", "16:30", None, 1, 280000),
    ("류예솔", "중1", "입시", "화금", "18:30", None, 5, 300000),
    ("나은솔", "중2", "입시", "화금", "18:30", None, 10, 300000),
]

TITLES = [
    "Fly Guy", "Biscuit", "Henry and Mudge", "Frog and Toad", "Nate the Great", "Mercy Watson",
    "Magic Tree House", "Junie B. Jones", "Arthur Chapter Book", "Marvin Redpost", "Horrid Henry",
    "Flat Stanley", "Cam Jansen", "A to Z Mysteries", "Zack Files", "Geronimo Stilton",
]


def demo_today(today: date) -> date:
    """주말이면 다음 월요일로 — 데모 화면에 등원 명단이 비지 않게."""
    while today.weekday() not in OPEN:
        today += timedelta(days=1)
    return today


def build(today: date, seed: int = 7) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    opened = today - timedelta(days=76)
    students, attendance, reading, sr, payments, contacts, leads = [], [], [], [], [], [], []

    def sched(name: str) -> list[date]:
        """그 학생의 지난 등원일(오늘 제외), 최근 순."""
        s = by_name[name]
        days = {D[ch] for ch in s["days"]}
        out, day = [], today - timedelta(days=1)
        while day >= s["enrolled"]:
            if day.weekday() in days:
                out.append(day)
            day -= timedelta(days=1)
        return out

    for name, grade, course, days, time, level, bill_day, fee in ROSTER:
        students.append({
            "name": name, "grade": grade, "course": course, "days": days, "time": time,
            "enrolled": opened, "status": "재원", "phone": "010-0000-0000",
            "level": level, "bill_day": bill_day, "fee": fee, "memo": "",
        })
    by_name = {s["name"]: s for s in students}
    by_name["노이든"]["enrolled"] = today - timedelta(days=8)
    by_name["양아람"]["enrolled"] = today - timedelta(days=100)
    by_name["권자람"]["enrolled"] = today - timedelta(days=100)
    by_name["양아람"]["memo"] = "인수 전부터 다닌 학생. 판타지 시리즈를 좋아함"
    students.append({
        "name": "탁한별", "grade": "초2", "course": "도서관", "days": "화목", "time": "15:00",
        "enrolled": opened, "status": "퇴원", "left": today - timedelta(days=4),
        "phone": "010-0000-0000", "level": 1.5, "bill_day": 1, "fee": 140000, "memo": "이사",
    })

    # 기본 흐름: 꾸준히 출석하고, 등원일의 85%쯤 퀴즈를 친다.
    for s in students:
        if s["status"] != "재원":
            continue
        name, level = s["name"], s["level"]
        for day in sched(name):
            attendance.append({"date": day, "name": name, "status": "출석", "reason": ""})
            if level is not None and rng.random() < 0.85:
                bl = round(level + rng.choice([-0.5, -0.4, -0.3, -0.1, 0.0, 0.1]), 1)
                reading.append({
                    "date": day, "name": name,
                    "title": f"{rng.choice(TITLES)} #{rng.randint(1, 24)}",
                    "bl": max(0.5, bl), "pct": min(100, max(60, round(rng.gauss(82, 7) / 5) * 5)),
                    "words": max(300, int(level * 3000 + rng.randint(-500, 1500))),
                })
        if level is not None:
            sr.append({"date": s["enrolled"] + timedelta(days=2), "name": name, "ge": round(level - 0.2, 1)})
        if (today - s["enrolled"]).days > 30:
            contacts.append({
                "date": today - timedelta(days=rng.randint(12, 45)), "name": name, "kind": "정기상담",
                "method": "전화", "note": "근황 공유, 특이사항 없음", "by": "관리자",
            })
        # 지난 달까지는 모두 납부, 이번 달은 결제일이 지났으면 납부
        for back in (2, 1, 0):
            month = (today.month - back - 1) % 12 + 1
            year = today.year - (1 if month > today.month else 0)
            due = date(year, month, s["bill_day"])
            if due < s["enrolled"] or due > today:
                continue
            payments.append({
                "month": f"{year}-{month:02d}", "name": name, "amount": s["fee"],
                "due": due, "paid": due - timedelta(days=rng.randint(0, 2)),
            })

    def rows(table: list[dict], name: str) -> list[dict]:
        return sorted((r for r in table if r["name"] == name), key=lambda r: r.get("date") or r.get("due"))

    def absent(name: str, count: int, reason: str = ""):
        for day in sched(name)[:count]:
            for r in attendance:
                if r["name"] == name and r["date"] == day:
                    r["status"], r["reason"] = "결석", reason
            reading[:] = [r for r in reading if not (r["name"] == name and r["date"] == day)]

    def last_scores(name: str, scores: list[int], at_level: bool = False):
        mine = rows(reading, name)[-len(scores):]
        for r, pct in zip(mine, scores):
            r["pct"] = pct
            if at_level:
                r["bl"] = by_name[name]["level"]

    def unpaid(name: str, days_late: int):
        payments[:] = [p for p in payments if not (p["name"] == name and p["month"] == f"{today.year}-{today.month:02d}")]
        payments.append({
            "month": f"{today.year}-{today.month:02d}", "name": name, "amount": by_name[name]["fee"],
            "due": today - timedelta(days=days_late), "paid": None,
        })

    def drop_contacts(name: str):
        contacts[:] = [c for c in contacts if c["name"] != name]

    # ── 심어 둔 상황 ──
    absent("서가온", 1)                                   # A1 무단결석
    absent("윤나래", 2)                                   # A2 연속 2회
    absent("배다솜", 3)                                   # A3 연속 3회 → 원장
    for r in rows(reading, "임라온")[-10:]:               # R1 레벨업
        r["bl"], r["pct"] = by_name["임라온"]["level"], rng.choice([85, 90, 95, 100])
    last_scores("고마루", [55, 50, 60])                   # R2 레벨 조정
    last_scores("문바다", [55, 50, 45, 60, 50, 55])       # R3 부진 지속 → 원장
    cut = today - timedelta(days=14)                      # R4 독서량 부족
    reading[:] = [r for r in reading if not (r["name"] == "손사랑" and r["date"] > cut)]
    for r in sr:                                          # R5 SR 재진단
        if r["name"] == "양아람":
            r["date"] = today - timedelta(days=95)
    mine = rows(reading, "권자람")                        # G1 누적 50권
    reading[:] = [r for r in reading if r["name"] != "권자람"]
    while len(mine) < 51:
        mine.insert(0, {**mine[0], "date": by_name["권자람"]["enrolled"] + timedelta(days=len(mine) % 20)})
    mine = sorted(mine[-51:], key=lambda r: r["date"])
    mine[-2]["date"], mine[-1]["date"] = sched("권자람")[1], sched("권자람")[0]
    reading.extend(mine)
    unpaid("홍하늘", 4)                                   # P2 리마인드
    unpaid("전한결", 9)                                   # P3 전화
    unpaid("남보람", 15)                                  # P4 → 원장
    drop_contacts("심슬기")                               # C1 2개월 룰
    contacts.append({                                     # C3 후속 조치
        "date": today - timedelta(days=5), "name": "하누리", "kind": "정기상담", "method": "전화",
        "note": "금요일 시간 변경 가능한지 문의 → 확인 후 회신 약속", "follow_up": today - timedelta(days=1),
        "follow_done": "", "by": "관리자",
    })
    drop_contacts("차여울")                               # X1 이탈 위험
    days = sched("차여울")
    for day in (days[2], days[5], days[8]):
        for r in attendance:
            if r["name"] == "차여울" and r["date"] == day:
                r["status"], r["reason"] = "결석", "컨디션"
    attendance.append({"date": days[1], "name": "차여울", "status": "보강", "reason": ""})
    attendance.append({"date": days[0] - timedelta(days=1), "name": "차여울", "status": "보강", "reason": ""})
    reading[:] = [r for r in reading if not (r["name"] == "차여울" and r["date"] > cut)]
    last_scores("차여울", [75, 70, 65, 65, 60])
    attendance[:] = [                                     # D1 출결 미입력
        r for r in attendance if not (r["name"] == "우소담" and r["date"] == sched("우소담")[0])
    ]
    reading.append({                                      # 데이터 점검: 오타 난 이름
        "date": today - timedelta(days=3), "name": "서가은", "title": "Fly Guy #3", "bl": 1.5, "pct": 80, "words": 400,
    })

    leads.extend([
        {"date": today - timedelta(days=1), "name": "문의하나", "grade": "초2", "phone": "010-0000-0001",
         "source": "네이버 지도", "status": "신규문의"},
        {"date": today - timedelta(days=8), "name": "문의두리", "grade": "초1", "phone": "010-0000-0002",
         "source": "지인 소개", "status": "회신완료", "last_contact": today - timedelta(days=6)},
        {"date": today - timedelta(days=4), "name": "문의세찬", "grade": "초3", "phone": "010-0000-0003",
         "source": "블로그", "status": "진단예약", "test_date": today, "last_contact": today - timedelta(days=1)},
        {"date": today - timedelta(days=3), "name": "문의네온", "grade": "초5", "phone": "010-0000-0004",
         "source": "간판", "status": "진단예약", "test_date": today + timedelta(days=1),
         "last_contact": today - timedelta(days=3)},
        {"date": today - timedelta(days=9), "name": "문의다섯", "grade": "초4", "phone": "010-0000-0005",
         "source": "지인 소개", "status": "진단완료", "test_date": today - timedelta(days=4),
         "last_contact": today - timedelta(days=4)},
        {"date": today - timedelta(days=24), "name": "문의여섯", "grade": "중1", "phone": "010-0000-0006",
         "source": "블로그", "status": "진단완료", "test_date": today - timedelta(days=16),
         "last_contact": today - timedelta(days=9)},
        {"date": today - timedelta(days=15), "name": "노이든", "grade": "초1", "phone": "010-0000-0007",
         "source": "지인 소개", "status": "등록", "test_date": today - timedelta(days=11),
         "last_contact": today - timedelta(days=9)},
    ])

    return {
        "students": students, "attendance": attendance, "reading": reading, "sr": sr,
        "payments": payments, "contacts": contacts, "leads": leads,
    }
