"""명령줄 진입점.

    python -m toms_ops demo                 데모 데이터로 세 화면을 모두 만들어 본다
    python -m toms_ops brief                오늘의 브리핑 (관리자용)
    python -m toms_ops weekly               주간 리포트 (원장용)
    python -m toms_ops monthly --month 2026-11   학부모 월간 독서 리포트
    python -m toms_ops check                데이터 입력 오류만 점검
    python -m toms_ops template             빈 운영시트(.xlsx) 만들기
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from . import metrics, render, sample
from .rules import Engine, todays_roster
from .store import Data, load, load_config, parse_date
from .workbook import write_workbook

ROOT = Path(__file__).resolve().parent.parent


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="toms_ops", description="탐스 운영 OS")
    p.add_argument("command", choices=["demo", "brief", "weekly", "monthly", "check", "template", "tasks"])
    p.add_argument("--data", default=str(ROOT / "data" / "운영시트.xlsx"), help="운영시트(.xlsx) 또는 CSV 폴더")
    p.add_argument("--config", default=str(ROOT / "config"), help="rules.toml이 있는 폴더")
    p.add_argument("--out", default=str(ROOT / "out"), help="결과를 저장할 폴더")
    p.add_argument("--today", help="기준일 (YYYY-MM-DD). 비우면 오늘")
    p.add_argument("--month", help="월간 리포트 대상 월 (YYYY-MM). 비우면 지난달")
    p.add_argument("--comments", help="월간 리포트 코멘트 JSON ({이름: 코멘트})")
    return p.parse_args()


def _write(out: Path, name: str, html: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(html, encoding="utf-8")
    return path


def _churn_rank(engine: Engine, data: Data) -> list:
    ranked = [(score, s, why) for s in data.active_students() for score, why in [engine.churn_score(s)] if score > 0]
    return sorted(ranked, key=lambda x: -x[0])


def _summary(tasks, issues, today, owner) -> str:
    mine = [t for t in tasks if t.assignee != owner]
    return (
        f"할 일 {len(mine)}건 (긴급 {sum(t.priority == 0 for t in mine)}, 기한 넘김 {sum(t.overdue(today) for t in mine)}) · "
        f"{owner} 전달 {len(tasks) - len(mine)}건 · 데이터 점검 {len(issues)}건"
    )


def _month(args, today: date) -> tuple[int, int]:
    if args.month:
        year, month = args.month.split("-")[:2]
        return int(year), int(month)
    return (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)


def main() -> int:
    args = _args()
    out = Path(args.out)
    today = parse_date(args.today) if args.today else date.today()
    if today is None:
        print(f"기준일을 읽을 수 없습니다: {args.today}", file=sys.stderr)
        return 2

    if args.command == "template":
        path = write_workbook(out / "탐스_운영시트_템플릿.xlsx")
        print(f"빈 운영시트를 만들었습니다: {path}")
        return 0

    if args.command == "demo":
        today = sample.demo_today(today)
        args.data = str(write_workbook(out / "데모_운영시트.xlsx", sample.build(today)))
        print(f"데모 운영시트: {args.data}")

    source = Path(args.data)
    if not source.exists():
        print(
            f"운영시트가 없습니다: {source}\n"
            "  처음이라면  python -m toms_ops demo      (데모로 먼저 보기)\n"
            "            python -m toms_ops template  (빈 운영시트 만들기)",
            file=sys.stderr,
        )
        return 2

    cfg, templates = load_config(args.config)
    data = load(source, Path(args.config) / "columns.toml")
    engine = Engine(data, cfg, templates, today)
    tasks = engine.run()
    owner = cfg["center"]["owner"]

    if args.command == "check":
        for i in data.issues:
            print(f"[{i.table}] {i.row or '-'}줄: {i.message}")
        print(f"데이터 점검 {len(data.issues)}건 · 학생 {len(data.active_students())}명(재원)")
        return 1 if data.issues else 0

    if args.command == "tasks":
        payload = [
            {**{k: getattr(t, k) for k in ("rule", "category", "assignee", "student", "title", "reason", "action", "sla")},
             "priority": t.priority, "since": t.since.isoformat(), "age": t.age(today), "overdue": t.overdue(today)}
            for t in tasks
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    made = []
    if args.command in {"brief", "demo"}:
        html = render.briefing(data, tasks, todays_roster(data, tasks, today), today, cfg)
        made.append(_write(out, f"브리핑_{today.isoformat()}.html", html))
    if args.command in {"weekly", "demo"}:
        html = render.weekly_report(metrics.weekly(data, today, cfg), tasks, _churn_rank(engine, data), today, cfg)
        made.append(_write(out, f"주간리포트_{today.isoformat()}.html", html))
    if args.command in {"monthly", "demo"}:
        year, month = (today.year, today.month) if args.command == "demo" and not args.month else _month(args, today)
        comments = json.loads(Path(args.comments).read_text(encoding="utf-8")) if args.comments else {}
        students = [s for s in data.active_students() if "도서관" in s.course]
        reports = [metrics.monthly(data, s, year, month) for s in students]
        if args.command == "monthly":
            pack = [
                {"이름": m["student"].name, "학년": m["student"].grade, "권수": m["count"], "지난달권수": m["count_prev"],
                 "평균정답률": round(m["avg_pct"]) if m["avg_pct"] is not None else None, "단어수": m["words"],
                 "누적권수": m["cum_books"], "출석": f"{m['present']}/{m['sessions']}",
                 "메모": m["student"].memo, "기본코멘트": metrics.default_comment(m)}
                for m in reports
            ]
            pack_path = out / f"월간리포트_{year}-{month:02d}_재료.json"
            out.mkdir(parents=True, exist_ok=True)
            pack_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
            made.append(pack_path)
        made.append(_write(out, f"월간리포트_{year}-{month:02d}.html", render.monthly_report(reports, comments, cfg, today)))

    for path in made:
        print(f"만들었습니다: {path}")
    print(_summary(tasks, data.issues, today, owner))
    return 0


if __name__ == "__main__":
    sys.exit(main())
