"""HTML 출력: 관리자 브리핑, 원장 주간 리포트, 학부모 월간 독서 리포트.

파일 하나로 열리고(외부 자원 없음), 태블릿에서 보고 A4로 인쇄할 수 있다.
"""

from __future__ import annotations

from datetime import date
from html import escape as e

from .metrics import default_comment
from .rules import PRIORITY_LABEL, TODAY, URGENT, WEEK, Task
from .store import WEEKDAYS, Data, Student

CSS = """
:root{--navy:#1f2a44;--gold:#b8935a;--bg:#f5f4f0;--card:#fff;--ink:#1c1f26;--sub:#4b5563;
--muted:#6b7280;--line:#e3e1da;--soft:#f0eee8;--crit:#d03b3b;--warn:#c98a00;--good:#0c8a0c}
@media (prefers-color-scheme:dark){:root{--bg:#15171c;--card:#1e2128;--ink:#ecebe7;--sub:#c3c6cc;
--muted:#9aa0aa;--line:#2f333c;--soft:#262a32;--crit:#ef6a6a;--warn:#fab219;--good:#3fbf3f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.55;
font-family:Pretendard,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",system-ui,sans-serif;font-size:15px}
.wrap{max-width:980px;margin:0 auto;padding:0 16px 48px}
header.top{background:var(--navy);color:#fff;padding:22px 0 18px;border-bottom:3px solid var(--gold);margin-bottom:20px}
header.top .wrap{padding-bottom:0}
header.top .brand{font-size:13px;letter-spacing:.08em;color:#d9c6a5}
header.top h1{margin:2px 0 0;font-size:24px;font-weight:700}
header.top .meta{margin-top:4px;font-size:14px;color:#cdd3e0}
h2{font-size:17px;margin:28px 0 10px;padding-left:10px;border-left:4px solid var(--gold)}
h2 small{font-weight:400;color:var(--muted);font-size:13px;margin-left:6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .label{font-size:13px;color:var(--muted)}
.tile .value{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.25}
.tile .value small{font-size:14px;font-weight:500;color:var(--sub);margin-left:2px}
.tile .delta{font-size:12.5px;color:var(--sub)}
.up{color:var(--good)}.down{color:var(--crit)}.flat{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:10px}
.slot{display:grid;grid-template-columns:64px 1fr;gap:10px;padding:8px 0;border-top:1px solid var(--line)}
.slot:first-child{border-top:0}
.slot .time{font-weight:650;font-variant-numeric:tabular-nums;color:var(--sub)}
.slot>div{min-width:0}
.kid{display:inline-block;margin:0 14px 4px 0;max-width:100%}
.kid small{color:var(--muted)}
.note{display:inline-block;font-size:12px;border:1px solid var(--gold);color:var(--sub);border-radius:999px;padding:0 7px;margin-left:4px}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px}
.filters button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--sub);
border-radius:999px;padding:4px 12px;cursor:pointer}
.filters button[aria-pressed=true]{background:var(--navy);border-color:var(--navy);color:#fff}
.prio{font-size:13px;font-weight:650;color:var(--sub);margin:16px 0 6px}
.task{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:8px;break-inside:avoid}
.task.p0{border-left-color:var(--crit)}.task.p1{border-left-color:var(--gold)}
.task.done{opacity:.45}
.task .head{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.task .head input{width:17px;height:17px;margin:0;align-self:center}
.tag{font-size:12px;background:var(--soft);color:var(--sub);border-radius:5px;padding:1px 7px;white-space:nowrap}
.task .who{font-weight:700}
.task .title{font-weight:600}
.late{margin-left:auto;font-size:12.5px;color:var(--sub);white-space:nowrap}
.late b{color:var(--crit)}
.task dl{display:grid;grid-template-columns:52px 1fr;gap:3px 10px;margin:8px 0 0;font-size:14px}
.task dt{color:var(--muted)}.task dd{margin:0}
.task ul{margin:0;padding-left:18px}
details{margin-top:8px}
summary{cursor:pointer;font-size:13.5px;color:var(--sub)}
pre{white-space:pre-wrap;word-break:break-word;font:inherit;font-size:14px;background:var(--soft);border-radius:8px;padding:10px 12px;margin:6px 0 4px}
button.copy{font:inherit;font-size:12.5px;border:1px solid var(--line);background:var(--card);color:var(--sub);border-radius:6px;padding:2px 10px;cursor:pointer}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12.5px;color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.empty{color:var(--muted);font-size:14px}
ul.check{list-style:none;padding:0;margin:0}
ul.check li{padding:5px 0}
ul.check input{width:16px;height:16px;margin-right:8px;vertical-align:-2px}
.headline{font-size:16px;font-weight:600}
.foot{margin-top:28px;font-size:12.5px;color:var(--muted);text-align:center}
.report{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:26px 28px;margin-bottom:18px}
.report .rhead{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid var(--navy);padding-bottom:8px;margin-bottom:14px}
.report .rhead .name{font-size:22px;font-weight:700}
.report .rhead .brand{color:var(--gold);font-weight:600;font-size:13px;letter-spacing:.06em}
.comment{background:var(--soft);border-radius:8px;padding:12px 14px;margin-top:6px}
@media print{:root{--bg:#fff;--card:#fff;--ink:#111;--sub:#333;--muted:#555;--line:#ccc;--soft:#f3f3f3}
body{font-size:12.5px}.filters,button.copy,.noprint{display:none!important}
header.top{-webkit-print-color-adjust:exact;print-color-adjust:exact}
details>*{display:block}.report{border:0;padding:0;break-after:page}.task.done{opacity:1}}
@media (max-width:520px){.task dl{grid-template-columns:1fr}.task dt{margin-top:4px}.late{margin-left:0}}
"""

JS = """
(function(){
  var key='toms-'+document.body.dataset.day, saved={};
  try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(_){}
  document.querySelectorAll('input[data-k]').forEach(function(box){
    var row=box.closest('.task');
    if(saved[box.dataset.k]){box.checked=true;if(row)row.classList.add('done')}
    box.addEventListener('change',function(){
      saved[box.dataset.k]=box.checked;if(row)row.classList.toggle('done',box.checked);
      try{localStorage.setItem(key,JSON.stringify(saved))}catch(_){}
    });
  });
  document.querySelectorAll('button.copy').forEach(function(btn){
    btn.addEventListener('click',function(){
      var text=btn.parentElement.querySelector('pre').textContent;
      (navigator.clipboard?navigator.clipboard.writeText(text):Promise.reject()).then(function(){
        btn.textContent='복사됨';setTimeout(function(){btn.textContent='복사'},1500);
      },function(){
        var r=document.createRange();r.selectNodeContents(btn.parentElement.querySelector('pre'));
        var s=getSelection();s.removeAllRanges();s.addRange(r);
      });
    });
  });
  var chips=document.querySelectorAll('.filters button');
  chips.forEach(function(chip){
    chip.addEventListener('click',function(){
      chips.forEach(function(c){c.setAttribute('aria-pressed',c===chip)});
      var cat=chip.dataset.cat;
      document.querySelectorAll('#tasks .task').forEach(function(t){
        t.style.display=(!cat||t.dataset.cat===cat)?'':'none';
      });
    });
  });
})();
"""


def _page(title: str, heading: str, meta: str, body: str, center: str, day: date) -> str:
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{e(title)}</title><style>{CSS}</style></head>"
        f'<body data-day="{day.isoformat()}"><header class="top"><div class="wrap">'
        f'<div class="brand">{e(center)}</div><h1>{e(heading)}</h1><div class="meta">{e(meta)}</div>'
        f'</div></header><div class="wrap">{body}'
        '<div class="foot">탐스는 한 명 한 명을 생각합니다</div></div>'
        f"<script>{JS}</script></body></html>"
    )


def _day(d: date) -> str:
    return f"{d:%Y-%m-%d} ({WEEKDAYS[d.weekday()]})"


def _tile(label: str, value: str, unit: str = "", delta: str = "") -> str:
    unit_html = f"<small>{e(unit)}</small>" if unit else ""
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return f'<div class="tile"><div class="label">{e(label)}</div><div class="value">{e(value)}{unit_html}</div>{delta_html}</div>'


def _task_html(t: Task, today: date, checkbox: bool = True) -> str:
    age = t.age(today)
    late = ""
    if t.overdue(today):
        late = f'<span class="late"><b>지연</b> · {age}일째 (기한 {t.sla}일)</span>'
    elif age > 0:
        late = f'<span class="late">{age}일째</span>'
    key = e(f"{t.rule}|{t.student}|{t.since.isoformat()}")
    box = f'<input type="checkbox" data-k="{key}" aria-label="처리함">' if checkbox else ""
    who = f'<span class="who">{e(t.student)}</span>' if t.student else ""
    rows = f"<dt>근거</dt><dd>{e(t.reason)}</dd><dt>할 일</dt><dd>{e(t.action)}</dd>"
    if t.done_when:
        rows += f"<dt>완료</dt><dd>{e(t.done_when)}</dd>"
    extra = ""
    if t.brief:
        items = "".join(f"<li>{e(line)}</li>" for line in t.brief)
        extra += f"<details><summary>통화 전 브리프</summary><ul>{items}</ul></details>"
    if t.message:
        extra += (
            f"<details><summary>문자 초안</summary><div><pre>{e(t.message)}</pre>"
            '<button class="copy" type="button">복사</button></div></details>'
        )
    return (
        f'<div class="task p{t.priority}" data-cat="{e(t.category)}"><div class="head">{box}'
        f'<span class="tag">{e(t.category)}</span>{who}<span class="title">{e(t.title)}</span>{late}</div>'
        f"<dl>{rows}</dl>{extra}</div>"
    )


def briefing(
    data: Data, tasks: list[Task], roster: list, today: date, cfg: dict
) -> str:
    owner, manager = cfg["center"]["owner"], cfg["center"]["manager"]
    mine = [t for t in tasks if t.assignee != owner]
    theirs = [t for t in tasks if t.assignee == owner]
    coming = sum(len(kids) for _, kids in roster)
    count = lambda p: sum(1 for t in mine if t.priority == p)  # noqa: E731
    late = sum(1 for t in mine if t.overdue(today))

    body = '<div class="tiles">' + "".join([
        _tile("오늘 등원", str(coming), "명"),
        _tile("긴급", str(count(URGENT)), "건"),
        _tile("오늘 안에", str(count(TODAY)), "건"),
        _tile("이번 주", str(count(WEEK)), "건"),
        _tile(f"{owner}에게 전달", str(len(theirs)), "건"),
        _tile("기한 넘긴 일", str(late), "건"),
    ]) + "</div>"

    body += "<h2>오늘 등원 <small>시간대별 · 눈여겨볼 점</small></h2><div class='card'>"
    if roster:
        for time, kids in roster:
            chips = ""
            for s, notes in kids:
                level = f" {s.level:.1f}" if s.level is not None else ""
                tags = "".join(f'<span class="note">{e(n)}</span>' for n in notes)
                chips += f'<span class="kid">{e(s.name)} <small>{e(s.grade)}{level}</small>{tags}</span>'
            body += f'<div class="slot"><div class="time">{e(time)}</div><div>{chips}</div></div>'
    else:
        body += '<div class="empty">오늘 등원 예정 학생이 없습니다.</div>'
    body += "</div>"

    body += f"<h2>{e(manager)}가 할 일 <small>위에서부터 순서대로 · 처리하면 운영시트에 기록</small></h2>"
    cats = sorted({t.category for t in mine})
    if mine:
        body += '<div class="filters noprint"><button type="button" aria-pressed="true" data-cat="">전체</button>'
        body += "".join(f'<button type="button" aria-pressed="false" data-cat="{e(c)}">{e(c)}</button>' for c in cats)
        body += '</div><div id="tasks">'
        for p in (URGENT, TODAY, WEEK):
            group = [t for t in mine if t.priority == p]
            if group:
                body += f'<div class="prio">{PRIORITY_LABEL[p]} · {len(group)}건</div>'
                body += "".join(_task_html(t, today) for t in group)
        body += "</div>"
    else:
        body += '<div class="card empty">오늘 할 일이 없습니다. 마감 체크리스트만 확인하세요.</div>'

    body += f"<h2>{e(owner)}에게 전달 <small>직접 처리하지 말고 그대로 전달</small></h2>"
    body += "".join(_task_html(t, today, checkbox=False) for t in theirs) or (
        f'<div class="card empty">{e(owner)}에게 넘길 일이 없습니다.</div>'
    )

    body += "<h2>데이터 점검 <small>고치지 않으면 브리핑이 틀어집니다</small></h2>"
    if data.issues:
        rows = "".join(
            f"<tr><td>{e(i.table)}</td><td class='num'>{i.row or ''}</td><td>{e(i.message)}</td></tr>"
            for i in data.issues
        )
        body += f'<div class="card"><table><tr><th>시트</th><th class="num">줄</th><th>문제</th></tr>{rows}</table></div>'
    else:
        body += '<div class="card empty">입력 오류가 없습니다.</div>'

    items = "".join(
        f'<li><label><input type="checkbox" data-k="close{n}">{e(text)}</label></li>'
        for n, text in enumerate(cfg["closing"]["checklist"])
    )
    body += f'<h2>마감 체크리스트</h2><div class="card"><ul class="check">{items}</ul></div>'

    return _page(
        f"탐스 브리핑 {today:%m-%d}", "오늘의 브리핑", f"{_day(today)} · {manager}용",
        body, cfg["center"]["name"], today,
    )


# ── 원장 주간 리포트 ─────────────────────────────────────────────────

def _delta(now, prev, unit: str, up_is_good: bool = True, scale: float = 1.0, digits: int = 0) -> str:
    if now is None or prev is None:
        return '<span class="flat">–</span> 지난주 기록 없음'
    diff = (now - prev) * scale
    if abs(diff) < 10 ** -digits / 2:
        return '<span class="flat">■</span> 지난주와 같음'
    good = (diff > 0) == up_is_good
    arrow = "▲" if diff > 0 else "▼"
    return f'<span class="{"up" if good else "down"}">{arrow}</span> 지난주 대비 {diff:+.{digits}f}{e(unit)}'


def _won(amount: int) -> str:
    return f"{amount / 10000:,.0f}"


def weekly_report(k: dict, tasks: list[Task], churn: list[tuple[int, Student, list[str]]], today: date, cfg: dict) -> str:
    owner, manager = cfg["center"]["owner"], cfg["center"]["manager"]
    theirs = [t for t in tasks if t.assignee == owner]
    mine = [t for t in tasks if t.assignee != owner]
    late = [t for t in mine if t.overdue(today)]

    headline = f"{owner} 직접 처리 {len(theirs)}건 · {manager} 지연 {len(late)}건"
    if k["overdue_amount"]:
        headline += f" · 미수금 {_won(k['overdue_amount'])}만원"
    body = f'<div class="card headline">{e(headline)}</div>'

    pct = lambda v: f"{v * 100:.0f}" if v is not None else "–"  # noqa: E731
    conv = f"{k['leads_enrolled']}/{k['leads']}" if k["leads"] else "–"
    body += '<div class="tiles">' + "".join([
        _tile("재원생", str(k["active"]), "명", f"신규 {len(k['new'])} · 퇴원 {len(k['left'])}"),
        _tile("출석률", pct(k["att_rate"]), "%", _delta(k["att_rate"], k["att_rate_prev"], "%p", scale=100)),
        _tile("주간 독서", str(k["quizzes"]), "권", _delta(k["quizzes"], k["quizzes_prev"], "권")),
        _tile("평균 정답률", f"{k['avg_pct']:.0f}" if k["avg_pct"] is not None else "–", "%",
              _delta(k["avg_pct"], k["avg_pct_prev"], "%p")),
        _tile("이번 달 수납률", pct(k["collect_rate"]), "%", f"청구 {_won(k['billed'])}만 · 입금 {_won(k['paid'])}만"),
        _tile("미수금", _won(k["overdue_amount"]), "만원", f"기한 경과 {k['overdue_count']}건"),
        _tile("상담 커버리지", pct(k["consult_cover"]), "%", f"최근 {cfg['consult']['interval_days']}일 안에 상담한 비율"),
        _tile("문의 → 등록", conv, "", f"최근 30일 · 진단 {k['leads_tested']}건"),
    ]) + "</div>"

    body += f"<h2>{e(owner)}이 직접 할 일 <small>시스템이 {e(manager)} 권한 밖으로 판단한 건</small></h2>"
    body += "".join(_task_html(t, today, checkbox=False) for t in theirs) or (
        '<div class="card empty">이번 주는 직접 나설 일이 없습니다.</div>'
    )

    body += f"<h2>{e(manager)} 처리 현황 <small>열려 있는 일 {len(mine)}건</small></h2><div class='card'>"
    cats: dict[str, list[int]] = {}
    for t in mine:
        row = cats.setdefault(t.category, [0, 0])
        row[0] += 1
        row[1] += t.overdue(today)
    if cats:
        rows = "".join(
            f"<tr><td>{e(c)}</td><td class='num'>{n}</td><td class='num'>{o}</td></tr>"
            for c, (n, o) in sorted(cats.items())
        )
        body += f"<table><tr><th>분류</th><th class='num'>열린 일</th><th class='num'>기한 넘김</th></tr>{rows}</table>"
    else:
        body += '<div class="empty">열려 있는 일이 없습니다.</div>'
    body += "</div>"
    if late:
        rows = "".join(
            f"<tr><td>{e(t.category)}</td><td>{e(t.student)}</td><td>{e(t.title)}</td>"
            f"<td class='num'>{t.age(today)}일</td><td class='num'>{t.sla}일</td></tr>"
            for t in sorted(late, key=lambda t: -t.age(today))
        )
        body += (
            "<div class='card'><table><tr><th>분류</th><th>학생</th><th>기한 넘긴 일</th>"
            f"<th class='num'>경과</th><th class='num'>기한</th></tr>{rows}</table></div>"
        )

    body += "<h2>이탈 위험 순위 <small>점수가 높은 순 · 상위 5명</small></h2><div class='card'>"
    if churn:
        rows = "".join(
            f"<tr><td>{e(s.name)}</td><td>{e(s.grade)} · {e(s.course)}</td><td class='num'>{score}</td>"
            f"<td>{e(' · '.join(why))}</td></tr>"
            for score, s, why in churn[:5]
        )
        body += f"<table><tr><th>학생</th><th>과정</th><th class='num'>점수</th><th>신호</th></tr>{rows}</table>"
    else:
        body += '<div class="empty">위험 신호가 있는 학생이 없습니다.</div>'
    body += "</div>"

    courses = " · ".join(f"{c} {n}명" for c, n in k["by_course"].items())
    by = " · ".join(f"{who} {n}건" for who, n in sorted(k["contacts_by"].items())) or "없음"
    facts = [
        ("과정별 인원", courses or "–"),
        ("월 교습비 합계(재원 기준)", f"{_won(k['monthly_revenue'])}만원"),
        ("이번 주 신규", ", ".join(k["new"]) or "없음"),
        ("이번 주 퇴원", ", ".join(k["left"]) or "없음"),
        ("이번 주 학부모 연락", f"{k['contacts']}건 ({by})"),
        ("주간 읽은 단어", f"{k['words']:,}"),
        (f"전환 대상({cfg['transition']['grade']} 도서관)", f"{k['pipeline']}명"),
    ]
    rows = "".join(f"<tr><th>{e(a)}</th><td>{e(b)}</td></tr>" for a, b in facts)
    body += f"<h2>재원·운영 현황</h2><div class='card'><table>{rows}</table></div>"

    return _page(
        f"탐스 주간 리포트 {today:%m-%d}", "주간 리포트",
        f"{k['start']:%m/%d} ~ {k['end']:%m/%d} · {owner}용", body, cfg["center"]["name"], today,
    )


# ── 학부모 월간 독서 리포트 ──────────────────────────────────────────

def monthly_report(reports: list[dict], comments: dict[str, str], cfg: dict, today: date) -> str:
    body = (
        '<div class="card noprint">학생 한 명이 A4 한 장입니다. 인쇄(Ctrl+P)하면 학생별로 나뉘어 나옵니다. '
        "코멘트는 보내기 전에 반드시 읽고 고쳐 주세요.</div>"
    )
    for m in reports:
        s: Student = m["student"]
        avg = f"{m['avg_pct']:.0f}" if m["avg_pct"] is not None else "–"
        attend = f"{m['present']}/{m['sessions']}" if m["sessions"] else "–"
        tiles = "".join([
            _tile("이번 달 읽은 책", str(m["count"]), "권", f"지난달 {m['count_prev']}권"),
            _tile("읽은 단어", f"{m['words']:,}"),
            _tile("평균 정답률", avg, "%"),
            _tile("출석", attend, "회"),
        ])
        level = []
        if s.level is not None:
            level.append(f"현재 리딩 레벨 <b>{s.level:.1f}</b>")
        if m["sr_last"]:
            text = f"SR 진단 <b>{m['sr_last']['ge']:.1f}</b> ({m['sr_last']['date']:%Y-%m-%d})"
            if m["sr_first"] and m["sr_first"] is not m["sr_last"]:
                text += f" · 처음 {m['sr_first']['ge']:.1f}"
            level.append(text)
        level.append(f"누적 <b>{m['cum_books']}</b>권 · <b>{m['cum_words']:,}</b> 단어")
        rows = ""
        for r in m["books"]:
            bl = f"{r['bl']:.1f}" if r["bl"] is not None else "–"
            score = f"{r['pct']:.0f}%" if r["pct"] is not None else "–"
            rows += (
                f"<tr><td>{r['date']:%m/%d}</td><td>{e(r['title'])}</td><td class='num'>{bl}</td>"
                f"<td class='num'>{score}</td><td class='num'>{r['words']:,}</td></tr>"
            )
        books = (
            "<table><tr><th>날짜</th><th>책</th><th class='num'>북레벨</th><th class='num'>정답률</th>"
            f"<th class='num'>단어 수</th></tr>{rows}</table>"
            if rows else '<div class="empty">이번 달 퀴즈 기록이 없습니다.</div>'
        )
        comment = comments.get(s.name) or default_comment(m)
        body += (
            f'<section class="report"><div class="rhead"><div><div class="name">{e(s.name)}</div>'
            f"<div>{e(s.grade)} · {m['year']}년 {m['month']}월 독서 리포트</div></div>"
            f'<div class="brand">{e(cfg["center"]["name"])}</div></div>'
            f'<div class="tiles">{tiles}</div>'
            f'<p>{" · ".join(level)}</p>'
            f"<h2>이번 달 읽은 책</h2>{books}"
            f'<h2>선생님 코멘트</h2><div class="comment">{e(comment)}</div></section>'
        )
    year, month = (reports[0]["year"], reports[0]["month"]) if reports else (today.year, today.month)
    return _page(
        f"탐스 월간 독서 리포트 {year}-{month:02d}", "월간 독서 리포트",
        f"{year}년 {month}월 · 학부모 발송용 · {len(reports)}명", body, cfg["center"]["name"], today,
    )
