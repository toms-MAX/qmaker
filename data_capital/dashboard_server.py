"""
DATA CAPITAL — 실시간 대시보드 서버
http://localhost:8765 에서 확인
"""
import json
import os
import re
import sys
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG_FILE = os.path.join(os.path.dirname(__file__), "trading.log")
PORT     = 9002
UTC      = timezone.utc

# ─── 로그 파싱 ────────────────────────────────────────────
def parse_log():
    if not os.path.exists(LOG_FILE):
        return [], [], {}, {}

    trades    = []
    live_rows = []
    summary   = {"daily_pnl_pct": 0.0, "open_positions": [], "halted": False}
    latest    = {"price": "—", "rsi": "—", "time": "—"}

    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception:
        return [], [], summary, latest

    # \r 제거 후 라인 분리
    lines = raw.replace("\r", "\n").splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # BUY 주문  ">>> [ORDER] 12:30:15 | momentum | 매수 | 96,760원 | 420,000원치"
        if "[ORDER]" in line and "매수" in line:
            m = re.search(
                r"\[ORDER\]\s+([\d:]+)\s+\|\s+(\S+)\s+\|\s+매수\s+\|\s+([\d,]+)원\s+\|\s+([\d,]+)원치",
                line,
            )
            if m:
                t, agent, price, amount = m.groups()
                trades.append({
                    "time": t, "agent": agent, "type": "BUY",
                    "result": "진입", "pnl_pct": "—",
                    "pnl_won": f"매수 {amount}원치 @ {price}원",
                    "daily": "—", "icon": "📈",
                })
                live_rows.append({"time": t, "text": f"📈 BUY [{agent}]  {price}원  {amount}원치", "cls": "entry"})

        # EXIT  "<<< [EXIT] momentum | STOP_LOSS | +0.12% | +460원 | 일일PnL: +0.015%"
        elif "[EXIT]" in line:
            m = re.search(
                r"\[EXIT\]\s+(\S+)\s+\|\s+(\S+)\s+\|\s+([+\-][.\d]+%)\s+\|\s+([+\-][\d,]+원)\s+\|\s+일일PnL:\s+([+\-][.\d]+%)",
                line,
            )
            if m:
                agent, result, pnl_pct, pnl_won, daily = m.groups()
                icon = "✅" if pnl_pct.startswith("+") else "❌"
                t = _extract_time(line)
                trades.append({
                    "time": t, "agent": agent, "type": "EXIT",
                    "result": result, "pnl_pct": pnl_pct,
                    "pnl_won": pnl_won, "daily": daily, "icon": icon,
                })
                cls = "exit-win" if pnl_pct.startswith("+") else "exit-loss"
                live_rows.append({"time": t, "text": f"{icon} EXIT [{agent}] {result}  {pnl_pct}  {pnl_won}", "cls": cls})
                try:
                    summary["daily_pnl_pct"] = float(daily.replace("%", ""))
                except Exception:
                    pass

        # Oracle  "  [Oracle] SPLIT | 단독 신호 ..."
        elif "[Oracle]" in line:
            m = re.search(r"\[Oracle\]\s+(\w+)\s+\|(.+)", line)
            if m:
                decision, reason = m.group(1), m.group(2).strip()[:55]
                t = _extract_time(line) or "—"
                live_rows.append({"time": t, "text": f"Oracle → {decision} | {reason}", "cls": "oracle"})

        # LIVE  "  [LIVE] 12:30:14 | 종가:96,760 | RSI:67.5 | 감시 중..."
        elif "[LIVE]" in line:
            m = re.search(r"\[LIVE\]\s+([\d:]+)\s+\|\s+종가:([\d,]+)\s+\|\s+RSI:([\d.]+)", line)
            if m:
                t, price, rsi = m.groups()
                latest = {"price": price, "rsi": rsi, "time": t}
                live_rows.append({"time": t, "text": f"KODEX200  {price}원  RSI {rsi}", "cls": "live"})

        # 일일 MDD 중단
        elif "MDD" in line and "중단" in line:
            summary["halted"] = True

    live_rows = live_rows[-300:]
    return trades, live_rows, summary, latest


def _extract_time(line: str) -> str:
    m = re.search(r"\b(\d{2}:\d{2}:\d{2})\b", line)
    return m.group(1) if m else "—"


# ─── HTML 대시보드 ────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DATA CAPITAL — Live Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e;
    --green: #3fb950; --red: #f85149; --blue: #58a6ff;
    --yellow: #d29922; --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', monospace; font-size: 14px; }

  header {
    background: var(--card); border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; align-items: center; gap: 16px;
  }
  header h1 { font-size: 18px; color: var(--blue); letter-spacing: 1px; }
  #status-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--green); animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  #last-update { font-size: 12px; color: var(--muted); margin-left: auto; }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px 24px; }
  @media(max-width:900px){ .grid { grid-template-columns: 1fr; } }

  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px;
  }
  .card h2 { font-size: 13px; color: var(--muted); text-transform: uppercase;
              letter-spacing: .8px; margin-bottom: 12px; }

  /* KPI 카드 */
  .kpi-row { display: flex; gap: 12px; padding: 0 24px 0; }
  .kpi {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px 20px; flex: 1; text-align: center;
  }
  .kpi .label { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
  .kpi .val   { font-size: 22px; font-weight: 700; }
  .pos { color: var(--green); } .neg { color: var(--red); } .neu { color: var(--blue); }

  /* 테이블 */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { color: var(--muted); font-weight: 500; text-align: left; padding: 6px 8px;
       border-bottom: 1px solid var(--border); }
  td { padding: 7px 8px; border-bottom: 1px solid #21262d; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }

  /* 로그 피드 */
  #log-feed { height: 340px; overflow-y: auto; font-size: 12px; font-family: monospace; }
  #log-feed .live   { color: var(--muted); }
  #log-feed .oracle { color: var(--purple); }
  #log-feed .entry  { color: var(--blue); }
  #log-feed .exit-win  { color: var(--green); }
  #log-feed .exit-loss { color: var(--red); }
  .log-line { padding: 2px 0; border-bottom: 1px solid #1a1e24; }
  .log-line:last-child { border: none; }
  .ts { color: var(--muted); margin-right: 8px; }

  /* 배지 */
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .badge-buy  { background: #1f3a2d; color: var(--green); }
  .badge-sell { background: #3d1f1f; color: var(--red); }
  .badge-hold { background: #1f2a3a; color: var(--blue); }

  .halted-banner {
    background: #3d1f1f; border: 1px solid var(--red); border-radius: 6px;
    padding: 10px 16px; margin: 0 24px 12px; color: var(--red); font-weight: 600;
    display: none;
  }
</style>
</head>
<body>

<header>
  <div id="status-dot"></div>
  <h1>DATA CAPITAL — Live Paper Trading</h1>
  <div id="last-update">업데이트 중...</div>
</header>

<div class="kpi-row" style="padding:16px 24px 0;" id="kpis">
  <div class="kpi"><div class="label">KODEX200</div><div class="val neu" id="kpi-price">—</div></div>
  <div class="kpi"><div class="label">RSI-14</div><div class="val neu" id="kpi-rsi">—</div></div>
  <div class="kpi"><div class="label">일일 PnL (포트폴리오)</div><div class="val neu" id="kpi-pnl">—</div></div>
  <div class="kpi"><div class="label">총 손익(원)</div><div class="val neu" id="kpi-won">—</div></div>
  <div class="kpi"><div class="label">승 / 패</div><div class="val neu" id="kpi-wr">—</div></div>
</div>

<div class="halted-banner" id="halt-banner">🚨 일일 MDD 한도 도달 — 시스템 거래 중단 중</div>

<div class="grid">
  <!-- 거래 내역 -->
  <div class="card">
    <h2>오늘 거래 내역</h2>
    <table>
      <thead>
        <tr><th>시각</th><th>에이전트</th><th>구분</th><th>결과</th><th>손익</th></tr>
      </thead>
      <tbody id="trades-tbody">
        <tr><td colspan="5" style="color:var(--muted);text-align:center">데이터 로딩 중...</td></tr>
      </tbody>
    </table>
  </div>

  <!-- 실시간 로그 -->
  <div class="card">
    <h2>실시간 로그 피드</h2>
    <div id="log-feed"></div>
  </div>
</div>

<script>
let prevTradeCount = 0;

async function refresh() {
  try {
    const r = await fetch('/api/data');
    const d = await r.json();

    const trades = d.trades;
    const exits  = trades.filter(t => t.type === 'EXIT');
    const wins   = exits.filter(t => t.pnl_pct && t.pnl_pct.startsWith('+'));
    const pnl    = d.summary.daily_pnl_pct;

    // KODEX200 현재가
    const px = d.latest && d.latest.price !== '—' ? d.latest.price : '—';
    const rsi = d.latest && d.latest.rsi !== '—' ? parseFloat(d.latest.rsi) : null;
    document.getElementById('kpi-price').textContent = px !== '—' ? px + '원' : '—';
    const rsiEl = document.getElementById('kpi-rsi');
    if (rsi !== null) {
      rsiEl.textContent = rsi.toFixed(1);
      rsiEl.className = 'val ' + (rsi > 70 ? 'pos' : rsi < 30 ? 'neg' : 'neu');
    }

    // 일일 PnL
    const pnlEl = document.getElementById('kpi-pnl');
    pnlEl.textContent = (pnl >= 0 ? '+' : '') + pnl.toFixed(3) + '%';
    pnlEl.className   = 'val ' + (pnl > 0 ? 'pos' : pnl < 0 ? 'neg' : 'neu');

    // 총 손익 (EXIT 기준)
    const wonRaw = exits.reduce((s, t) => {
      const n = parseFloat((t.pnl_won || '0').replace(/[^\\d.\\-]/g, '')) * (t.pnl_won && t.pnl_won.startsWith('-') ? -1 : 1);
      return s + (isNaN(n) ? 0 : n);
    }, 0);
    const wonEl = document.getElementById('kpi-won');
    wonEl.textContent = (wonRaw >= 0 ? '+' : '') + Math.round(wonRaw).toLocaleString() + '원';
    wonEl.className   = 'val ' + (wonRaw > 0 ? 'pos' : wonRaw < 0 ? 'neg' : 'neu');

    // 승/패
    document.getElementById('kpi-wr').textContent =
      exits.length ? wins.length + '승 / ' + (exits.length - wins.length) + '패' : '—';

    // 거래 테이블
    const tbody = document.getElementById('trades-tbody');
    if (trades.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="color:var(--muted);text-align:center;padding:20px">아직 거래 없음</td></tr>';
    } else {
      tbody.innerHTML = trades.slice().reverse().map(t => {
        const badge = t.type === 'BUY'
          ? '<span class="badge badge-buy">📈 매수</span>'
          : '<span class="badge badge-sell">💰 청산</span>';
        const pnlCls = t.pnl_pct === '—' ? '' : t.pnl_pct.startsWith('+') ? 'pos' : 'neg';
        return `<tr>
          <td>${t.time}</td>
          <td style="color:var(--blue)">${t.agent}</td>
          <td>${badge}</td>
          <td>${t.result}</td>
          <td class="${pnlCls}" style="font-weight:600">${t.pnl_pct !== '—' ? t.pnl_pct + '&nbsp;&nbsp;' : ''}${t.pnl_won}</td>
        </tr>`;
      }).join('');
    }

    // 로그 피드 (최신이 위)
    const feed = document.getElementById('log-feed');
    feed.innerHTML = d.live_rows.slice().reverse().map(row => {
      const clsMap = {live:'live', oracle:'oracle', entry:'entry', 'exit-win':'exit-win', 'exit-loss':'exit-loss'};
      const cls = clsMap[row.cls] || 'live';
      return `<div class="log-line ${cls}"><span class="ts">${row.time}</span>${escHtml(row.text)}</div>`;
    }).join('');

    // 헬스
    document.getElementById('halt-banner').style.display = d.summary.halted ? 'block' : 'none';
    document.getElementById('last-update').textContent = '마지막 갱신: ' + new Date().toLocaleTimeString('ko-KR');

    // 새 거래 브라우저 알림
    if (trades.length > prevTradeCount && prevTradeCount > 0) {
      const nt = trades[trades.length - 1];
      if (Notification.permission === 'granted') {
        new Notification('DATA CAPITAL', { body: `${nt.icon} [${nt.agent}] ${nt.result} ${nt.pnl_pct}` });
      }
    }
    prevTradeCount = trades.length;

  } catch(e) {
    document.getElementById('last-update').textContent = '⚠ 연결 오류: ' + e.message;
  }
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

if (Notification.permission !== 'granted') Notification.requestPermission();
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""

# ─── HTTP 핸들러 ──────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 서버 로그 조용히

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send(200, "text/html; charset=utf-8", HTML.encode())
        elif self.path == "/api/data":
            trades, live_rows, summary, latest = parse_log()
            payload = json.dumps(
                {"trades": trades, "live_rows": live_rows, "summary": summary, "latest": latest},
                ensure_ascii=False,
            ).encode()
            self._send(200, "application/json; charset=utf-8", payload)
        else:
            self._send(404, "text/plain", b"Not Found")

    def _send(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[Dashboard] http://localhost:{PORT}  (Ctrl+C 로 종료)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard] 종료")
