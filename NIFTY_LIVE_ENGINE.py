#!/usr/bin/env python3
"""
NIFTY R8 LIVE RUNNER (GitHub Actions)
=====================================

Purpose
-------
- Pull recent NIFTY spot 1-minute data from Yahoo Finance (^NSEI).
- Keep a rolling raw-minute cache in the repository.
- Replay the exact local NIFTY_RULES_ONLY_ENGINE.py using ONLY data known so far.
- Send immediate Telegram alerts for newly generated entries/exits.
- Send a 30-minute heartbeat/status during the trading session.
- 09:00 IST scheduler-alive message.
- 09:10 IST pre-session Yahoo last-price report.
- 09:15 IST market-open report.
- No NEW live entry alert after 15:15 IST.
- 15:30 IST close report uses the completed 15:29 spot minute when available.

The live runner does NOT contain historical trades, P&L ledgers, date-specific
winning trades, or forced results. The persisted JSON state is only for alert
deduplication and the static viewer; strategy decisions are regenerated from
raw OHLC by NIFTY_RULES_ONLY_ENGINE.py on each run.

GitHub Secrets expected:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

Optional environment variables:
    NIFTY_SYMBOL=^NSEI
    NIFTY_ENGINE=NIFTY_RULES_ONLY_ENGINE.py
    LIVE_CACHE=live_cache.csv
    LIVE_STATE=live_state.json
    LIVE_STATUS=live_status.json
    LIVE_TRADES=live_trades.csv
    MIN_WARMUP_SESSIONS=5

For local/offline testing:
    python NIFTY_LIVE_ENGINE.py --offline-csv path/to/nifty.csv \
        --now "2026-05-15 15:30:00+05:30" --no-telegram
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

IST = "Asia/Kolkata"
SYMBOL = os.getenv("NIFTY_SYMBOL", "^NSEI")
ENGINE_FILE = Path(os.getenv("NIFTY_ENGINE", "NIFTY_RULES_ONLY_ENGINE.py"))
CACHE_FILE = Path(os.getenv("LIVE_CACHE", "live_cache.csv"))
STATE_FILE = Path(os.getenv("LIVE_STATE", "live_state.json"))
STATUS_FILE = Path(os.getenv("LIVE_STATUS", "live_status.json"))
TRADES_FILE = Path(os.getenv("LIVE_TRADES", "live_trades.csv"))
MIN_WARMUP_SESSIONS = int(os.getenv("MIN_WARMUP_SESSIONS", "5"))
LIVE_ENTRY_CUTOFF_MIN = 15 * 60 + 15
REGULAR_END_MIN = 15 * 60 + 30
HEARTBEAT_MINUTES = 30
CHECKPOINT = {
    "name": "R8-14 locked / exact replay",
    "period": "2020-01-01 to 2026-05-15",
    "points": 42371.65,
    "pf": 3.43843858533484,
    "dd": 261.45,
    "trades": 2716,
    "green_weeks": 224,
    "red_weeks": 70,
    "flat_weeks": 39,
}


def now_ist(arg: str | None = None) -> pd.Timestamp:
    if arg:
        t = pd.Timestamp(arg)
        if t.tzinfo is None:
            return t.tz_localize(IST)
        return t.tz_convert(IST)
    return pd.Timestamp.now(tz=IST)


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    q = df.copy()
    if isinstance(q.columns, pd.MultiIndex):
        # yfinance may return (field, ticker) or (ticker, field)
        q.columns = [str(c[0]).lower() for c in q.columns]
    else:
        q.columns = [str(c).lower().strip() for c in q.columns]
    if "timestamp" not in q.columns:
        q = q.reset_index()
        q = q.rename(columns={q.columns[0]: "timestamp"})
    q["timestamp"] = pd.to_datetime(q["timestamp"])
    # Convert timezone-aware timestamps to IST, then store naive IST to match backtest engine.
    if getattr(q["timestamp"].dt, "tz", None) is not None:
        q["timestamp"] = q["timestamp"].dt.tz_convert(IST).dt.tz_localize(None)
    keep = [c for c in ["timestamp", "open", "high", "low", "close"] if c in q.columns]
    q = q[keep].copy()
    for c in ["open", "high", "low", "close"]:
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q = q.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return q.reset_index(drop=True)


def fetch_yahoo_1m() -> pd.DataFrame:
    import yfinance as yf
    q = yf.download(SYMBOL, period="7d", interval="1m", auto_adjust=False, prepost=True, progress=False, threads=False)
    return normalize_ohlc(q)


def fetch_fast_price() -> tuple[float | None, float | None]:
    try:
        import yfinance as yf
        fi = yf.Ticker(SYMBOL).fast_info
        last = float(fi["last_price"]) if fi.get("last_price") is not None else None
        prev = float(fi["previous_close"]) if fi.get("previous_close") is not None else None
        return last, prev
    except Exception:
        return None, None


def combine_cache(new: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    old = pd.DataFrame()
    if CACHE_FILE.exists():
        try:
            old = normalize_ohlc(pd.read_csv(CACHE_FILE))
        except Exception:
            old = pd.DataFrame()
    q = pd.concat([old, new], ignore_index=True) if len(old) else new.copy()
    if not len(q):
        return q
    q = q.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    # Never feed the currently forming minute into the rules engine.
    completed_before = now.tz_localize(None).floor("min")
    q = q[q.timestamp < completed_before]
    # Keep a rolling 90-day repository cache; Yahoo bootstrap may initially be shorter.
    q = q[q.timestamp >= completed_before - pd.Timedelta(days=90)]
    m = q.timestamp.dt.hour * 60 + q.timestamp.dt.minute
    q = q[(m >= 9 * 60 + 15) & (m < 15 * 60 + 30)].reset_index(drop=True)
    q.to_csv(CACHE_FILE, index=False)
    return q


def session_phase(now: pd.Timestamp) -> str:
    m = now.hour * 60 + now.minute
    if m < 9 * 60: return "pre-09:00"
    if m < 9 * 60 + 15: return "pre-session"
    if m <= LIVE_ENTRY_CUTOFF_MIN: return "live-trading"
    if m < REGULAR_END_MIN: return "reporting-only"
    return "closed"


def heartbeat_slot_key(now: pd.Timestamp) -> str | None:
    """Return the latest 30-minute status slot anchored at 09:15 IST.

    This is delay-tolerant: if GitHub starts a scheduled job a few minutes late,
    the latest unsent slot is still delivered instead of being lost forever.
    """
    m = now.hour * 60 + now.minute
    start = 9 * 60 + 15
    if m < start or m > LIVE_ENTRY_CUTOFF_MIN:
        return None
    slot_min = start + ((m - start) // HEARTBEAT_MINUTES) * HEARTBEAT_MINUTES
    hh, mm = divmod(slot_min, 60)
    return f"{now.strftime('%Y-%m-%d')}|HB{hh:02d}{mm:02d}"


def parent_30m(cache: pd.DataFrame) -> dict[str, Any]:
    if len(cache) == 0:
        return {"direction": "UNKNOWN"}
    x = cache.copy(); minute = x.timestamp.dt.hour * 60 + x.timestamp.dt.minute
    x["day"] = x.timestamp.dt.normalize(); x["bucket"] = ((minute - 555) // 30).astype(int)
    b = x.groupby(["day", "bucket"], sort=False).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), cnt=("close", "size"), end=("timestamp", "last")).reset_index(drop=True)
    b = b[b.cnt >= 30].copy()
    if len(b) < 2:
        return {"direction": "WARMUP"}
    b["ema19"] = b.close.ewm(span=19, adjust=False).mean(); b["ema29"] = b.close.ewm(span=29, adjust=False).mean()
    r = b.iloc[-1]; p = b.iloc[-2]
    direction = "BULL" if r.ema19 > r.ema29 else "BEAR"
    cross = None
    if r.ema19 > r.ema29 and p.ema19 <= p.ema29: cross = "BULL_CROSS"
    if r.ema19 < r.ema29 and p.ema19 >= p.ema29: cross = "BEAR_CROSS"
    return {"direction": direction, "ema19": float(r.ema19), "ema29": float(r.ema29), "gap": float(abs(r.ema19-r.ema29)), "last_completed_30m": str(r.end), "cross": cross}


def chronological_loss_streak(trades: pd.DataFrame) -> int:
    if trades is None or len(trades) == 0: return 0
    q = trades.sort_values(["Exit_Time", "Entry_Time"])
    st = 0
    for p in q.Points.to_numpy(float): st = st + 1 if p <= 0 else 0
    return int(st)


def run_rules_engine(cache: pd.DataFrame, runtime_dir: Path) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    if not ENGINE_FILE.exists() or len(cache) == 0:
        return None, pd.DataFrame()
    sessions = cache.timestamp.dt.normalize().nunique()
    if sessions < MIN_WARMUP_SESSIONS:
        return None, pd.DataFrame()
    start = cache.timestamp.min().normalize().strftime("%Y-%m-%d")
    end = (cache.timestamp.max() + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    raw = runtime_dir / "live_input.csv"; cache.to_csv(raw, index=False)
    cmd = [sys.executable, str(ENGINE_FILE), "--nifty", str(raw), "--start", start, "--end", end, "--out", str(runtime_dir)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=150)
    if p.returncode != 0:
        raise RuntimeError(f"R8 engine failed: {p.stderr[-2000:]}")
    summary = load_json(runtime_dir / "summary.json", None)
    trades = pd.read_csv(runtime_dir / "all_generated.csv") if (runtime_dir / "all_generated.csv").exists() else pd.DataFrame()
    if len(trades):
        trades["Entry_Time"] = pd.to_datetime(trades.Entry_Time); trades["Exit_Time"] = pd.to_datetime(trades.Exit_Time)
    return summary, trades


def telegram(text: str, disabled: bool = False) -> bool:
    """Send a formatted Telegram message. Telegram has no arbitrary text colors,
    so the runner uses HTML emphasis plus colored emoji status markers.
    """
    if disabled:
        print(text)
        return False
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        print("TELEGRAM not configured; message follows:\n" + text)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"Telegram error: {e}", file=sys.stderr)
        return False


def normalize_state_schema(state: Any) -> dict[str, Any]:
    """Migrate any older live_state.json shape without crashing the runner."""
    if not isinstance(state, dict):
        state = {}
    for key in ("seen_entries", "seen_exits", "heartbeats", "special"):
        if not isinstance(state.get(key), list):
            state[key] = []
    return state


def delivered(ok: bool, no_telegram: bool) -> bool:
    """Offline tests count as delivered; live runs count only real Telegram success."""
    return bool(ok or no_telegram)


def send_once(state: dict[str, Any], bucket: str, key: str, text: str, no_telegram: bool) -> bool:
    """Send and de-duplicate only after confirmed delivery. Failed sends are retried."""
    arr = state.setdefault(bucket, [])
    if key in arr:
        return True
    ok = telegram(text, no_telegram)
    if delivered(ok, no_telegram):
        arr.append(key)
        return True
    return False


def trade_key(row: pd.Series) -> str:
    return f"{row.get('Module','?')}|{row.get('Direction','?')}|{pd.Timestamp(row.Entry_Time).isoformat()}"


def fmt_points(x: float | None) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:+.2f}"


def pnl_icon(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "⚪"
    if x > 0:
        return "🟢"
    if x < 0:
        return "🔴"
    return "⚪"


def direction_meta(direction: str) -> tuple[str, str, str]:
    d = str(direction or "").upper()
    if d == "LONG" or d == "BULL":
        return "🟢", "LONG", "BUY ITM3 CE"
    if d == "SHORT" or d == "BEAR":
        return "🔴", "SHORT", "BUY ITM3 PE"
    return "🟡", d or "UNKNOWN", "WAIT"


def module_target_points(module: str) -> float | None:
    # Fixed-harvest R7/R8 child modules. Older parent/runner modules are
    # managed by their own trailing / state exits and intentionally show no
    # synthetic target in Telegram.
    targets = {
        "R7_3_STRONG_MICRO": 35.0,
        "R7_4_STRONG_CHILD": 40.0,
        "R7_5_INTERMEDIATE": 40.0,
        "R7_6_SECOND_CHILD": 25.0,
        "R7_7_MORNING_PROOF_LONG": 20.0,
        "R7_8_LATE_CHILD": 20.0,
        "R8_1_REARM": 20.0,
        "R8_2_15M_LONG_SHORT_CHILD": 20.0,
        "R8_SHADOW_LONG": 25.0,
        "R8_5_GAP_OR30": 20.0,
        "R8_OPENING_DRIVE_SHORT": 100.0,
        "R8_PULLBACK_RECLAIM_HQ": 40.0,
        "R8_COUNTERSWING_RECLAIM": 60.0,
        "R8_LATE_COUNTER_RECLAIM": 40.0,
        "R8_MIDDAY_LONG_TRANSITION": 40.0,
        "R8_LATE_LONG_TRANSITION": 40.0,
    }
    return targets.get(str(module))


def cache_price_at(cache: pd.DataFrame, when: pd.Timestamp, field: str = "open") -> float | None:
    if cache is None or len(cache) == 0:
        return None
    t = pd.Timestamp(when)
    q = cache.loc[cache.timestamp == t, field] if field in cache.columns else pd.Series(dtype=float)
    if len(q):
        try:
            return float(q.iloc[-1])
        except Exception:
            return None
    return None


def price_plan(entry_spot: float | None, direction: str, module: str) -> tuple[str, str]:
    if entry_spot is None or pd.isna(entry_spot):
        return "—", "module-managed"
    d = str(direction).upper()
    sign = 1.0 if d == "LONG" else -1.0
    stop = entry_spot - sign * 10.0
    target_pts = module_target_points(module)
    target = entry_spot + sign * target_pts if target_pts is not None else None
    return f"{stop:.2f}", (f"{target:.2f} (+{target_pts:.0f})" if target is not None else "module-managed / trailing")


def status_message(
    now: pd.Timestamp,
    spot: float | None,
    spot_ts: str | None,
    parent: dict[str, Any],
    day_pnl: float,
    week_pnl: float,
    streak: int,
    trades_today: int,
    active_count: int = 0,
) -> str:
    spot_text = "—" if spot is None or pd.isna(spot) else f"{spot:.2f}"
    parent_dir = str(parent.get("direction", "UNKNOWN"))
    picon, _, _ = direction_meta(parent_dir)
    parent_gap = float(parent.get("gap", 0.0) or 0.0)
    ts_text = "—"
    if spot_ts:
        try:
            ts_text = pd.Timestamp(spot_ts).strftime("%H:%M")
        except Exception:
            ts_text = str(spot_ts)
    active_line = f"\n🔔 <b>Active edge rows:</b> {active_count}" if active_count else ""
    return (
        f"🔵 <b>NIFTY R8 LIVE STATUS</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🕒 <b>{now.strftime('%d-%b-%Y %H:%M IST')}</b>\n"
        f"📍 <b>Spot:</b> <code>{spot_text}</code>  <i>(1m {html.escape(ts_text)})</i>\n"
        f"{picon} <b>30m Parent:</b> {html.escape(parent_dir)}  |  EMA gap <code>{parent_gap:.2f}</code>\n"
        f"{pnl_icon(day_pnl)} <b>Today:</b> <code>{fmt_points(day_pnl)}</code> pts\n"
        f"{pnl_icon(week_pnl)} <b>This week:</b> <code>{fmt_points(week_pnl)}</code> pts\n"
        f"🧯 <b>Current loss streak:</b> {streak}\n"
        f"🧾 <b>Generated trades today:</b> {trades_today}"
        f"{active_line}\n"
        f"⏰ <b>New-entry cutoff:</b> 15:15 IST\n"
        f"🧠 <b>Engine:</b> R8 rules-only · completed candles → next-1m fill"
    )


def entry_message(
    row: pd.Series,
    cache: pd.DataFrame,
    parent: dict[str, Any],
    day_pnl: float,
    week_pnl: float,
    streak: int,
) -> str:
    module = str(row.get("Module", "?"))
    direction = str(row.get("Direction", "?"))
    icon, dlabel, option_action = direction_meta(direction)
    et = pd.Timestamp(row.Entry_Time)
    entry_spot = cache_price_at(cache, et, "open")
    stop_text, target_text = price_plan(entry_spot, direction, module)
    entry_text = "—" if entry_spot is None else f"{entry_spot:.2f}"
    parent_dir = str(parent.get("direction", "UNKNOWN"))
    return (
        f"🚨 <b>R8 NEW TRADE / SIGNAL</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{icon} <b>{html.escape(dlabel)}</b>  →  <b>{html.escape(option_action)}</b>\n"
        f"🧩 <b>Module:</b> <code>{html.escape(module)}</code>\n"
        f"🕒 <b>Entry:</b> {et.strftime('%d-%b %H:%M IST')}\n"
        f"📍 <b>Spot entry:</b> <code>{entry_text}</code>\n"
        f"🛡️ <b>Initial spot SL:</b> <code>{stop_text}</code>  (~10 pts risk)\n"
        f"🎯 <b>Spot target/management:</b> <code>{html.escape(target_text)}</code>\n"
        f"📈 <b>30m parent:</b> {html.escape(parent_dir)}\n"
        f"{pnl_icon(day_pnl)} <b>Day:</b> {fmt_points(day_pnl)}  |  "
        f"{pnl_icon(week_pnl)} <b>Week:</b> {fmt_points(week_pnl)}\n"
        f"🧯 <b>Loss streak:</b> {streak}\n"
        f"✅ <i>Completed-candle signal; entry is next regular 1-minute open.</i>"
    )


def exit_message(row: pd.Series, cache: pd.DataFrame, day_pnl: float, week_pnl: float) -> str:
    module = str(row.get("Module", "?"))
    direction = str(row.get("Direction", "?"))
    _, dlabel, option_action = direction_meta(direction)
    points = float(row.get("Points", 0.0))
    xt = pd.Timestamp(row.Exit_Time)
    exit_spot = cache_price_at(cache, xt, "open")
    exit_text = "—" if exit_spot is None else f"{exit_spot:.2f}"
    if points > 0:
        header = "✅ <b>R8 PROFIT EXIT</b>"
        result_icon = "🟢"
    elif points <= -9.5:
        header = "🛑 <b>R8 STOP / LOSS EXIT</b>"
        result_icon = "🔴"
    else:
        header = "⚪ <b>R8 FLAT / PROTECTED EXIT</b>"
        result_icon = "⚪"
    return (
        f"{header}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🧩 <b>Module:</b> <code>{html.escape(module)}</code>\n"
        f"↔️ <b>Side:</b> {html.escape(dlabel)} / {html.escape(option_action)}\n"
        f"🕒 <b>Exit:</b> {xt.strftime('%d-%b %H:%M IST')}\n"
        f"📍 <b>Exit spot:</b> <code>{exit_text}</code>\n"
        f"{result_icon} <b>Realized spot points:</b> <code>{points:+.2f}</code>\n"
        f"{pnl_icon(day_pnl)} <b>Day:</b> {fmt_points(day_pnl)}  |  "
        f"{pnl_icon(week_pnl)} <b>Week:</b> {fmt_points(week_pnl)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-csv", default=None, help="Use local OHLC instead of Yahoo (testing)")
    ap.add_argument("--now", default=None, help="Override current IST timestamp for testing")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()
    now = now_ist(args.now); phase = session_phase(now)
    state = normalize_state_schema(load_json(STATE_FILE, {}))

    # A manual GitHub Actions run always sends one explicit Telegram test message.
    # This makes Telegram verification immediate even when run outside a heartbeat slot.
    if os.getenv("NIFTY_TRIGGER", "").strip() == "workflow_dispatch" and not args.no_telegram:
        telegram(
            f"🧪 <b>NIFTY R8 MANUAL TEST</b>\n━━━━━━━━━━━━━━━━━━\n"
            f"✅ GitHub Actions reached the live runner\n"
            f"🕒 {now.strftime('%d-%b-%Y %H:%M IST')}\n"
            f"📡 Telegram connection is working."
        )

    # Delay-tolerant scheduled alerts. GitHub cron is not guaranteed to start at
    # the exact minute, so each special report has a catch-up window.
    m = now.hour * 60 + now.minute
    date_key = now.strftime("%Y-%m-%d")

    # Scheduled 09:00 BOT ONLINE. Catch up through 09:29 if GitHub started late.
    if 9*60 <= m < 9*60+30:
        key = date_key + "|0900"
        online_text = (
            f"🟦 <b>NIFTY R8 BOT ONLINE</b>\n━━━━━━━━━━━━━━━━━━\n"
            f"🕘 <b>Scheduled:</b> 09:00 IST\n"
            f"▶️ <b>Actual wake:</b> {now.strftime('%d-%b-%Y %H:%M IST')}\n"
            f"📡 Pre-session monitoring active\n"
            f"🧠 R8 rules-only engine ready\n"
            f"⏳ Next: 09:10 pre-session · 09:15 market start"
        )
        send_once(state, "special", key, online_text, args.no_telegram)

    # Scheduled 09:10 Yahoo last/previous-close report. Catch up through 09:29.
    if 9*60+10 <= m < 9*60+30:
        key = date_key + "|0910"
        if key not in state["special"]:
            lp, pc = fetch_fast_price() if not args.offline_csv else (None, None)
            chg = (lp-pc) if lp is not None and pc is not None else None
            if lp is not None:
                pct = (chg / pc * 100.0) if chg is not None and pc not in (None, 0) else None
                gap_icon = pnl_icon(chg)
                pre_text = (
                    f"🟣 <b>NIFTY R8 PRE-SESSION</b>\n━━━━━━━━━━━━━━━━━━\n"
                    f"🕘 <b>Scheduled:</b> 09:10 IST  |  <b>Delivered:</b> {now.strftime('%H:%M IST')}\n"
                    f"📍 <b>Yahoo latest:</b> <code>{lp:.2f}</code>"
                )
                if pc is not None:
                    pre_text += f"\n📌 <b>Previous close:</b> <code>{pc:.2f}</code>"
                if chg is not None:
                    pre_text += f"\n{gap_icon} <b>Gap:</b> <code>{chg:+.2f}</code> pts"
                    if pct is not None:
                        pre_text += f"  (<code>{pct:+.2f}%</code>)"
                pre_text += "\n⏳ R8 entries begin only after completed market candles are available."
            else:
                pre_text = (
                    f"🟡 <b>NIFTY R8 PRE-SESSION</b>\n━━━━━━━━━━━━━━━━━━\n"
                    f"🕘 <b>Scheduled:</b> 09:10 IST  |  <b>Delivered:</b> {now.strftime('%H:%M IST')}\n"
                    f"⚠️ Yahoo fast price unavailable. The next 5-minute run will retry market data."
                )
            send_once(state, "special", key, pre_text, args.no_telegram)

    # Persist immediately so a later Yahoo/engine failure cannot erase successful
    # pre-session Telegram delivery state.
    atomic_json(STATE_FILE, state)

    data_error = None
    if args.offline_csv:
        new = normalize_ohlc(pd.read_csv(args.offline_csv))
    else:
        try:
            new = fetch_yahoo_1m()
        except Exception as e:
            data_error = f"Yahoo 1m fetch failed: {type(e).__name__}: {e}"
            print(data_error, file=sys.stderr)
            new = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
            err_key = date_key + "|DATA_ERROR"
            send_once(
                state, "special", err_key,
                f"⚠️ <b>NIFTY LIVE DATA WARNING</b>\n━━━━━━━━━━━━━━━━━━\n"
                f"🕒 {now.strftime('%d-%b-%Y %H:%M IST')}\n"
                f"⚠️ {html.escape(data_error[-800:])}\n"
                f"🔁 Existing cache will be used if available; next run retries Yahoo automatically.",
                args.no_telegram,
            )
            atomic_json(STATE_FILE, state)
    cache = combine_cache(new, now)
    today_naive = now.tz_localize(None).normalize()
    today_rows = cache[cache.timestamp.dt.normalize() == today_naive] if len(cache) else cache
    spot = float(today_rows.close.iloc[-1]) if len(today_rows) else (float(cache.close.iloc[-1]) if len(cache) else None)
    spot_ts = str(today_rows.timestamp.iloc[-1]) if len(today_rows) else (str(cache.timestamp.iloc[-1]) if len(cache) else None)
    parent = parent_30m(cache)

    summary = None; trades = pd.DataFrame(); runtime_error = None
    try:
        with tempfile.TemporaryDirectory(prefix="nifty_r8_live_") as td:
            summary, trades = run_rules_engine(cache, Path(td))
    except Exception as e:
        runtime_error = str(e); print(runtime_error, file=sys.stderr)
        err_key = now.strftime("%Y-%m-%dT%H") + "|ENGINE_ERROR"
        send_once(
            state, "special", err_key,
            f"🚨 <b>NIFTY R8 ENGINE WARNING</b>\n━━━━━━━━━━━━━━━━━━\n"
            f"🕒 {now.strftime('%d-%b-%Y %H:%M IST')}\n"
            f"⚠️ <code>{html.escape(runtime_error[-1200:])}</code>\n"
            f"🔁 The next scheduled run will retry automatically.",
            args.no_telegram,
        )

    day_pnl = week_pnl = 0.0; streak = 0; recent = []; active = []
    if len(trades):
        tday = pd.Timestamp(today_naive)
        week0 = tday - pd.Timedelta(days=tday.weekday())
        exd = trades.Exit_Time.dt.normalize()
        day_pnl = float(trades.loc[exd == tday, "Points"].sum())
        week_pnl = float(trades.loc[(exd >= week0) & (exd <= tday), "Points"].sum())
        streak = chronological_loss_streak(trades)
        recent = trades.sort_values(["Entry_Time", "Exit_Time"]).tail(25).copy()
        # Immediate new entry alerts for TODAY only, and never after 15:15 live cutoff.
        seen_e = set(state.get("seen_entries", [])); seen_x = set(state.get("seen_exits", [])); latest = cache.timestamp.max() if len(cache) else None
        for _, r in trades.sort_values("Entry_Time").iterrows():
            if pd.Timestamp(r.Entry_Time).normalize() != tday: continue
            k = trade_key(r); em = pd.Timestamp(r.Entry_Time).hour*60 + pd.Timestamp(r.Entry_Time).minute
            if k not in seen_e and em <= LIVE_ENTRY_CUTOFF_MIN:
                ok = telegram(entry_message(r, cache, parent, day_pnl, week_pnl, streak), args.no_telegram)
                if delivered(ok, args.no_telegram):
                    seen_e.add(k)
            xk = k + "|" + pd.Timestamp(r.Exit_Time).isoformat()
            # A row ending exactly at current data edge may be a live/open row forced by replay; don't announce it as exit yet.
            if latest is not None and pd.Timestamp(r.Exit_Time) < pd.Timestamp(latest) and xk not in seen_x:
                ok = telegram(exit_message(r, cache, day_pnl, week_pnl), args.no_telegram)
                if delivered(ok, args.no_telegram):
                    seen_x.add(xk)
        state["seen_entries"] = list(seen_e)[-1500:]; state["seen_exits"] = list(seen_x)[-1500:]
        if latest is not None:
            active = trades[(trades.Entry_Time.dt.normalize()==tday) & (trades.Exit_Time==pd.Timestamp(latest))].tail(10).to_dict("records")

    trades_today = int((trades.Entry_Time.dt.normalize()==today_naive).sum()) if len(trades) else 0

    # 09:15 market-open report. Catch up through 09:44 if GitHub starts late.
    # Mark the 09:15 heartbeat slot too, preventing a duplicate status message.
    if 9*60+15 <= m < 9*60+45:
        key = date_key + "|0915"
        if key not in state["special"]:
            live_px, prev_px = fetch_fast_price() if not args.offline_csv else (None, None)
            open_spot = live_px if live_px is not None else spot
            open_text = status_message(now, open_spot, spot_ts, parent, day_pnl, week_pnl, streak, trades_today, len(active))
            open_text = open_text.replace("🔵 <b>NIFTY R8 LIVE STATUS</b>", "🟢 <b>NIFTY R8 MARKET OPEN / LIVE START</b>", 1)
            if send_once(state, "special", key, open_text, args.no_telegram):
                hb0915 = date_key + "|HB0915"
                if hb0915 not in state["heartbeats"]:
                    state["heartbeats"].append(hb0915)

    # 30-minute heartbeat/status: 09:15, 09:45, 10:15 ... 15:15.
    hb_key = heartbeat_slot_key(now)
    if hb_key and hb_key not in state["heartbeats"]:
        ok = telegram(status_message(now, spot, spot_ts, parent, day_pnl, week_pnl, streak, trades_today, len(active)), args.no_telegram)
        if delivered(ok, args.no_telegram):
            state["heartbeats"].append(hb_key)
    # 15:30 run reports completed 15:29 spot; no new entries are permitted here.
    if 15*60+29 <= m <= 15*60+40:
        key = date_key + "|CLOSE1529"
        if key not in state["special"]:
            close1529 = today_rows.loc[(today_rows.timestamp.dt.hour==15)&(today_rows.timestamp.dt.minute==29), "close"] if len(today_rows) else pd.Series(dtype=float)
            c = float(close1529.iloc[-1]) if len(close1529) else spot
            ctext = "—" if c is None else f"{c:.2f}"
            eod_text = (
                f"🌙 <b>NIFTY R8 END-OF-DAY REPORT</b>\n━━━━━━━━━━━━━━━━━━\n"
                f"📅 <b>{now.strftime('%d-%b-%Y')}</b>\n"
                f"📍 <b>15:29 spot:</b> <code>{ctext}</code>\n"
                f"{pnl_icon(day_pnl)} <b>Day:</b> <code>{fmt_points(day_pnl)}</code> pts\n"
                f"{pnl_icon(week_pnl)} <b>Week:</b> <code>{fmt_points(week_pnl)}</code> pts\n"
                f"🧾 <b>Trades today:</b> {trades_today}\n"
                f"🧯 <b>Loss streak:</b> {streak}\n"
                f"📈 <b>30m parent:</b> {html.escape(str(parent.get('direction','UNKNOWN')))}\n"
                f"🔒 New live entries were disabled after 15:15 IST."
            )
            send_once(state, "special", key, eod_text, args.no_telegram)

    # Keep state arrays bounded.
    state["heartbeats"] = state.get("heartbeats", [])[-200:]; state["special"] = state.get("special", [])[-100:]
    atomic_json(STATE_FILE, state)

    recent_records = []
    if isinstance(recent, pd.DataFrame) and len(recent):
        for _, r in recent.iterrows():
            recent_records.append({"module":str(r.get("Module","")),"direction":str(r.get("Direction","")),"entry":str(r.Entry_Time),"exit":str(r.Exit_Time),"points":float(r.Points)})
        recent[[c for c in ["Module","Direction","Entry_Time","Exit_Time","Points"] if c in recent.columns]].to_csv(TRADES_FILE,index=False)
    status = {
        "updated_at_ist": now.isoformat(), "phase": phase, "symbol": SYMBOL,
        "source": "Yahoo Finance 1m + rolling repository cache" if not args.offline_csv else "offline CSV test",
        "spot": {"last_completed_1m": spot, "timestamp": spot_ts}, "parent_30m": parent,
        "live": {"day_points":day_pnl,"week_points":week_pnl,"loss_streak":streak,"sessions_in_cache":int(cache.timestamp.dt.normalize().nunique()) if len(cache) else 0,"warmup_ready":bool(len(cache) and cache.timestamp.dt.normalize().nunique()>=MIN_WARMUP_SESSIONS),"entry_cutoff_ist":"15:15","active_edge_rows":active},
        "engine_replay": summary.get("combined") if summary else None,
        "checkpoint": CHECKPOINT, "recent_trades": recent_records, "error": runtime_error or data_error,
    }
    atomic_json(STATUS_FILE, status)
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()
