#!/usr/bin/env python3
"""
R2 LIVE GITHUB + TELEGRAM RUNNER
================================

Purpose
-------
Live alert companion for:
    NIFTY_NEW_RULES_ONLY_ENGINE_R2.py

It does NOT contain historical trades, historical timestamps, P&L ledgers,
backtest rows, encoded trades, or target-result forcing.

The live runner:
  1. Loads the exact R2 rules-only engine from the same folder.
  2. Loads/updates raw NIFTY 1-minute OHLC.
  3. Evaluates the R2 rules causally through the latest COMPLETED minute.
  4. Sends Telegram alerts for every newly detected ENTRY and EXIT.
  5. Sends a market-status Telegram alert every 30 minutes.
  6. Persists only runtime alert/data state so alerts are not duplicated.

IMPORTANT DATA REQUIREMENT
--------------------------
For exact live parity, your 1-minute source must be continuous up to the live
session. Yahoo Finance is included as a convenient fallback, but it usually
provides only recent 1-minute history and may be delayed. If your base history
has a gap before the Yahoo window, this runner prints/sends a DATA GAP warning.

Recommended production input:
  --provider csv
  --live-csv /path/to/continuously_updated_nifty_1min.csv

Convenient monitoring/testing input:
  --provider yahoo

Environment variables
---------------------
TELEGRAM_BOT_TOKEN      Telegram bot token.
TELEGRAM_CHAT_ID        Telegram target chat/channel ID.
NIFTY_HISTORY_PATH      Base NIFTY CSV/ZIP; default data/nifty_1min_2015_2026.csv.zip
R2_ENGINE_PATH          R2 engine; default ./NIFTY_NEW_RULES_ONLY_ENGINE_R2.py
R2_STATE_PATH           Runtime state JSON; default runtime/r2_live_state.json
R2_LIVE_CACHE           Incremental live cache; default runtime/nifty_live.csv
R2_PROVIDER             yahoo | csv | none; default yahoo
R2_LIVE_CSV             Continuously updated CSV when provider=csv
R2_ALERT_LOOKBACK_MIN   Alert catch-up window; default 15
R2_STATUS_MINUTES       Status cadence; default 30
R2_PERSIST_GIT          1 to git-add/commit/push runtime state/cache after each run
R2_ALLOW_SHA_MISMATCH   1 to bypass the pinned R2 engine hash safety check

Suggested GitHub Actions cadence
--------------------------------
Run this file every 5 minutes on weekdays. The runner itself suppresses
out-of-session status alerts and sends a status only once per 30-minute slot.

Dependencies
------------
pip install pandas numpy requests yfinance

This file sends alerts only. It does not place broker orders.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ENGINE_VERSION = "R2"
EXPECTED_ENGINE_SHA256 = "256058aa41dc30f64c1e819f81dacaedab8da7ecc97e7b109cc6db62b6c2a168"

SESSION_START = dtime(9, 15)
SESSION_END = dtime(15, 29)
TZ_NAME = "Asia/Kolkata"


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ_NAME)


def normalize_ts(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(x.dt, "tz", None) is not None:
            x = x.dt.tz_convert(TZ_NAME).dt.tz_localize(None)
    except Exception:
        pass
    return x


def load_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist()
                     if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
            if len(names) != 1:
                raise ValueError(f"Expected one CSV inside {path}; found {names}")
            with z.open(names[0]) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    out = df[[cols[c] for c in required]].copy()
    out.columns = required
    out["timestamp"] = normalize_ts(out["timestamp"])
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    mm = out.timestamp.dt.hour * 60 + out.timestamp.dt.minute
    out = out[(mm >= 555) & (mm <= 929)].copy()
    return out.reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def floor_completed_minute(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop the currently forming minute; keep only completed minute bars."""
    if df.empty:
        return df
    cutoff = now.floor("min").tz_localize(None)
    return df[df.timestamp < cutoff].copy()


def merge_minutes(*frames: pd.DataFrame) -> pd.DataFrame:
    xs = [x for x in frames if x is not None and len(x)]
    if not xs:
        return pd.DataFrame(columns=["timestamp","open","high","low","close"])
    z = pd.concat(xs, ignore_index=True)
    z = z.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    mm = z.timestamp.dt.hour * 60 + z.timestamp.dt.minute
    z = z[(mm >= 555) & (mm <= 929)]
    return z.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live data providers
# ---------------------------------------------------------------------------

def fetch_yahoo_1m(symbol: str = "^NSEI") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance is required for --provider yahoo") from e

    df = yf.download(
        symbol,
        period="7d",
        interval="1m",
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no 1-minute data for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    tcol = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
    out = pd.DataFrame({
        "timestamp": normalize_ts(df[tcol]),
        "open": pd.to_numeric(df["Open"], errors="coerce"),
        "high": pd.to_numeric(df["High"], errors="coerce"),
        "low": pd.to_numeric(df["Low"], errors="coerce"),
        "close": pd.to_numeric(df["Close"], errors="coerce"),
    }).dropna()
    mm = out.timestamp.dt.hour * 60 + out.timestamp.dt.minute
    out = out[(mm >= 555) & (mm <= 929)]
    return out.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def get_live_minutes(provider: str, live_csv: Optional[Path]) -> pd.DataFrame:
    provider = provider.lower().strip()
    if provider == "none":
        return pd.DataFrame(columns=["timestamp","open","high","low","close"])
    if provider == "csv":
        if live_csv is None:
            raise ValueError("--live-csv or R2_LIVE_CSV is required for provider=csv")
        return load_any(live_csv)
    if provider == "yahoo":
        return fetch_yahoo_1m("^NSEI")
    raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def telegram_send(text: str, *, quiet: bool = False) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        print("\n[TELEGRAM NOT CONFIGURED]\n" + text + "\n")
        return False
    try:
        import requests
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat,
                "text": text,
                "disable_notification": bool(quiet),
            },
            timeout=20,
        )
        if not r.ok:
            print("Telegram error:", r.status_code, r.text, file=sys.stderr)
            return False
        return True
    except Exception as e:
        print("Telegram exception:", repr(e), file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "initialized": False,
            "seen_entries": [],
            "seen_exits": [],
            "last_status_slot": None,
            "last_data_warning": None,
        }
    try:
        x = json.loads(path.read_text())
        x.setdefault("initialized", False)
        x.setdefault("seen_entries", [])
        x.setdefault("seen_exits", [])
        x.setdefault("last_status_slot", None)
        x.setdefault("last_data_warning", None)
        return x
    except Exception:
        return {
            "initialized": False,
            "seen_entries": [],
            "seen_exits": [],
            "last_status_slot": None,
            "last_data_warning": None,
        }


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(path)


def bounded_add(items: List[str], key: str, max_items: int = 5000) -> None:
    if key not in items:
        items.append(key)
    if len(items) > max_items:
        del items[:-max_items]


# ---------------------------------------------------------------------------
# Exact R2 engine loading + live-final-period adapters
# ---------------------------------------------------------------------------

def load_r2_engine(engine_path: Path):
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)

    sha = hashlib.sha256(engine_path.read_bytes()).hexdigest()
    if sha != EXPECTED_ENGINE_SHA256 and not env_bool("R2_ALLOW_SHA_MISMATCH", False):
        raise RuntimeError(
            "R2 engine SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_ENGINE_SHA256}\n"
            f"Actual  : {sha}\n"
            "Use the exact delivered R2 engine or explicitly set "
            "R2_ALLOW_SHA_MISMATCH=1 after reviewing the change."
        )

    spec = importlib.util.spec_from_file_location("r2_exact_live", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {engine_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha


def build_live_core_sim(r2):
    """
    Reuse the exact R2 sim() source and extend ONLY its final parent regime to
    the latest completed minute. A non-stopped final 'handoff' at the last
    minute is interpreted by this runner as an OPEN position, not a real exit.
    """
    src = inspect.getsource(r2.sim)
    src = src.replace("def sim(", "def _sim_live_core(", 1)

    old_loop = "for j in range(len(sig)-1):"
    new_loop = "for j in range(len(sig)):"
    if old_loop not in src:
        raise RuntimeError("R2 sim() structure changed: parent loop not found")
    src = src.replace(old_loop, new_loop, 1)

    old_line = "bi=int(sig[j]);bn=int(sig[j+1]);base=int(last[bi]+1);xi=int(last[bn]+1)"
    new_line = (
        "bi=int(sig[j]);"
        "bn=(int(sig[j+1]) if j+1<len(sig) else None);"
        "base=int(last[bi]+1);"
        "xi=(int(last[bn]+1) if bn is not None else len(df)-1)"
    )
    if old_line not in src:
        raise RuntimeError("R2 sim() structure changed: regime boundary line not found")
    src = src.replace(old_line, new_line, 1)

    ns = dict(r2.__dict__)
    exec(src, ns)
    return ns["_sim_live_core"]


def build_live_swing_collector(r2):
    """
    Reuse the exact R2 swing collector source and extend ONLY the current final
    swing to the latest completed minute. Extra final-swing non-stop rows are
    interpreted as OPEN.
    """
    src = inspect.getsource(r2.run_swing_collector)
    src = src.replace("def run_swing_collector(", "def _run_swing_collector_live(", 1)

    old_loop = "for j,e in enumerate(events[:-1]):"
    new_loop = "for j,e in enumerate(events):"
    if old_loop not in src:
        raise RuntimeError("R2 swing collector structure changed: loop not found")
    src = src.replace(old_loop, new_loop, 1)

    old_line = "sidx,d,signal_time,completed,atr=e;end=events[j+1][0];ratio=completed/atr if atr else 0.;er=qualities[j]"
    new_line = (
        "sidx,d,signal_time,completed,atr=e;"
        "end=(events[j+1][0] if j+1<len(events) else N-1);"
        "ratio=completed/atr if atr else 0.;er=qualities[j]"
    )
    if old_line not in src:
        raise RuntimeError("R2 swing collector structure changed: boundary line not found")
    src = src.replace(old_line, new_line, 1)

    ns = dict(r2.__dict__)
    exec(src, ns)
    return ns["_run_swing_collector_live"]


def core_key_from_row(r) -> Tuple:
    return (
        pd.Timestamp(r.Entry_Time),
        str(getattr(r, "type", "")),
        int(getattr(r, "master_id", -1)),
        int(getattr(r, "re", -1)),
    )


def swing_key_from_row(r) -> Tuple:
    return (
        pd.Timestamp(r.Entry_Time),
        str(getattr(r, "Route", "")),
        int(getattr(r, "d", 0)),
        round(float(getattr(r, "Entry", np.nan)), 4),
    )


def future_exit_time(last_ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(last_ts) + pd.Timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Snapshot generation
# ---------------------------------------------------------------------------

@dataclass
class LiveTrade:
    module: str
    rule: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    status: str                    # OPEN | CLOSED
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    points: Optional[float] = None
    exit_reason: Optional[str] = None

    def entry_key(self) -> str:
        return "|".join([
            ENGINE_VERSION, self.module, self.rule, self.direction,
            self.entry_time.isoformat(), f"{self.entry_price:.4f}"
        ])

    def exit_key(self) -> str:
        return self.entry_key() + "|" + (
            self.exit_time.isoformat() if self.exit_time is not None else ""
        ) + "|" + str(self.exit_reason or "")


def build_live_snapshot(r2, data: pd.DataFrame) -> Tuple[List[LiveTrade], Dict[str, Any]]:
    if len(data) < 500:
        raise RuntimeError("Not enough 1-minute data to initialize R2 live rules")

    last_ts = pd.Timestamp(data.timestamp.iloc[-1])
    last_close = float(data.close.iloc[-1])

    # Initialize exact R2 engine on the current continuous raw-minute file.
    # Use a temporary CSV because R2.init() accepts CSV/ZIP paths.
    runtime_dir = Path(os.getenv("R2_RUNTIME_DIR", "runtime"))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    replay_csv = runtime_dir / "_r2_live_replay.csv"
    data.to_csv(replay_csv, index=False)
    r2.init(replay_csv)

    start = str(pd.Timestamp(data.timestamp.iloc[0]).date())
    end = str((last_ts.normalize() + pd.Timedelta(days=1)).date())
    p = r2.r2_core_params(start, end)

    # Closed/core reference using the exact unmodified engine.
    _, closed_legs, closed_masters = r2.sim(p, frames=True)
    closed_core = r2._core_frame(closed_legs, closed_masters)
    closed_core_keys = {
        (pd.Timestamp(x.entry_time), str(x.type), int(x.master_id), int(x.re))
        for x in closed_legs.itertuples(index=False)
    }

    # Current final parent regime using exact sim source + current-end boundary.
    sim_live = build_live_core_sim(r2)
    _, live_legs, live_masters = sim_live(p, frames=True)
    live_core = r2._core_frame(live_legs, live_masters)

    trades: List[LiveTrade] = []
    core_for_state = []

    for row in live_legs.itertuples(index=False):
        md = live_masters[live_masters.master_id == row.master_id]
        d = int(md.iloc[0].d) if len(md) else 1
        key = (pd.Timestamp(row.entry_time), str(row.type), int(row.master_id), int(row.re))
        is_extra = key not in closed_core_keys
        pseudo_open = (
            is_extra
            and str(row.exit_reason).lower() == "natural"
            and pd.Timestamp(row.exit_time) >= last_ts
        )

        if pseudo_open:
            trades.append(LiveTrade(
                module="CORE",
                rule=str(row.type),
                direction="LONG" if d == 1 else "SHORT",
                entry_time=pd.Timestamp(row.entry_time),
                entry_price=float(row.entry_price),
                status="OPEN",
            ))
            core_for_state.append({
                "Entry_Time": pd.Timestamp(row.entry_time),
                "Exit_Time": future_exit_time(last_ts),
                "Points": 0.0,
                "Direction": "LONG" if d == 1 else "SHORT",
            })
        else:
            trades.append(LiveTrade(
                module="CORE",
                rule=str(row.type),
                direction="LONG" if d == 1 else "SHORT",
                entry_time=pd.Timestamp(row.entry_time),
                entry_price=float(row.entry_price),
                status="CLOSED",
                exit_time=pd.Timestamp(row.exit_time),
                exit_price=float(row.exit_price),
                points=float(row.points),
                exit_reason=str(row.exit_reason).upper(),
            ))
            core_for_state.append({
                "Entry_Time": pd.Timestamp(row.entry_time),
                "Exit_Time": pd.Timestamp(row.exit_time),
                "Points": float(row.points),
                "Direction": "LONG" if d == 1 else "SHORT",
            })

    core_state_df = pd.DataFrame(core_for_state)
    if core_state_df.empty:
        core_state_df = pd.DataFrame(columns=["Entry_Time","Exit_Time","Points","Direction"])

    # Swing: exact closed reference + current final swing adapter.
    sdf = r2.DF.copy().reset_index(drop=True)
    closed_swing = r2.run_r2_swing(sdf, closed_core)
    closed_swing_keys = {
        (pd.Timestamp(x.Entry_Time), str(x.Route), int(x.d), round(float(x.Entry),4))
        for x in closed_swing.itertuples(index=False)
    }

    swing_live_fn = build_live_swing_collector(r2)
    swing_live = swing_live_fn(
        sdf, core_state_df,
        swing_k=.75, breakout_bars=4, stop_points=10.0,
        prior_abs_min=50.0, prior_atr_min=1.5, prior_er_max=.75,
        prior_proof=20.0, prior_pull=25.0, prior_reclaim=5.0,
        max_paid_per_swing=3, max_prior_loss_streak=5,
        prior_loss_trigger=3, prior_cooldown_swings=1,
        micro_abs_max=15.0, micro_atr_max=.30, micro_anchor_atr=3.0,
        micro_proof=20.0, micro_pull=20.0, micro_reclaim=15.0,
    )

    swing_state_rows = []
    for row in swing_live.itertuples(index=False):
        key = swing_key_from_row(row)
        is_extra = key not in closed_swing_keys
        stopped = float(row.Points) <= -9.999
        pseudo_open = is_extra and not stopped and pd.Timestamp(row.Exit_Time) >= last_ts

        if pseudo_open:
            trades.append(LiveTrade(
                module="SWING",
                rule=str(row.Route),
                direction="LONG" if int(row.d) == 1 else "SHORT",
                entry_time=pd.Timestamp(row.Entry_Time),
                entry_price=float(row.Entry),
                status="OPEN",
            ))
            swing_state_rows.append({
                "Entry_Time": pd.Timestamp(row.Entry_Time),
                "Exit_Time": future_exit_time(last_ts),
                "Points": 0.0,
                "Module": "SWING",
            })
        else:
            reason = "STOP" if stopped else "SWING_REVERSAL"
            trades.append(LiveTrade(
                module="SWING",
                rule=str(row.Route),
                direction="LONG" if int(row.d) == 1 else "SHORT",
                entry_time=pd.Timestamp(row.Entry_Time),
                entry_price=float(row.Entry),
                status="CLOSED",
                exit_time=pd.Timestamp(row.Exit_Time),
                exit_price=float(row.Exit),
                points=float(row.Points),
                exit_reason=reason,
            ))
            swing_state_rows.append({
                "Entry_Time": pd.Timestamp(row.Entry_Time),
                "Exit_Time": pd.Timestamp(row.Exit_Time),
                "Points": float(row.Points),
                "Module": "SWING",
            })

    # Existing state for Repair A: core + swing, including current open occupancy.
    base_state = core_state_df[["Entry_Time","Exit_Time","Points"]].copy()
    base_state["Module"] = "CORE"
    if swing_state_rows:
        base_state = pd.concat([base_state, pd.DataFrame(swing_state_rows)], ignore_index=True)

    feat = r2._build_repair_features(sdf)
    repair_a = r2._run_week_repair(sdf, base_state, feat, r2._r2_repair_a_config(), "REPAIR_A")

    repair_a_state = []
    for row in repair_a.itertuples(index=False):
        pseudo_open = (
            str(row.Exit_Reason).upper() == "EOD"
            and pd.Timestamp(row.Exit_Time) >= last_ts
            and last_ts.time() < SESSION_END
        )
        if pseudo_open:
            trades.append(LiveTrade(
                module="REPAIR_A",
                rule=f"{row.Parent_30m}/{row.Child_15m}/{row.Trigger_5m}",
                direction=str(row.Direction),
                entry_time=pd.Timestamp(row.Entry_Time),
                entry_price=float(row.Entry_Price),
                status="OPEN",
            ))
            repair_a_state.append({
                "Entry_Time": pd.Timestamp(row.Entry_Time),
                "Exit_Time": future_exit_time(last_ts),
                "Points": 0.0,
                "Module": "REPAIR_A",
            })
        else:
            trades.append(LiveTrade(
                module="REPAIR_A",
                rule=f"{row.Parent_30m}/{row.Child_15m}/{row.Trigger_5m}",
                direction=str(row.Direction),
                entry_time=pd.Timestamp(row.Entry_Time),
                entry_price=float(row.Entry_Price),
                status="CLOSED",
                exit_time=pd.Timestamp(row.Exit_Time),
                exit_price=float(row.Exit_Price),
                points=float(row.Points),
                exit_reason=str(row.Exit_Reason),
            ))
            repair_a_state.append({
                "Entry_Time": pd.Timestamp(row.Entry_Time),
                "Exit_Time": pd.Timestamp(row.Exit_Time),
                "Points": float(row.Points),
                "Module": "REPAIR_A",
            })

    with_a = base_state.copy()
    if repair_a_state:
        with_a = pd.concat([with_a, pd.DataFrame(repair_a_state)], ignore_index=True)

    repair_b = r2._run_week_repair(sdf, with_a, feat, r2._r2_repair_b_config(), "REPAIR_B")

    for row in repair_b.itertuples(index=False):
        pseudo_open = (
            str(row.Exit_Reason).upper() == "EOD"
            and pd.Timestamp(row.Exit_Time) >= last_ts
            and last_ts.time() < SESSION_END
        )
        if pseudo_open:
            trades.append(LiveTrade(
                module="REPAIR_B",
                rule=f"{row.Parent_30m}/{row.Child_15m}/{row.Trigger_5m}",
                direction=str(row.Direction),
                entry_time=pd.Timestamp(row.Entry_Time),
                entry_price=float(row.Entry_Price),
                status="OPEN",
            ))
        else:
            trades.append(LiveTrade(
                module="REPAIR_B",
                rule=f"{row.Parent_30m}/{row.Child_15m}/{row.Trigger_5m}",
                direction=str(row.Direction),
                entry_time=pd.Timestamp(row.Entry_Time),
                entry_price=float(row.Entry_Price),
                status="CLOSED",
                exit_time=pd.Timestamp(row.Exit_Time),
                exit_price=float(row.Exit_Price),
                points=float(row.Points),
                exit_reason=str(row.Exit_Reason),
            ))

    trades.sort(key=lambda t: (t.entry_time, t.module, t.rule))

    # Runtime realized P&L from all CLOSED snapshot rows.
    closed = [t for t in trades if t.status == "CLOSED" and t.exit_time is not None]
    today = last_ts.date()
    monday = today - timedelta(days=today.weekday())
    day_pts = sum(float(t.points or 0.0) for t in closed if t.exit_time.date() == today)
    week_pts = sum(float(t.points or 0.0) for t in closed
                   if monday <= t.exit_time.date() <= today)

    open_trades = [t for t in trades if t.status == "OPEN"]
    for t in open_trades:
        d = 1 if t.direction == "LONG" else -1
        t.points = d * (last_close - t.entry_price)

    meta = {
        "last_timestamp": last_ts,
        "last_close": last_close,
        "day_realized": float(day_pts),
        "week_realized": float(week_pts),
        "open_count": len(open_trades),
    }
    return trades, meta


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def option_hint(direction: str) -> str:
    return "BUY ITM3 CE" if direction == "LONG" else "BUY ITM3 PE"


def entry_message(t: LiveTrade) -> str:
    return (
        f"🟢 NIFTY {ENGINE_VERSION} ENTRY\n"
        f"Module: {t.module}\n"
        f"Rule: {t.rule}\n"
        f"Direction: {t.direction}\n"
        f"Option hint: {option_hint(t.direction)}\n"
        f"Entry: {t.entry_price:.2f}\n"
        f"Initial spot SL: 10.00 points\n"
        f"Time: {t.entry_time:%Y-%m-%d %H:%M}"
    )


def exit_message(t: LiveTrade) -> str:
    pts = float(t.points or 0.0)
    icon = "✅" if pts > 0 else ("🔴" if pts < 0 else "⚪")
    return (
        f"{icon} NIFTY {ENGINE_VERSION} EXIT\n"
        f"Module: {t.module}\n"
        f"Rule: {t.rule}\n"
        f"Direction: {t.direction}\n"
        f"Entry: {t.entry_price:.2f}\n"
        f"Exit: {float(t.exit_price):.2f}\n"
        f"Spot points: {pts:+.2f}\n"
        f"Reason: {t.exit_reason}\n"
        f"Time: {t.exit_time:%Y-%m-%d %H:%M}"
    )


def status_message(meta: Dict[str, Any], trades: List[LiveTrade], warnings: List[str]) -> str:
    opens = [t for t in trades if t.status == "OPEN"]
    if opens:
        pos = "\n".join(
            f"• {t.module} {t.direction} @ {t.entry_price:.2f} | "
            f"{float(t.points or 0):+.2f} pts | {t.rule}"
            for t in opens
        )
    else:
        pos = "• No open R2 position"

    warn_text = ""
    if warnings:
        warn_text = "\n⚠️ " + "\n⚠️ ".join(warnings)

    return (
        f"📊 NIFTY {ENGINE_VERSION} 30-MIN STATUS\n"
        f"Last bar: {meta['last_timestamp']:%Y-%m-%d %H:%M}\n"
        f"NIFTY: {meta['last_close']:.2f}\n"
        f"Day realized: {meta['day_realized']:+.2f} pts\n"
        f"Week realized: {meta['week_realized']:+.2f} pts\n"
        f"Open positions: {meta['open_count']}\n"
        f"{pos}"
        f"{warn_text}"
    )


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

def data_warnings(base: pd.DataFrame, live: pd.DataFrame, combined: pd.DataFrame) -> List[str]:
    w: List[str] = []
    if combined.empty:
        return ["No NIFTY minute data available"]

    if len(base) and len(live):
        bmax = pd.Timestamp(base.timestamp.max())
        lmin = pd.Timestamp(live.timestamp.min())
        if lmin - bmax > pd.Timedelta(days=10):
            w.append(
                f"1m history gap: base ends {bmax:%Y-%m-%d}, "
                f"live source begins {lmin:%Y-%m-%d}. "
                "Backfill the missing 1m period for exact live parity."
            )

    # Detect missing bars inside the latest session.
    latest_date = pd.Timestamp(combined.timestamp.max()).date()
    z = combined[combined.timestamp.dt.date == latest_date]
    if len(z) >= 2:
        diffs = z.timestamp.diff().dropna().dt.total_seconds() / 60.0
        misses = int((diffs > 1.5).sum())
        if misses:
            w.append(f"Latest session contains {misses} intraday minute gap(s).")
    return w


# ---------------------------------------------------------------------------
# 30-minute slot logic
# ---------------------------------------------------------------------------

def market_open_now(now: pd.Timestamp) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.timetz().replace(tzinfo=None)
    return SESSION_START <= t <= dtime(15, 40)


def status_slot(now: pd.Timestamp, interval_min: int) -> Optional[str]:
    if not market_open_now(now):
        return None
    naive = now.tz_localize(None)
    anchor = naive.normalize() + pd.Timedelta(hours=9, minutes=15)
    if naive < anchor:
        return None
    elapsed = int((naive - anchor).total_seconds() // 60)
    slot = elapsed // interval_min
    slot_time = anchor + pd.Timedelta(minutes=slot * interval_min)
    if slot_time.time() > dtime(15, 15):
        slot_time = naive.normalize() + pd.Timedelta(hours=15, minutes=15)
    return slot_time.strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Git persistence (optional)
# ---------------------------------------------------------------------------

def git_persist(paths: Iterable[Path]) -> None:
    if not env_bool("R2_PERSIST_GIT", False):
        return
    try:
        subprocess.run(["git","config","user.name","r2-live-bot"], check=False)
        subprocess.run(["git","config","user.email","r2-live-bot@users.noreply.github.com"], check=False)
        existing = [str(p) for p in paths if p.exists()]
        if not existing:
            return
        subprocess.run(["git","add",*existing], check=False)
        diff = subprocess.run(["git","diff","--cached","--quiet"])
        if diff.returncode == 0:
            return
        subprocess.run(["git","commit","-m","R2 live state [skip ci]"], check=True)
        subprocess.run(["git","push"], check=True)
    except Exception as e:
        print("Git persistence warning:", repr(e), file=sys.stderr)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def run_once(args) -> int:
    now = now_ist()

    engine_path = Path(args.engine)
    base_path = Path(args.history)
    state_path = Path(args.state)
    cache_path = Path(args.cache)
    live_csv = Path(args.live_csv) if args.live_csv else None

    r2, engine_sha = load_r2_engine(engine_path)

    base = load_any(base_path)
    cache = load_any(cache_path) if cache_path.exists() else pd.DataFrame(
        columns=["timestamp","open","high","low","close"]
    )
    live = get_live_minutes(args.provider, live_csv)
    live = floor_completed_minute(live, now)

    combined = merge_minutes(base, cache, live)
    combined = floor_completed_minute(combined, now)
    if combined.empty:
        raise RuntimeError("No completed NIFTY 1-minute bars available")

    # Persist only bars after the fixed base history.
    if len(base):
        base_max = pd.Timestamp(base.timestamp.max())
        new_cache = combined[combined.timestamp > base_max].copy()
        if len(new_cache):
            save_csv(new_cache, cache_path)

    warnings = data_warnings(base, merge_minutes(cache, live), combined)

    trades, meta = build_live_snapshot(r2, combined)
    state = load_state(state_path)
    seen_entries: List[str] = list(state.get("seen_entries", []))
    seen_exits: List[str] = list(state.get("seen_exits", []))

    lookback = pd.Timedelta(minutes=int(args.alert_lookback_min))
    last_ts = pd.Timestamp(meta["last_timestamp"])
    cutoff = last_ts - lookback

    # On first ever run, suppress historical spam. Current/recent open entries
    # are still announced so the operator sees current live state.
    first_run = not bool(state.get("initialized"))

    new_entries = []
    new_exits = []
    for t in trades:
        ek = t.entry_key()
        xk = t.exit_key() if t.status == "CLOSED" and t.exit_time is not None else None

        if ek not in seen_entries:
            if (not first_run) or t.entry_time >= cutoff or t.status == "OPEN":
                new_entries.append(t)
            bounded_add(seen_entries, ek)

        if xk and xk not in seen_exits:
            if (not first_run) or (t.exit_time is not None and t.exit_time >= cutoff):
                new_exits.append(t)
            bounded_add(seen_exits, xk)

    # Send in chronological order. If a trade both entered and exited between
    # polls, the entry alert is sent before its exit alert.
    events = []
    for t in new_entries:
        events.append((t.entry_time, 0, "entry", t))
    for t in new_exits:
        events.append((t.exit_time, 1, "exit", t))
    events.sort(key=lambda x: (x[0], x[1]))

    for _, _, typ, t in events:
        telegram_send(entry_message(t) if typ == "entry" else exit_message(t))

    # 30-minute status alert.
    slot = status_slot(now, int(args.status_minutes))
    if slot and slot != state.get("last_status_slot"):
        telegram_send(status_message(meta, trades, warnings), quiet=False)
        state["last_status_slot"] = slot

    # Send a data-gap warning once per unique warning signature.
    warn_sig = " | ".join(warnings)
    if warnings and warn_sig != state.get("last_data_warning"):
        telegram_send(
            f"⚠️ NIFTY {ENGINE_VERSION} DATA WARNING\n" +
            "\n".join(f"• {x}" for x in warnings)
        )
        state["last_data_warning"] = warn_sig

    if first_run:
        telegram_send(
            f"🤖 NIFTY {ENGINE_VERSION} LIVE RUNNER INITIALIZED\n"
            f"Engine SHA: {engine_sha[:16]}…\n"
            f"Last completed bar: {meta['last_timestamp']:%Y-%m-%d %H:%M}\n"
            f"Open positions detected: {meta['open_count']}\n"
            f"Provider: {args.provider}"
        )
        state["initialized"] = True

    state["seen_entries"] = seen_entries
    state["seen_exits"] = seen_exits
    state["last_bar"] = str(meta["last_timestamp"])
    state["engine_sha256"] = engine_sha
    state["updated_at_ist"] = str(now)
    save_state(state_path, state)

    git_persist([state_path, cache_path])

    print(json.dumps({
        "engine": ENGINE_VERSION,
        "engine_sha256": engine_sha,
        "provider": args.provider,
        "last_bar": str(meta["last_timestamp"]),
        "last_close": meta["last_close"],
        "open_positions": meta["open_count"],
        "day_realized": meta["day_realized"],
        "week_realized": meta["week_realized"],
        "new_entry_alerts": len(new_entries),
        "new_exit_alerts": len(new_exits),
        "warnings": warnings,
    }, indent=2, default=str))
    return 0


def build_args():
    ap = argparse.ArgumentParser(
        description="R2 GitHub live-market + Telegram alert runner"
    )
    ap.add_argument(
        "--engine",
        default=os.getenv("R2_ENGINE_PATH", "NIFTY_NEW_RULES_ONLY_ENGINE_R2.py"),
    )
    ap.add_argument(
        "--history",
        default=os.getenv("NIFTY_HISTORY_PATH", "data/nifty_1min_2015_2026.csv.zip"),
    )
    ap.add_argument(
        "--provider",
        choices=["yahoo","csv","none"],
        default=os.getenv("R2_PROVIDER", "yahoo"),
    )
    ap.add_argument(
        "--live-csv",
        default=os.getenv("R2_LIVE_CSV"),
        help="Continuously updated raw 1m CSV for provider=csv",
    )
    ap.add_argument(
        "--state",
        default=os.getenv("R2_STATE_PATH", "runtime/r2_live_state.json"),
    )
    ap.add_argument(
        "--cache",
        default=os.getenv("R2_LIVE_CACHE", "runtime/nifty_live.csv"),
    )
    ap.add_argument(
        "--alert-lookback-min",
        type=int,
        default=int(os.getenv("R2_ALERT_LOOKBACK_MIN", "15")),
    )
    ap.add_argument(
        "--status-minutes",
        type=int,
        default=int(os.getenv("R2_STATUS_MINUTES", "30")),
    )
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_once(build_args()))
