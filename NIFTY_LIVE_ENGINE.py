#!/usr/bin/env python3
"""
NIFTY LIVE GITHUB + TELEGRAM ENGINE — CAS AWARE
=================================================

Companion for:
    NIFTY_RULES_ONLY_ENGINE.py

This live runner intentionally does NOT require the historical NIFTY backtest
CSV/ZIP. It consumes only live-provider minute data plus the rolling live cache
that this runner itself builds in the GitHub repository.

Why a live cache still exists
-----------------------------
strategy uses 5m/15m/30m EMA/state rules. Those indicators cannot be calculated from
only the current minute. On the first run Yahoo supplies its recent live-source
1-minute window as warm-up. From then onward the runner persists its own live
minutes in runtime/nifty_live_minutes.csv and continuously extends that cache.
No static 2020-2026 backtest file is required by this GitHub runner.

Live phases (IST)
-----------------
09:00-09:14  : boot / provider warm-up / no new strategy entry.
09:15-15:15  : strategy spot-signal engine active. Completed-candle logic only.
15:15 onward : freeze new strategy spot entries/signals for the day.
15:15-15:29  : CAS observation phase; keep option market management available.
15:29-15:35  : capture/update provisional closing NIFTY reference.
15:35        : lock latest available CAS/final NIFTY reference.
15:35-15:40  : option-management window only; no new strategy spot entries.
15:40        : F&O management window ends.

The dedicated NSE CAS page currently states that CAS runs 15:15-15:35 and
equity derivatives trade until 15:40. This runner therefore treats any
15:29/15:30 NIFTY value as provisional and keeps updating it until 15:35.

Telegram
--------
- 09:00 live-day boot notice.
- 09:10 pre-session NIFTY spot report.
- 09:15 strategy-session start notice.
- Every newly detected accepted strategy SIGNAL.
- Every newly detected strategy ENTRY.
- Every newly detected strategy EXIT / stop.
- Every 30-minute status during the live session.
- CAS provisional/final price notices.
- CAS option-management notices for open strategy positions.
- 15:40 final F&O-management notice.

Repository logs
---------------
logs/nifty_trade_events.csv
logs/nifty_status_30m.csv
logs/nifty_cas_management.csv
logs/nifty_errors.csv
runtime/nifty_live_state.json
runtime/nifty_live_minutes.csv

The workflow should git-add/commit/push runtime/ and logs/ after every run.

CAS option-management defaults
------------------------------
If an open strategy child reaches +70 spot points, roll it into a fresh ITM3.
Also roll a child at the two-session age boundary while the parent remains active.
These are live-execution management defaults only; they do NOT alter the
backtest strategy research engine.

Environment variables
---------------------
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
NIFTY_ENGINE_PATH              default NIFTY_RULES_ONLY_ENGINE.py
NIFTY_PROVIDER                 yahoo | csv | none; default yahoo
NIFTY_LIVE_CSV                 optional continuously updated live CSV
NIFTY_STATE_PATH               default runtime/nifty_live_state.json
NIFTY_LIVE_CACHE               default runtime/nifty_live_minutes.csv
NIFTY_CACHE_DAYS               default 90
NIFTY_ALERT_LOOKBACK_MIN       default 15
NIFTY_STATUS_MINUTES           default 30
NIFTY_CAS_HARVEST_TRIGGER      default 70
NIFTY_EVENT_LOG                default logs/nifty_trade_events.csv
NIFTY_STATUS_LOG               default logs/nifty_status_30m.csv
NIFTY_CAS_LOG                  default logs/nifty_cas_management.csv
NIFTY_ERROR_LOG                default logs/nifty_errors.csv
NIFTY_EXCEL_REPORT             default reports/NIFTY_LIVE_Trade_Report.xlsx
NIFTY_ALLOW_SHA_MISMATCH       default 0

Dependencies
------------
pip install pandas numpy requests yfinance

This file sends alerts only. It does not place broker orders.
"""

from __future__ import annotations

import argparse
import calendar
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

ENGINE_VERSION = "R4"
EXPECTED_ENGINE_SHA256 = "2b73c7b13b0ab025b16f0c368814f4f40a844b71d133af143fe47bffeb3b2c24"

def _env_clock(name: str, default: str) -> dtime:
    raw = os.getenv(name, default).strip()
    try:
        hh, mm = raw.split(":", 1)
        return dtime(int(hh), int(mm))
    except Exception as exc:
        raise ValueError(f"{name} must be HH:MM, got {raw!r}") from exc

TZ_NAME = "Asia/Kolkata"
LIVE_WINDOW_START = _env_clock("NIFTY_BOOT_TIME", "09:00")
PRESESSION_REPORT_TIME = _env_clock("NIFTY_PRESESSION_TIME", "09:10")
STRATEGY_START = _env_clock("NIFTY_STRATEGY_START_TIME", "09:15")
STRATEGY_LAST = _env_clock("NIFTY_STRATEGY_LAST_TIME", "15:15")
CAS_PROVISIONAL_FROM = _env_clock("NIFTY_CAS_PROVISIONAL_TIME", "15:29")
CAS_FINAL_LOCK = _env_clock("NIFTY_CAS_FINAL_TIME", "15:35")
FNO_CLOSE = _env_clock("NIFTY_FNO_CLOSE_TIME", "15:40")


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
    out = out[(mm >= 540) & (mm <= 940)].copy()
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
    z = z[(mm >= 540) & (mm <= 940)]
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
    out = out[(mm >= 540) & (mm <= 940)]
    return out.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def get_live_minutes(provider: str, live_csv: Optional[Path]) -> pd.DataFrame:
    provider = provider.lower().strip()
    if provider == "none":
        return pd.DataFrame(columns=["timestamp","open","high","low","close"])
    if provider == "csv":
        if live_csv is None:
            raise ValueError("--live-csv or NIFTY_LIVE_CSV is required for provider=csv")
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
            "seen_signals": [],
            "seen_entries": [],
            "seen_exits": [],
            "last_status_slot": None,
            "last_data_warning": None,
            "boot_notice_date": None,
            "presession_notice_date": None,
            "session_start_notice_date": None,
        }
    try:
        x = json.loads(path.read_text())
        x.setdefault("initialized", False)
        x.setdefault("seen_signals", [])
        x.setdefault("seen_entries", [])
        x.setdefault("seen_exits", [])
        x.setdefault("last_status_slot", None)
        x.setdefault("last_data_warning", None)
        x.setdefault("boot_notice_date", None)
        x.setdefault("presession_notice_date", None)
        x.setdefault("session_start_notice_date", None)
        return x
    except Exception:
        return {
            "initialized": False,
            "seen_signals": [],
            "seen_entries": [],
            "seen_exits": [],
            "last_status_slot": None,
            "last_data_warning": None,
            "boot_notice_date": None,
            "presession_notice_date": None,
            "session_start_notice_date": None,
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
# Exact strategy engine loading + live-final-period adapters
# ---------------------------------------------------------------------------

def load_strategy_engine(engine_path: Path):
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)

    sha = hashlib.sha256(engine_path.read_bytes()).hexdigest()
    if sha != EXPECTED_ENGINE_SHA256 and not env_bool("NIFTY_ALLOW_SHA_MISMATCH", False):
        raise RuntimeError(
            "strategy engine SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_ENGINE_SHA256}\n"
            f"Actual  : {sha}\n"
            "Use the exact delivered strategy engine or explicitly set "
            "NIFTY_ALLOW_SHA_MISMATCH=1 after reviewing the change."
        )

    spec = importlib.util.spec_from_file_location("nifty_strategy_live", engine_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {engine_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha


def build_live_core_sim(strategy):
    """
    Reuse the exact strategy sim() source and extend ONLY its final parent regime to
    the latest completed minute. A non-stopped final 'handoff' at the last
    minute is interpreted by this runner as an OPEN position, not a real exit.
    """
    src = inspect.getsource(strategy.sim)
    src = src.replace("def sim(", "def _sim_live_core(", 1)

    old_loop = "for j in range(len(sig)-1):"
    new_loop = "for j in range(len(sig)):"
    if old_loop not in src:
        raise RuntimeError("strategy sim() structure changed: parent loop not found")
    src = src.replace(old_loop, new_loop, 1)

    old_line = "bi=int(sig[j]);bn=int(sig[j+1]);base=int(last[bi]+1);xi=int(last[bn]+1)"
    new_line = (
        "bi=int(sig[j]);"
        "bn=(int(sig[j+1]) if j+1<len(sig) else None);"
        "base=int(last[bi]+1);"
        "xi=(int(last[bn]+1) if bn is not None else len(df)-1)"
    )
    if old_line not in src:
        raise RuntimeError("strategy sim() structure changed: regime boundary line not found")
    src = src.replace(old_line, new_line, 1)

    ns = dict(strategy.__dict__)
    exec(src, ns)
    return ns["_sim_live_core"]


def build_live_swing_collector(strategy):
    """
    Reuse the exact strategy swing collector source and extend ONLY the current final
    swing to the latest completed minute. Extra final-swing non-stop rows are
    interpreted as OPEN.
    """
    src = inspect.getsource(strategy.run_swing_collector)
    src = src.replace("def run_swing_collector(", "def _run_swing_collector_live(", 1)

    old_loop = "for j,e in enumerate(events[:-1]):"
    new_loop = "for j,e in enumerate(events):"
    if old_loop not in src:
        raise RuntimeError("strategy swing collector structure changed: loop not found")
    src = src.replace(old_loop, new_loop, 1)

    old_line = "sidx,d,signal_time,completed,atr=e;end=events[j+1][0];ratio=completed/atr if atr else 0.;er=qualities[j]"
    new_line = (
        "sidx,d,signal_time,completed,atr=e;"
        "end=(events[j+1][0] if j+1<len(events) else N-1);"
        "ratio=completed/atr if atr else 0.;er=qualities[j]"
    )
    if old_line not in src:
        raise RuntimeError("strategy swing collector structure changed: boundary line not found")
    src = src.replace(old_line, new_line, 1)

    ns = dict(strategy.__dict__)
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


def build_live_snapshot(strategy, data: pd.DataFrame) -> Tuple[List[LiveTrade], Dict[str, Any]]:
    if len(data) < 500:
        raise RuntimeError("Not enough 1-minute data to initialize NIFTY live rules")

    last_ts = pd.Timestamp(data.timestamp.iloc[-1])
    last_close = float(data.close.iloc[-1])

    # Initialize exact strategy engine on the current continuous raw-minute file.
    # Use a temporary CSV because strategy.init() accepts CSV/ZIP paths.
    runtime_dir = Path(os.getenv("NIFTY_RUNTIME_DIR", "runtime"))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    replay_csv = runtime_dir / "_nifty_live_replay.csv"
    data.to_csv(replay_csv, index=False)
    strategy.init(replay_csv)

    start = str(pd.Timestamp(data.timestamp.iloc[0]).date())
    end = str((last_ts.normalize() + pd.Timedelta(days=1)).date())
    p = strategy.strategy_core_params(start, end)

    # Closed/core reference using the exact unmodified engine.
    _, closed_legs, closed_masters = strategy.sim(p, frames=True)
    closed_core = strategy._core_frame(closed_legs, closed_masters)
    closed_core_keys = {
        (pd.Timestamp(x.entry_time), str(x.type), int(x.master_id), int(x.re))
        for x in closed_legs.itertuples(index=False)
    }

    # Current final parent regime using exact sim source + current-end boundary.
    sim_live = build_live_core_sim(strategy)
    _, live_legs, live_masters = sim_live(p, frames=True)
    live_core = strategy._core_frame(live_legs, live_masters)

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
    sdf = strategy.DF.copy().reset_index(drop=True)
    closed_swing = strategy.run_strategy_swing(sdf, closed_core)
    closed_swing_keys = {
        (pd.Timestamp(x.Entry_Time), str(x.Route), int(x.d), round(float(x.Entry),4))
        for x in closed_swing.itertuples(index=False)
    }

    swing_live_fn = build_live_swing_collector(strategy)
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

    feat = strategy._build_repair_features(sdf)
    repair_a = strategy._run_week_repair(sdf, base_state, feat, strategy._strategy_repair_a_config(), "REPAIR_A")

    repair_a_state = []
    for row in repair_a.itertuples(index=False):
        pseudo_open = (
            str(row.Exit_Reason).upper() == "EOD"
            and pd.Timestamp(row.Exit_Time) >= last_ts
            and last_ts.time() < STRATEGY_LAST
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

    repair_b = strategy._run_week_repair(sdf, with_a, feat, strategy._strategy_repair_b_config(), "REPAIR_B")

    for row in repair_b.itertuples(index=False):
        pseudo_open = (
            str(row.Exit_Reason).upper() == "EOD"
            and pd.Timestamp(row.Exit_Time) >= last_ts
            and last_ts.time() < STRATEGY_LAST
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
        pos = "• No open strategy position"

    warn_text = ""
    if warnings:
        warn_text = "\n⚠️ " + "\n⚠️ ".join(warnings)

    quote_line = ""
    if meta.get("live_quote") is not None:
        quote_line = (
            f"Live NIFTY: {float(meta['live_quote']):.2f} "
            f"({meta.get('live_quote_time'):%H:%M})\n"
        )

    return (
        f"📊 NIFTY {ENGINE_VERSION} 30-MIN STATUS\n"
        f"Phase: {meta.get('phase','')}\n"
        f"Last strategy bar: {meta['last_timestamp']:%Y-%m-%d %H:%M}\n"
        f"{quote_line}"
        f"Day realized: {meta['day_realized']:+.2f} pts\n"
        f"Week realized: {meta['week_realized']:+.2f} pts\n"
        f"Open positions: {meta['open_count']}\n"
        f"{pos}"
        f"{warn_text}"
    )


def signal_message(t: LiveTrade) -> str:
    return (
        f"🔔 NIFTY {ENGINE_VERSION} STRATEGY SIGNAL\n"
        f"Module: {t.module}\n"
        f"Rule: {t.rule}\n"
        f"Direction: {t.direction}\n"
        f"Action: {option_hint(t.direction)}\n"
        f"Engine next-fill time: {t.entry_time:%Y-%m-%d %H:%M}\n"
        f"Initial spot SL: 10.00 points"
    )


def _quote_at_or_before(df: pd.DataFrame, now: pd.Timestamp, cutoff: dtime
                        ) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    if df.empty:
        return None, None
    today = now.tz_localize(None).date()
    z = df[(df.timestamp.dt.date == today) & (df.timestamp.dt.time <= cutoff)]
    if z.empty:
        # If the live provider has no pre-open print, use the latest completed
        # quote available and label it clearly rather than fabricating 09:10 data.
        z = df[df.timestamp < now.tz_localize(None)]
    if z.empty:
        return None, None
    r = z.iloc[-1]
    return pd.Timestamp(r.timestamp), float(r.close)


def handle_session_notices(now: pd.Timestamp, live_all: pd.DataFrame,
                           state: Dict[str, Any]) -> None:
    """Stateful 09:00 boot, 09:10 pre-session quote, and 09:15 start alerts."""
    if now.weekday() >= 5:
        return
    t = now.timetz().replace(tzinfo=None)
    today_key = str(now.date())

    if t >= LIVE_WINDOW_START and state.get("boot_notice_date") != today_key:
        telegram_send(
            f"🟦 NIFTY {ENGINE_VERSION} LIVE DAY START — {LIVE_WINDOW_START:%H:%M} IST\n"
            f"Provider/cache warm-up started.\n"
            f"Strategy entries remain disabled until {STRATEGY_START:%H:%M} IST."
        )
        state["boot_notice_date"] = today_key

    if t >= PRESESSION_REPORT_TIME and state.get("presession_notice_date") != today_key:
        qt, qp = _quote_at_or_before(live_all, now, PRESESSION_REPORT_TIME)
        if qp is None:
            qline = "NIFTY spot: unavailable from live provider"
        else:
            qline = f"NIFTY spot: {qp:.2f} | latest completed quote {qt:%H:%M}"
        telegram_send(
            f"🌅 NIFTY {ENGINE_VERSION} PRE-SESSION REPORT — {PRESESSION_REPORT_TIME:%H:%M} IST\n"
            f"{qline}\n"
            f"Strategy starts at {STRATEGY_START:%H:%M} IST."
        )
        state["presession_notice_date"] = today_key

    if t >= STRATEGY_START and state.get("session_start_notice_date") != today_key:
        qt, qp = _quote_at_or_before(live_all, now, STRATEGY_START)
        qline = "NIFTY spot: unavailable" if qp is None else f"NIFTY spot: {qp:.2f} | quote {qt:%H:%M}"
        telegram_send(
            f"🟢 NIFTY {ENGINE_VERSION} STRATEGY SESSION START — {STRATEGY_START:%H:%M} IST\n"
            f"{qline}\n"
            f"Completed-candle rules active; new accepted signals use the engine's next-1m fill."
        )
        state["session_start_notice_date"] = today_key


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

def data_warnings(combined: pd.DataFrame, now: pd.Timestamp) -> List[str]:
    w: List[str] = []
    if combined.empty:
        return ["No live NIFTY minute data available"]

    latest = pd.Timestamp(combined.timestamp.max())
    today = now.tz_localize(None).date()

    if latest.date() != today and now.timetz().replace(tzinfo=None) >= STRATEGY_START:
        w.append(
            f"No current-day NIFTY minute received yet. Latest provider bar is {latest:%Y-%m-%d %H:%M}."
        )

    # Latest-session intraday minute gaps. CAS/transition naturally may contain
    # sparse prints, so only enforce this check through the 15:15 strategy cut.
    z = combined[
        (combined.timestamp.dt.date == latest.date()) &
        (combined.timestamp.dt.time >= STRATEGY_START) &
        (combined.timestamp.dt.time <= STRATEGY_LAST)
    ]
    if len(z) >= 2:
        diffs = z.timestamp.diff().dropna().dt.total_seconds() / 60.0
        misses = int((diffs > 1.5).sum())
        if misses:
            w.append(f"Strategy session contains {misses} intraday minute gap(s).")

    # Warm-up warning on a new repository/live cache.
    dates = sorted(set(combined.timestamp.dt.date))
    if len(dates) < 3:
        w.append(
            "Live cache has fewer than 3 trading days of warm-up. "
            "strategy EMA/state values are still initializing."
        )
    return w


# ---------------------------------------------------------------------------
# 30-minute slot logic
# ---------------------------------------------------------------------------

def market_open_now(now: pd.Timestamp) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.timetz().replace(tzinfo=None)
    return LIVE_WINDOW_START <= t <= FNO_CLOSE


def status_slot(now: pd.Timestamp, interval_min: int) -> Optional[str]:
    """30-minute status slots anchored to 09:15: 09:15, 09:45, ... 15:15."""
    if not market_open_now(now):
        return None
    naive = now.tz_localize(None)
    anchor = naive.normalize() + pd.Timedelta(hours=9, minutes=15)
    if naive < anchor:
        return None
    elapsed = int((naive - anchor).total_seconds() // 60)
    slot = elapsed // interval_min
    slot_time = anchor + pd.Timedelta(minutes=slot * interval_min)
    if slot_time.time() > STRATEGY_LAST:
        slot_time = naive.normalize() + pd.Timedelta(hours=15, minutes=15)
    return slot_time.strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Repository CSV logs
# ---------------------------------------------------------------------------

def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def log_trade_event(path: Path, event: str, t: LiveTrade, now: pd.Timestamp) -> None:
    append_csv_row(path, {
        "logged_at_ist": str(now),
        "event": event,
        "module": t.module,
        "rule": t.rule,
        "direction": t.direction,
        "entry_time": str(t.entry_time),
        "entry_price": t.entry_price,
        "exit_time": str(t.exit_time) if t.exit_time is not None else "",
        "exit_price": t.exit_price if t.exit_price is not None else "",
        "points": t.points if t.points is not None else "",
        "exit_reason": t.exit_reason or "",
    })


def log_status(path: Path, meta: Dict[str, Any], now: pd.Timestamp, slot: str) -> None:
    append_csv_row(path, {
        "logged_at_ist": str(now),
        "slot": slot,
        "last_strategy_bar": str(meta.get("last_timestamp", "")),
        "strategy_close": meta.get("last_close", ""),
        "live_quote_time": str(meta.get("live_quote_time", "")),
        "live_quote": meta.get("live_quote", ""),
        "day_realized": meta.get("day_realized", 0.0),
        "week_realized": meta.get("week_realized", 0.0),
        "open_positions": meta.get("open_count", 0),
        "phase": meta.get("phase", ""),
    })


def log_cas(path: Path, now: pd.Timestamp, event: str, spot: Optional[float],
            quote_time: Optional[pd.Timestamp], detail: str) -> None:
    append_csv_row(path, {
        "logged_at_ist": str(now),
        "event": event,
        "spot": "" if spot is None else spot,
        "quote_time": "" if quote_time is None else str(quote_time),
        "detail": detail,
    })


def live_phase(now: pd.Timestamp) -> str:
    t = now.timetz().replace(tzinfo=None)
    if t < STRATEGY_START:
        return "PREOPEN_WARMUP"
    if t <= STRATEGY_LAST:
        return "SPOT_STRATEGY_ACTIVE"
    if t < CAS_PROVISIONAL_FROM:
        return "CAS_WAIT"
    if t < CAS_FINAL_LOCK:
        return "CAS_PROVISIONAL"
    if t < FNO_CLOSE:
        return "OPTION_MANAGEMENT"
    return "FNO_CLOSE"


def filter_strategy_minutes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tm = df.timestamp.dt.time
    return df[(tm >= STRATEGY_START) & (tm <= STRATEGY_LAST)].copy().reset_index(drop=True)


def today_latest_quote(df: pd.DataFrame, now: pd.Timestamp,
                       from_time: dtime = CAS_PROVISIONAL_FROM
                       ) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    if df.empty:
        return None, None
    today = now.tz_localize(None).date()
    z = df[(df.timestamp.dt.date == today) &
           (df.timestamp.dt.time >= from_time)]
    if z.empty:
        return None, None
    r = z.iloc[-1]
    return pd.Timestamp(r.timestamp), float(r.close)


def _child_session_age(live_all: pd.DataFrame, child_date: str, today_key: str) -> int:
    try:
        a=pd.Timestamp(child_date).date(); b=pd.Timestamp(today_key).date()
    except Exception:
        return 1
    days=sorted(set(live_all.timestamp.dt.date))
    return max(1, sum(1 for d in days if a <= d <= b))


def cas_management_message(t: LiveTrade, spot: float, pnl: float,
                           provisional: bool, action: str, session_age: int,
                           child_entry: float) -> str:
    pxlabel = "PROVISIONAL CAS" if provisional else "FINAL CAS"
    if action == "ROLL_PROFIT":
        advice = (
            "Action: +70 child threshold reached. Close the current ITM3 child "
            "and buy a fresh ITM3 in the SAME direction; parent trend remains active."
        )
    elif action == "ROLL_AGE":
        advice = (
            "Action: two-session age boundary reached. Roll the full current ITM3 "
            "into a fresh ITM3 in the SAME direction; parent trend remains active."
        )
    else:
        advice = (
            "Action: HOLD current ITM3 child. No +70 profit roll and age boundary "
            "has not been reached."
        )
    return (
        f"🟣 NIFTY LIVE CAS OPTION MANAGEMENT\n"
        f"{pxlabel} NIFTY: {spot:.2f}\n"
        f"Module: {t.module}\n"
        f"Direction: {t.direction}\n"
        f"Current child spot reference: {child_entry:.2f}\n"
        f"Child session age: {session_age}\n"
        f"CAS-referenced child P&L: {pnl:+.2f}\n"
        f"{advice}"
    )


def handle_cas_management(
    now: pd.Timestamp,
    live_all: pd.DataFrame,
    trades: List[LiveTrade],
    state: Dict[str, Any],
    cas_log: Path,
) -> None:
    tnow = now.timetz().replace(tzinfo=None)
    if tnow < CAS_PROVISIONAL_FROM:
        return

    qtime, qspot = today_latest_quote(live_all, now, CAS_PROVISIONAL_FROM)
    if qspot is None:
        return

    today_key = str(now.date())
    state["cas_latest_spot"] = float(qspot)
    state["cas_latest_quote_time"] = str(qtime)

    provisional = tnow < CAS_FINAL_LOCK
    phase_key = "provisional" if provisional else "final"

    sent_key = f"{today_key}:{phase_key}:{qtime}"
    sent_prices = state.setdefault("cas_price_notices", [])
    if sent_key not in sent_prices:
        telegram_send(
            ("🟪 NIFTY LIVE CAS PRICE\n"
             f"{'Provisional' if provisional else 'Final/locked'} NIFTY: {qspot:.2f}\n"
             f"Quote time: {qtime:%Y-%m-%d %H:%M}\n"
             f"New strategy spot entries are frozen after 15:15.\n"
             f"F&O remains in management mode until 15:40.")
        )
        log_cas(
            cas_log, now,
            "CAS_PROVISIONAL_PRICE" if provisional else "CAS_FINAL_PRICE",
            qspot, qtime,
            "strategy spot entries frozen; option management only."
        )
        bounded_add(sent_prices, sent_key, 200)

    if not provisional:
        state["cas_final_spot"] = float(qspot)
        state["cas_final_quote_time"] = str(qtime)
        state["cas_final_date"] = today_key

    harvest_trigger = float(os.getenv("NIFTY_CAS_HARVEST_TRIGGER", "70"))
    mgmt_seen = state.setdefault("cas_management_notices", [])
    children = state.setdefault("itm3_children", {})

    for tr in [x for x in trades if x.status == "OPEN"]:
        ek=tr.entry_key()
        child=children.get(ek)
        if child is None:
            child={"entry_price":float(tr.entry_price),"entry_date":str(tr.entry_time.date())}
            children[ek]=child
        child_entry=float(child.get("entry_price",tr.entry_price))
        child_date=str(child.get("entry_date",tr.entry_time.date()))
        age=_child_session_age(live_all,child_date,today_key)
        d = 1 if tr.direction == "LONG" else -1
        pnl = d * (float(qspot) - child_entry)
        action = "ROLL_PROFIT" if pnl >= harvest_trigger else ("ROLL_AGE" if age >= 2 else "HOLD")
        key = f"{today_key}:{phase_key}:{ek}:{action}:{child_date}"
        if key in mgmt_seen:
            continue
        telegram_send(cas_management_message(tr,qspot,pnl,provisional,action,age,child_entry))
        log_cas(
            cas_log, now, f"OPTION_{action}", qspot, qtime,
            f"{tr.module} {tr.direction} child_entry={child_entry:.2f} "
            f"child_age={age} child_pnl={pnl:+.2f}"
        )
        bounded_add(mgmt_seen,key,3000)
        # Only the final/locked CAS reference changes the live child state.
        if (not provisional) and action in ("ROLL_PROFIT","ROLL_AGE"):
            children[ek]={"entry_price":float(qspot),"entry_date":today_key,
                          "rolled_at":str(qtime),"roll_reason":action}

    if tnow >= FNO_CLOSE and state.get("fno_close_notice_date") != today_key:
        opens = [x for x in trades if x.status == "OPEN"]
        telegram_send(
            f"⏰ NIFTY LIVE F&O WINDOW COMPLETE — 15:40 IST\n"
            f"CAS/final NIFTY reference: {qspot:.2f}\n"
            f"Open strategy positions: {len(opens)}\n"
            f"No new strategy spot signal will be created after 15:15."
        )
        log_cas(cas_log,now,"FNO_CLOSE_1540",qspot,qtime,
                f"Open strategy positions={len(opens)}")
        state["fno_close_notice_date"] = today_key


# ---------------------------------------------------------------------------
# Git persistence (optional)
# ---------------------------------------------------------------------------

def git_persist(paths: Iterable[Path]) -> None:
    if not env_bool("NIFTY_PERSIST_GIT", False):
        return
    try:
        subprocess.run(["git","config","user.name","strategy-live-bot"], check=False)
        subprocess.run(["git","config","user.email","strategy-live-bot@users.noreply.github.com"], check=False)
        existing = [str(p) for p in paths if p.exists()]
        if not existing:
            return
        subprocess.run(["git","add",*existing], check=False)
        diff = subprocess.run(["git","diff","--cached","--quiet"])
        if diff.returncode == 0:
            return
        subprocess.run(["git","commit","-m","NIFTY live state [skip ci]"], check=True)
        subprocess.run(["git","push"], check=True)
    except Exception as e:
        print("Git persistence warning:", repr(e), file=sys.stderr)



# ---------------------------------------------------------------------------
# Colorful Excel live report
# ---------------------------------------------------------------------------

def _status_name(points: float) -> str:
    return "POSITIVE" if points > 0 else ("NEGATIVE" if points < 0 else "FLAT")


def write_live_excel_report(
    report_path: Path,
    trades: List[LiveTrade],
    market_minutes: pd.DataFrame,
    meta: Dict[str, Any],
    engine_sha: str,
) -> None:
    """
    Create/update a colorful live workbook on every run.

    Sheets:
      Dashboard
      Trades
      Daily
      Weekly
      Monthly
      Yearly
      Monthly Heatmap
      Rule Summary
    """
    try:
        import xlsxwriter
    except ImportError as e:
        raise RuntimeError(
            "xlsxwriter is required for Excel output. "
            "Install with: pip install xlsxwriter"
        ) from e

    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert current strategy snapshot to one closed/open trade table.
    rows = []
    for i, t in enumerate(sorted(trades, key=lambda x: (x.entry_time, x.module, x.rule)), 1):
        pts = float(t.points or 0.0)
        rows.append({
            "Trade_ID": i,
            "Module": t.module,
            "Rule_Type": t.rule,
            "Direction": t.direction,
            "Entry_Time": pd.Timestamp(t.entry_time),
            "Exit_Time": pd.Timestamp(t.exit_time) if t.exit_time is not None else pd.NaT,
            "Entry_Price": float(t.entry_price),
            "Exit_Price": float(t.exit_price) if t.exit_price is not None else np.nan,
            "Points": pts,
            "Status": t.status,
            "Result": _status_name(pts) if t.status == "CLOSED" else "OPEN",
            "Exit_Reason": t.exit_reason or "",
        })
    tdf = pd.DataFrame(rows)

    # Period ledgers are realized P&L only; open P&L is shown on Dashboard/Trades.
    closed = tdf[tdf["Status"] == "CLOSED"].copy() if len(tdf) else pd.DataFrame()
    if len(closed):
        closed["Exit_Date"] = pd.to_datetime(closed["Exit_Time"]).dt.date
    market_dates = sorted(set(market_minutes.timestamp.dt.date)) if len(market_minutes) else []

    daily_rows = []
    for d in market_dates:
        z = closed[closed["Exit_Date"] == d] if len(closed) else closed
        pts = float(z["Points"].sum()) if len(z) else 0.0
        daily_rows.append({
            "Date": d,
            "Day": pd.Timestamp(d).day_name(),
            "Trades": int(len(z)) if len(z) else 0,
            "Points": pts,
            "Status": _status_name(pts),
        })
    ddf = pd.DataFrame(daily_rows)

    def period_frame(freq: str) -> pd.DataFrame:
        if ddf.empty:
            return pd.DataFrame(columns=["Period","Market_Days","Trades","Points","Status"])
        x = ddf.copy()
        x["Date"] = pd.to_datetime(x["Date"])
        if freq == "W":
            x["Period"] = x["Date"] - pd.to_timedelta(x["Date"].dt.weekday, unit="D")
        elif freq == "M":
            x["Period"] = x["Date"].dt.to_period("M").astype(str)
        else:
            x["Period"] = x["Date"].dt.year
        y = x.groupby("Period", as_index=False).agg(
            Market_Days=("Date","count"),
            Trades=("Trades","sum"),
            Points=("Points","sum"),
        )
        y["Status"] = y["Points"].map(_status_name)
        return y

    wdf = period_frame("W")
    mdf = period_frame("M")
    ydf = period_frame("Y")

    wb = xlsxwriter.Workbook(str(report_path))
    wb.set_properties({
        "title": "NIFTY Live Trade Report",
        "subject": "Rules-only live strategy report",
        "author": "NIFTY live engine",
        "comments": f"Strategy engine SHA-256: {engine_sha}",
    })

    navy = "#102A43"; white = "#FFFFFF"; green = "#DCFCE7"; green_d = "#166534"
    red = "#FEE2E2"; red_d = "#991B1B"; gray = "#E5E7EB"; gray_d = "#4B5563"
    pale = "#F8FAFC"; orange = "#FFF7ED"; orange_d = "#9A3412"

    fmt_title = wb.add_format({"bold":True,"font_color":white,"bg_color":navy,
                               "font_size":18,"align":"center","valign":"vcenter"})
    fmt_header = wb.add_format({"bold":True,"font_color":white,"bg_color":navy,
                                "align":"center","valign":"vcenter","border":1})
    fmt_kpi = wb.add_format({"bold":True,"font_color":navy,"bg_color":pale,
                             "font_size":14,"align":"center","border":1})
    fmt_pos = wb.add_format({"bg_color":green,"font_color":green_d,"bold":True})
    fmt_neg = wb.add_format({"bg_color":red,"font_color":red_d,"bold":True})
    fmt_flat = wb.add_format({"bg_color":gray,"font_color":gray_d})
    fmt_open = wb.add_format({"bg_color":"#DBEAFE","font_color":"#1E40AF","bold":True})
    fmt_num = wb.add_format({"num_format":"0.00"})
    fmt_dt = wb.add_format({"num_format":"yyyy-mm-dd hh:mm"})
    fmt_date = wb.add_format({"num_format":"yyyy-mm-dd"})
    fmt_note = wb.add_format({"bg_color":orange,"font_color":orange_d,"italic":True,
                              "text_wrap":True})

    dash = wb.add_worksheet("Dashboard")
    dash.merge_range("A1:H2", "NIFTY LIVE — Trade & Calendar Report", fmt_title)
    dash.merge_range("A3:H3",
                     f"Live-only rules engine | Strategy SHA-256 {engine_sha[:16]}… | "
                     f"Phase: {meta.get('phase','')}",
                     wb.add_format({"italic":True,"font_color":gray_d,"align":"center"}))

    closed_pts = closed["Points"].to_numpy(float) if len(closed) else np.array([], dtype=float)
    if len(closed_pts):
        gains = float(closed_pts[closed_pts > 0].sum())
        losses = float(-closed_pts[closed_pts < 0].sum())
        pf = gains / losses if losses > 0 else float("inf")
        eq = np.cumsum(closed_pts)
        peaks = np.maximum.accumulate(np.r_[0.0, eq])[:-1]
        dd = float(np.max(peaks - eq)) if len(eq) else 0.0
        net = float(closed_pts.sum())
        win = float((closed_pts > 0).mean() * 100)
    else:
        pf = dd = net = win = 0.0

    open_pnl = float(sum(float(t.points or 0.0) for t in trades if t.status == "OPEN"))
    kpis = [
        ("Realized Points", net),
        ("Open P&L", open_pnl),
        ("Profit Factor", pf),
        ("Realized DD", dd),
        ("Closed Trades", int(len(closed))),
        ("Open Trades", int(sum(t.status=="OPEN" for t in trades))),
    ]
    for c,(lab,val) in enumerate(kpis):
        dash.write(4,c,lab,fmt_header)
        dash.write(5,c,val,fmt_kpi)

    def cnt(df):
        if df is None or df.empty:
            return (0,0,0)
        s = df["Status"].value_counts()
        return int(s.get("POSITIVE",0)), int(s.get("NEGATIVE",0)), int(s.get("FLAT",0))

    dash.write_row("A8", ["Period","Positive","Negative","Flat"], fmt_header)
    for r,(name,df) in enumerate([("Daily",ddf),("Weekly",wdf),("Monthly",mdf),("Yearly",ydf)], start=8):
        p,n,f = cnt(df)
        dash.write(r,0,name)
        dash.write(r,1,p,fmt_pos); dash.write(r,2,n,fmt_neg); dash.write(r,3,f,fmt_flat)

    dash.merge_range("A14:H14",
        "Period status uses realized closed-trade P&L. Open-position P&L is shown separately.",
        fmt_note)
    dash.set_column("A:H", 17)

    # Trades
    sh = wb.add_worksheet("Trades")
    trade_cols = ["Trade_ID","Module","Rule_Type","Direction","Entry_Time","Exit_Time",
                  "Entry_Price","Exit_Price","Points","Status","Result","Exit_Reason"]
    sh.write_row(0,0,trade_cols,fmt_header)
    for r,row in enumerate(rows, start=1):
        for c,k in enumerate(trade_cols):
            v = row[k]
            if k in ("Entry_Time","Exit_Time") and pd.notna(v):
                sh.write_datetime(r,c,pd.Timestamp(v).to_pydatetime(),fmt_dt)
            elif k in ("Entry_Price","Exit_Price","Points") and pd.notna(v):
                sh.write_number(r,c,float(v),fmt_num)
            else:
                sh.write(r,c,"" if pd.isna(v) else v)
        if row["Status"] == "OPEN":
            sh.set_row(r, None, fmt_open)
    if rows:
        sh.conditional_format(1,8,len(rows),8,{"type":"cell","criteria":">","value":0,"format":fmt_pos})
        sh.conditional_format(1,8,len(rows),8,{"type":"cell","criteria":"<","value":0,"format":fmt_neg})
    sh.freeze_panes(1,0)
    sh.autofilter(0,0,max(1,len(rows)),len(trade_cols)-1)
    sh.set_column("A:B",12); sh.set_column("C:C",32); sh.set_column("D:D",11)
    sh.set_column("E:F",19); sh.set_column("G:I",13); sh.set_column("J:L",16)

    def write_period_sheet(name: str, df: pd.DataFrame):
        ws = wb.add_worksheet(name)
        if df is None or df.empty:
            ws.write(0,0,"No data yet",fmt_note)
            return
        cols = list(df.columns)
        ws.write_row(0,0,cols,fmt_header)
        for r,rec in enumerate(df.to_dict("records"), start=1):
            for c,k in enumerate(cols):
                v = rec[k]
                if isinstance(v, (pd.Timestamp, datetime)):
                    ws.write_datetime(r,c,pd.Timestamp(v).to_pydatetime(),fmt_date)
                elif hasattr(v, "year") and hasattr(v, "month") and hasattr(v, "day") and not isinstance(v,str):
                    ws.write_datetime(r,c,pd.Timestamp(v).to_pydatetime(),fmt_date)
                elif isinstance(v,(int,np.integer)):
                    ws.write_number(r,c,int(v))
                elif isinstance(v,(float,np.floating)):
                    ws.write_number(r,c,float(v),fmt_num)
                else:
                    ws.write(r,c,str(v))
        if "Points" in cols:
            pc=cols.index("Points")
            ws.conditional_format(1,pc,len(df),pc,{"type":"cell","criteria":">","value":0,"format":fmt_pos})
            ws.conditional_format(1,pc,len(df),pc,{"type":"cell","criteria":"<","value":0,"format":fmt_neg})
            ws.conditional_format(1,pc,len(df),pc,{"type":"cell","criteria":"==","value":0,"format":fmt_flat})
        ws.freeze_panes(1,0); ws.autofilter(0,0,len(df),len(cols)-1); ws.set_column(0,len(cols)-1,15)

    write_period_sheet("Daily", ddf)
    write_period_sheet("Weekly", wdf)
    write_period_sheet("Monthly", mdf)
    write_period_sheet("Yearly", ydf)

    # Monthly heatmap
    hm = wb.add_worksheet("Monthly Heatmap")
    hm.write(0,0,"Year",fmt_header)
    for mo in range(1,13):
        hm.write(0,mo,calendar.month_abbr[mo],fmt_header)
    years = sorted({pd.Timestamp(d).year for d in market_dates})
    month_points = {}
    if not mdf.empty:
        for rec in mdf.to_dict("records"):
            p = str(rec["Period"])
            yy,mm = p.split("-")
            month_points[(int(yy),int(mm))] = float(rec["Points"])
    for r,yy in enumerate(years,start=1):
        hm.write(r,0,yy,fmt_header)
        for mo in range(1,13):
            if (yy,mo) in month_points:
                val=month_points[(yy,mo)]
                hm.write_number(r,mo,val,fmt_pos if val>0 else (fmt_neg if val<0 else fmt_flat))
            else:
                hm.write(r,mo,"")
    hm.set_column("A:M",12)

    rules = wb.add_worksheet("Rule Summary")
    rules.write_row(0,0,["Live Phase","Rule"],fmt_header)
    rule_rows = [
        ("09:00-09:14","Provider/cache warm-up; no new strategy entry."),
        ("09:15-15:15","Rules-only strategy active; completed-candle logic; next-minute fills."),
        ("After 15:15","Freeze new spot-strategy entries."),
        ("15:15-15:29","CAS observation / option-management preparation."),
        ("15:29-15:35","CAS price treated as provisional and updated when provider changes."),
        ("15:35-15:40","Final CAS reference locked; +70/age-based ITM3 rolling only."),
        ("Risk","Initial paid stop is 10 NIFTY spot points."),
    ]
    for r,row in enumerate(rule_rows,start=1):
        rules.write_row(r,0,row)
    rules.set_column("A:A",18); rules.set_column("B:B",90)

    wb.close()

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def run_once(args) -> int:
    now = now_ist()

    engine_path = Path(args.engine)
    state_path = Path(args.state)
    cache_path = Path(args.cache)
    live_csv = Path(args.live_csv) if args.live_csv else None
    event_log = Path(args.event_log)
    status_log = Path(args.status_log)
    cas_log = Path(args.cas_log)
    report_path = Path(args.excel_report)

    strategy, engine_sha = load_strategy_engine(engine_path)

    # LIVE ONLY: no static historical NIFTY file is loaded here.
    cache = load_any(cache_path) if cache_path.exists() else pd.DataFrame(
        columns=["timestamp","open","high","low","close"]
    )
    live = get_live_minutes(args.provider, live_csv)
    live = floor_completed_minute(live, now)

    combined = merge_minutes(cache, live)
    combined = floor_completed_minute(combined, now)
    if combined.empty:
        raise RuntimeError("No completed live NIFTY minute bars available")

    # Keep a rolling repository cache built only from live-provider data.
    cache_days = int(os.getenv("NIFTY_CACHE_DAYS", "0"))
    if cache_days > 0:
        cutoff_day = now.tz_localize(None).normalize() - pd.Timedelta(days=max(7, cache_days))
        combined = combined[combined.timestamp >= cutoff_day].copy().reset_index(drop=True)
    save_csv(combined, cache_path)

    warnings = data_warnings(combined, now)

    # Session notices must work even during first-day EMA warm-up.
    state = load_state(state_path)
    handle_session_notices(now, combined, state)

    strategy_data = filter_strategy_minutes(combined)
    if len(strategy_data) < 500:
        # Warm-up phase: cache data now, but do not invent a strategy state.
        state["engine_sha256"] = engine_sha
        state["updated_at_ist"] = str(now)
        state["warmup_rows"] = int(len(strategy_data))
        save_state(state_path, state)
        warm_meta = {"phase": live_phase(now), "last_timestamp": pd.Timestamp(combined.timestamp.iloc[-1]), "last_close": float(combined.close.iloc[-1]), "open_count": 0}
        write_live_excel_report(report_path, [], combined, warm_meta, engine_sha)
        telegram_send(
            f"🟡 NIFTY LIVE WARM-UP\n"
            f"Strategy minutes available: {len(strategy_data)}\n"
            f"Need more live-source warm-up before strategy signals are enabled.",
            quiet=True,
        )
        print(json.dumps({
            "engine": ENGINE_VERSION,
            "mode": "LIVE_ONLY_WARMUP",
            "strategy_rows": len(strategy_data),
            "cache_rows": len(combined),
        }, indent=2))
        return 0

    trades, meta = build_live_snapshot(strategy, strategy_data)

    # Live quote may continue after the 15:15 spot-signal freeze.
    today = now.tz_localize(None).date()
    tz = combined[combined.timestamp.dt.date == today]
    if len(tz):
        qr = tz.iloc[-1]
        meta["live_quote_time"] = pd.Timestamp(qr.timestamp)
        meta["live_quote"] = float(qr.close)
    else:
        meta["live_quote_time"] = pd.Timestamp(combined.timestamp.iloc[-1])
        meta["live_quote"] = float(combined.close.iloc[-1])
    meta["phase"] = live_phase(now)

    seen_signals: List[str] = list(state.get("seen_signals", []))
    seen_entries: List[str] = list(state.get("seen_entries", []))
    seen_exits: List[str] = list(state.get("seen_exits", []))

    lookback = pd.Timedelta(minutes=int(args.alert_lookback_min))
    last_ts = pd.Timestamp(meta["last_timestamp"])
    cutoff = last_ts - lookback
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

    events = []
    for t in new_entries:
        events.append((t.entry_time, 0, "ENTRY", t))
    for t in new_exits:
        events.append((t.exit_time, 1, "EXIT", t))
    events.sort(key=lambda x: (x[0], x[1]))

    for _, _, typ, t in events:
        # No new spot-strategy entry should originate after the strategy cutoff.
        if typ == "ENTRY" and t.entry_time.time() > STRATEGY_LAST:
            continue
        if typ == "ENTRY":
            sk = t.entry_key()
            if sk not in seen_signals:
                telegram_send(signal_message(t))
                log_trade_event(event_log, "SIGNAL", t, now)
                bounded_add(seen_signals, sk)
            telegram_send(entry_message(t))
        else:
            telegram_send(exit_message(t))
        log_trade_event(event_log, typ, t, now)

    # 30-minute strategy/status cadence.
    slot = status_slot(now, int(args.status_minutes))
    if slot and slot != state.get("last_status_slot"):
        telegram_send(status_message(meta, trades, warnings), quiet=False)
        log_status(status_log, meta, now, slot)
        state["last_status_slot"] = slot

    # CAS + derivative-management phase.
    handle_cas_management(now, combined, trades, state, cas_log)

    warn_sig = " | ".join(warnings)
    if warnings and warn_sig != state.get("last_data_warning"):
        telegram_send(
            f"⚠️ NIFTY {ENGINE_VERSION} LIVE DATA WARNING\n" +
            "\n".join(f"• {x}" for x in warnings)
        )
        state["last_data_warning"] = warn_sig

    if first_run:
        telegram_send(
            f"🤖 NIFTY {ENGINE_VERSION} LIVE-ONLY RUNNER INITIALIZED\n"
            f"Engine SHA: {engine_sha[:16]}…\n"
            f"Provider: {args.provider}\n"
            f"No static historical NIFTY CSV is required.\n"
            f"Live cache rows: {len(combined)}"
        )
        state["initialized"] = True

    write_live_excel_report(report_path, trades, combined, meta, engine_sha)

    state["seen_signals"] = seen_signals
    state["seen_entries"] = seen_entries
    state["seen_exits"] = seen_exits
    state["last_strategy_bar"] = str(meta["last_timestamp"])
    state["last_live_quote_time"] = str(meta.get("live_quote_time", ""))
    state["last_live_quote"] = meta.get("live_quote")
    state["engine_sha256"] = engine_sha
    state["updated_at_ist"] = str(now)
    state["phase"] = meta.get("phase")
    save_state(state_path, state)

    print(json.dumps({
        "engine": ENGINE_VERSION,
        "engine_sha256": engine_sha,
        "mode": "LIVE_ONLY",
        "provider": args.provider,
        "phase": meta.get("phase"),
        "last_strategy_bar": str(meta["last_timestamp"]),
        "live_quote_time": str(meta.get("live_quote_time")),
        "live_quote": meta.get("live_quote"),
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
        description="NIFTY live-only GitHub + Telegram + CAS runner"
    )
    ap.add_argument(
        "--engine",
        default=os.getenv("NIFTY_ENGINE_PATH", "NIFTY_RULES_ONLY_ENGINE.py"),
    )
    ap.add_argument(
        "--provider",
        choices=["yahoo","csv","none"],
        default=os.getenv("NIFTY_PROVIDER", "yahoo"),
    )
    ap.add_argument(
        "--live-csv",
        default=os.getenv("NIFTY_LIVE_CSV"),
        help="Continuously updated live 1m CSV for provider=csv",
    )
    ap.add_argument(
        "--state",
        default=os.getenv("NIFTY_STATE_PATH", "runtime/nifty_live_state.json"),
    )
    ap.add_argument(
        "--cache",
        default=os.getenv("NIFTY_LIVE_CACHE", "runtime/nifty_live_minutes.csv"),
    )
    ap.add_argument(
        "--event-log",
        default=os.getenv("NIFTY_EVENT_LOG", "logs/nifty_trade_events.csv"),
    )
    ap.add_argument(
        "--status-log",
        default=os.getenv("NIFTY_STATUS_LOG", "logs/nifty_status_30m.csv"),
    )
    ap.add_argument(
        "--cas-log",
        default=os.getenv("NIFTY_CAS_LOG", "logs/nifty_cas_management.csv"),
    )
    ap.add_argument(
        "--error-log",
        default=os.getenv("NIFTY_ERROR_LOG", "logs/nifty_errors.csv"),
    )
    ap.add_argument(
        "--excel-report",
        default=os.getenv("NIFTY_EXCEL_REPORT", "reports/NIFTY_LIVE_Trade_Report.xlsx"),
    )
    ap.add_argument(
        "--alert-lookback-min",
        type=int,
        default=int(os.getenv("NIFTY_ALERT_LOOKBACK_MIN", "15")),
    )
    ap.add_argument(
        "--status-minutes",
        type=int,
        default=int(os.getenv("NIFTY_STATUS_MINUTES", "30")),
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = build_args()
    try:
        raise SystemExit(run_once(args))
    except Exception as exc:
        try:
            elog = Path(getattr(args, "error_log", "logs/nifty_errors.csv"))
            append_csv_row(elog, {
                "logged_at_ist": str(now_ist()),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            telegram_send(
                f"🚨 NIFTY LIVE RUNNER ERROR\n"
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            raise

