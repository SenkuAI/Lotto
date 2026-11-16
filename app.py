"""Lotto Lab - big, simple PCSO lotto picks. Built with Streamlit.

Run locally:   streamlit run app.py

For every game (and every draw time) the page shows:
  * the latest official result,
  * the pick we made for that draw and how it matched,
  * the pick for the NEXT draw.

Picks are calculated only from past results, so they stay the same until a
new result arrives - then the old pick is checked against it and a fresh one
is made. The pick blends several signals (long-run frequency, recent trend,
short-term trend, overdue numbers) and a balance filter.

Every draw is random, so no method can guarantee a win. Not affiliated with PCSO.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Lotto Lab", page_icon="🎱", layout="centered", initial_sidebar_state="collapsed")

INK, MUTED, CARD, LINE = "#464a58", "#7b7f90", "#fcfbf8", "#e3ded4"
PEACH, SAGE, LAVENDER = "#f0c9a6", "#a8d1c0", "#c3c9ee"

st.markdown(
    f"""
    <style>
    html {{font-size: 19px;}}
    .block-container {{max-width: 820px; padding-top: 2rem;}}
    h1 {{color:{INK}; font-weight:700; font-size:2.4rem !important}}
    .card {{background:{CARD}; border:1px solid {LINE}; border-radius:22px; padding:1.3rem 1.5rem; margin-bottom:1.1rem}}
    .card .head {{color:{INK}; font-size:1.45rem; font-weight:700; margin-bottom:.6rem}}
    .card .label {{color:{MUTED}; font-size:.95rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; margin-top:.9rem}}
    .card .note {{color:{MUTED}; font-size:1.02rem}}
    .balls {{display:flex; gap:.55rem; flex-wrap:wrap; margin:.35rem 0}}
    .ball {{display:inline-flex; align-items:center; justify-content:center; width:3.3rem; height:3.3rem;
            border-radius:50%; font-weight:700; color:{INK}; font-size:1.55rem; border:1px solid rgba(0,0,0,.08)}}
    .ball.pick {{background:radial-gradient(circle at 30% 25%, #fff7e8, {PEACH})}}
    .ball.result {{background:radial-gradient(circle at 30% 25%, #f6f7ff, {LAVENDER})}}
    .ball.small {{width:2.4rem; height:2.4rem; font-size:1.15rem}}
    .ball.hit {{background:radial-gradient(circle at 30% 25%, #f1faf5, {SAGE}); border-color:#8ebba9}}
    .ball.miss {{background:#f1eee8; color:#9a9dab}}
    .backup {{color:{MUTED}; font-size:1.05rem; margin-top:.15rem}}
    .foot {{color:{MUTED}; font-size:.9rem}}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Games
# --------------------------------------------------------------------------- #
DATA_PATH = Path(__file__).parent / "data" / "seed_draws.csv.gz"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "lottotayo/pcso-lotto-results"
NUM_COLS = [f"n{i}" for i in range(1, 7)]
LOOKBACK = 300  # past draws used to score numbers


@dataclass(frozen=True)
class Game:
    key: str
    label: str
    picks: int
    low: int
    high: int
    ordered: bool  # True: position matters (digit games, EZ2)

    @property
    def size(self) -> int:
        return self.high - self.low + 1

    @property
    def wide(self) -> bool:
        return self.high > 9


GAMES: list[Game] = [
    Game("6-58", "Ultra Lotto 6/58", 6, 1, 58, False),
    Game("6-55", "Grand Lotto 6/55", 6, 1, 55, False),
    Game("6-49", "Super Lotto 6/49", 6, 1, 49, False),
    Game("6-45", "Mega Lotto 6/45", 6, 1, 45, False),
    Game("6-42", "Lotto 6/42", 6, 1, 42, False),
    Game("6d", "6D Lotto", 6, 0, 9, True),
    Game("4d", "4D Lotto", 4, 0, 9, True),
    Game("swertres", "Swertres 3D", 3, 0, 9, True),
    Game("ez2", "EZ2 2D", 2, 1, 31, True),
]
BY_KEY = {g.key: g for g in GAMES}

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["game"] = df["game"].astype(str).str.strip().str.lower()
    df = df[df["game"].isin(BY_KEY)].copy()
    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce")
    df["draw_time"] = df["draw_time"].fillna("").astype(str).str[:5]
    for col in NUM_COLS + ["jackpot_php", "winners"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["draw_date"])

    valid = pd.Series(False, index=df.index)
    for g in GAMES:
        block = df.loc[df["game"] == g.key, NUM_COLS[: g.picks]]
        ok = block.notna().all(axis=1) & ((block >= g.low) & (block <= g.high)).all(axis=1)
        valid.loc[ok.index[ok]] = True
    df = df[valid].copy()

    clock = df["draw_time"].where(df["draw_time"].str.match(r"^\d{2}:\d{2}$"), "00:00")
    df["ts"] = df["draw_date"] + pd.to_timedelta(clock + ":00")
    df["winners"] = df["winners"].fillna(0).astype(int)
    df = df.drop_duplicates(["game", "draw_date", "draw_time"], keep="last")
    return df.sort_values("ts", ascending=False).reset_index(drop=True)


def fetch_live(pages: int = 5) -> pd.DataFrame:
    rows: list[dict] = []
    for page in range(pages):
        resp = requests.get(
            HF_ROWS_URL,
            params={"dataset": HF_DATASET, "config": "default", "split": "train", "offset": page * 100, "length": 100},
            timeout=15,
        )
        resp.raise_for_status()
        rows += [item["row"] for item in resp.json()["rows"]]
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner="Loading latest results...")
def get_data() -> tuple[pd.DataFrame, str]:
    frames = [pd.read_csv(DATA_PATH)]
    status = "bundled data"
    try:
        frames.append(fetch_live())
        status = "live data"
    except Exception as exc:
        status = f"bundled data (live feed unavailable: {type(exc).__name__})"
    return normalise(pd.concat(frames, ignore_index=True)), status


def picks(df: pd.DataFrame, g: Game) -> np.ndarray:
    return df[NUM_COLS[: g.picks]].to_numpy(dtype=int)


# --------------------------------------------------------------------------- #
# The strategy: every signal blended into one score
# --------------------------------------------------------------------------- #
# long-run frequency, recent trend (50 draws), short trend (20 draws), overdue
W_FREQ, W_R50, W_R20, W_DUE = 0.30, 0.25, 0.15, 0.30


def zs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def blend_scores(prev: np.ndarray, g: Game) -> np.ndarray:
    """Score each value using past draws (newest first). Higher = better.

    Unordered games -> shape (size,). Ordered games -> (picks, size).
    """
    D = len(prev)
    if g.ordered:
        out = np.zeros((g.picks, g.size))
        grid = np.arange(g.size)
        for i in range(g.picks):
            col = prev[:, i] - g.low
            seen = col[:, None] == grid[None, :]
            gap = np.where(seen.any(0), seen.argmax(0), D)
            out[i] = (
                W_FREQ * zs(np.bincount(col, minlength=g.size))
                + W_R50 * zs(np.bincount(col[:50], minlength=g.size))
                + W_R20 * zs(np.bincount(col[:20], minlength=g.size))
                + W_DUE * zs(gap)
            )
        return out
    values = np.arange(g.low, g.high + 1)
    present = (prev[:, :, None] == values[None, None, :]).any(axis=1)
    gap = np.where(present.any(0), present.argmax(0), D)

    def count(rows: np.ndarray) -> np.ndarray:
        return np.bincount((rows - g.low).ravel(), minlength=g.size)

    return W_FREQ * zs(count(prev)) + W_R50 * zs(count(prev[:50])) + W_R20 * zs(count(prev[:20])) + W_DUE * zs(gap)


@lru_cache(maxsize=None)
def combo_index(m: int, k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(m), k)))


def make_picks(prev: np.ndarray, g: Game, n: int = 3) -> list[list[int]]:
    """Best line first, then backups. Lotto lines must also look 'typical':
    balanced odd/even and high/low, normal sum, no long runs of consecutive numbers."""
    score = blend_scores(prev, g)
    if g.ordered:
        best = score.argmax(axis=1)
        top2 = np.sort(score, axis=1)
        margin = top2[:, -1] - top2[:, -2]
        runner = np.argsort(-score, axis=1)[:, 1]
        lines = [best + g.low]
        for pos in np.argsort(margin)[: n - 1]:  # flip the least-certain positions
            alt = best.copy()
            alt[pos] = runner[pos]
            lines.append(alt + g.low)
        return [ln.tolist() for ln in lines]

    k = g.picks
    order = np.argsort(-score)[:14]
    idx = combo_index(len(order), k)
    nums = order[idx] + g.low
    total = score[order][idx].sum(axis=1)
    sums = nums.sum(axis=1)
    odd = (nums % 2).sum(axis=1)
    high = (nums > (g.low + g.high) / 2).sum(axis=1)
    consecutive = (np.diff(np.sort(nums, axis=1), axis=1) == 1).sum(axis=1)
    lo, hi = np.percentile(prev.sum(axis=1), [10, 90])
    ok = (np.abs(odd - k / 2) <= 1) & (np.abs(high - k / 2) <= 1) & (sums >= lo) & (sums <= hi) & (consecutive <= 1)
    if not ok.any():
        ok[:] = True
    total = np.where(ok, total, -np.inf)
    best = np.argsort(-total)[:n]
    return [sorted(nums[i].tolist()) for i in best if np.isfinite(total[i])]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def fmt(n: int, g: Game) -> str:
    return f"{n:02d}" if g.wide else str(n)


def balls(line, g: Game, cls: str, hits: list[bool] | None = None, small: bool = False) -> str:
    out = []
    for i, n in enumerate(line):
        c = f"ball {cls}" + (" small" if small else "")
        if hits is not None:
            c = "ball small " + ("hit" if hits[i] else "miss")
        out.append(f"<span class='{c}'>{fmt(int(n), g)}</span>")
    return f"<div class='balls'>{''.join(out)}</div>"


def peso(v) -> str:
    return f"₱{v:,.0f}" if pd.notna(v) else "—"


def clock12(t: str) -> str:
    try:
        return pd.to_datetime(t, format="%H:%M").strftime("%I:%M %p").lstrip("0")
    except Exception:
        return t


def hit_flags(pick: list[int], actual: np.ndarray, g: Game) -> list[bool]:
    if g.ordered:
        return [int(p) == int(a) for p, a in zip(pick, actual)]
    return [p in set(actual.tolist()) for p in pick]


def slot_card(g: Game, slot: pd.DataFrame) -> str:
    arr = picks(slot, g)
    latest = slot.iloc[0]
    when = f"{latest['ts']:%a, %b %d}"
    title = g.label + (f" · {clock12(latest['draw_time'])}" if latest["draw_time"] else "")

    prev_pick = make_picks(arr[1 : 1 + LOOKBACK], g, 1)[0]
    flags = hit_flags(prev_pick, arr[0], g)
    next_picks = make_picks(arr[:LOOKBACK], g, 3)

    prize_word = "Top prize" if g.ordered else "Jackpot"
    winners = int(latest["winners"])
    prize = f"{prize_word} {peso(latest['jackpot_php'])} · {winners:,} winner{'' if winners == 1 else 's'}"

    backups = "".join(
        f"<div class='backup'>Backup &nbsp;{' &nbsp;'.join(fmt(n, g) for n in line)}</div>" for line in next_picks[1:]
    )
    return (
        f"<div class='card'><div class='head'>{title}</div>"
        f"<div class='label'>Latest result · {when}</div>{balls(arr[0], g, 'result')}"
        f"<div class='note'>{prize}</div>"
        f"<div class='label'>Our last pick · matched {sum(flags)} of {g.picks}</div>"
        f"{balls(prev_pick, g, '', flags, small=True)}"
        f"<div class='label'>Pick for the next draw</div>{balls(next_picks[0], g, 'pick')}{backups}</div>"
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
df_all, data_status = get_data()
if df_all.empty:
    st.error("No draw data could be loaded.")
    st.stop()

st.title("🎱 Lotto Lab")
st.markdown(
    "<span class='foot'>Picks for every game. They stay the same until a new result comes in, "
    "then the old pick is checked against it and a new one is made.</span>",
    unsafe_allow_html=True,
)

for g in GAMES:
    hist = df_all[df_all["game"] == g.key]
    recent_times = hist.head(60)["draw_time"].unique().tolist()
    times = sorted(t for t in recent_times if (hist["draw_time"] == t).sum() > 60) or [None]
    for t in times:
        slot = hist if t is None else hist[hist["draw_time"] == t]
        if len(slot) < 60:
            continue
        st.markdown(slot_card(g, slot), unsafe_allow_html=True)

st.divider()
left, right = st.columns([3, 1])
left.markdown(
    f"<span class='foot'>Source: {data_status} · latest draw {df_all['draw_date'].max():%b %d, %Y}. "
    "Results from PCSO via lottotayo/pcso-lotto-results; not affiliated with PCSO. "
    "Every draw is random, so no method can guarantee a win. Play responsibly.</span>",
    unsafe_allow_html=True,
)
if right.button("Refresh"):
    get_data.clear()
    st.rerun()
