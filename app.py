"""Lotto Lab - one-page PCSO lotto helper built with Streamlit.

Run locally:   streamlit run app.py

What it does
  1. Shows the latest draw for the game you pick.
  2. Builds ONE "Smart pick": a blend of three signals (long-run frequency,
     recent trend, and how overdue a number is) with a balance filter.
  3. Back-tests that exact method on past draws and compares it with what
     pure chance predicts - so you can see for yourself how it performs.

Every draw is independent, so no method can raise your odds. Not affiliated
with PCSO.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Lotto Lab", page_icon="🎱", layout="centered", initial_sidebar_state="collapsed")

# Soft, low-contrast palette
INK, MUTED, CARD, LINE = "#5b5f6e", "#8d90a0", "#fbfaf7", "#e6e1d8"
LAVENDER, PEACH, SAGE = "#a9b1de", "#efc9a9", "#a8cdbd"

st.markdown(
    f"""
    <style>
    .block-container {{max-width: 860px; padding-top: 2rem;}}
    h1, h2, h3 {{color:{INK}; font-weight:600; letter-spacing:-.01em}}
    .card {{background:{CARD}; border:1px solid {LINE}; border-radius:18px; padding:1.1rem 1.3rem; margin-bottom:.8rem}}
    .card .title {{color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.2rem}}
    .balls {{display:flex; gap:.45rem; flex-wrap:wrap; margin:.45rem 0 .35rem}}
    .ball {{display:inline-flex; align-items:center; justify-content:center; width:2.5rem; height:2.5rem;
            border-radius:50%; font-weight:600; color:{INK}; font-size:1rem;
            background:radial-gradient(circle at 30% 25%, #fff6e4, {PEACH}); border:1px solid #e7cdb3}}
    .ball.sage {{background:radial-gradient(circle at 30% 25%, #f1faf5, {SAGE}); border-color:#98bcab}}
    .small {{color:{MUTED}; font-size:.85rem}}
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
    def combos(self) -> int:
        return self.size**self.picks if self.ordered else math.comb(self.size, self.picks)

    @property
    def wide(self) -> bool:
        return self.high > 9


GAMES: dict[str, Game] = {
    g.key: g
    for g in [
        Game("6-58", "Ultra 6/58", 6, 1, 58, False),
        Game("6-55", "Grand 6/55", 6, 1, 55, False),
        Game("6-49", "Super 6/49", 6, 1, 49, False),
        Game("6-45", "Mega 6/45", 6, 1, 45, False),
        Game("6-42", "Lotto 6/42", 6, 1, 42, False),
        Game("6d", "6D", 6, 0, 9, True),
        Game("4d", "4D", 4, 0, 9, True),
        Game("swertres", "Swertres 3D", 3, 0, 9, True),
        Game("ez2", "EZ2", 2, 1, 31, True),
    ]
}

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["game"] = df["game"].astype(str).str.strip().str.lower()
    df = df[df["game"].isin(GAMES)].copy()
    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce")
    df["draw_time"] = df["draw_time"].fillna("").astype(str).str[:5]
    for col in NUM_COLS + ["jackpot_php", "winners"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["draw_date"])

    valid = pd.Series(False, index=df.index)
    for key, g in GAMES.items():
        block = df.loc[df["game"] == key, NUM_COLS[: g.picks]]
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


@st.cache_data(ttl=3600, show_spinner="Loading draw results...")
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
# The one strategy: blended score -> best combination
# --------------------------------------------------------------------------- #
W_FREQ, W_RECENT, W_DUE = 0.4, 0.3, 0.3  # long-run frequency, recent trend, overdue


def zs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def blend_scores(prev: np.ndarray, g: Game) -> np.ndarray:
    """Score every value from past draws (newest first). Higher = better.

    Unordered games return shape (size,); ordered games return (picks, size).
    """
    D = len(prev)
    if g.ordered:
        out = np.zeros((g.picks, g.size))
        grid = np.arange(g.size)
        for i in range(g.picks):
            col = prev[:, i] - g.low
            freq = np.bincount(col, minlength=g.size)
            recent = np.bincount(col[:50], minlength=g.size)
            seen = col[:, None] == grid[None, :]
            gap = np.where(seen.any(0), seen.argmax(0), D)
            out[i] = W_FREQ * zs(freq) + W_RECENT * zs(recent) + W_DUE * zs(gap)
        return out
    values = np.arange(g.low, g.high + 1)
    freq = np.bincount((prev - g.low).ravel(), minlength=g.size)
    recent = np.bincount((prev[:50] - g.low).ravel(), minlength=g.size)
    present = (prev[:, :, None] == values[None, None, :]).any(axis=1)
    gap = np.where(present.any(0), present.argmax(0), D)
    return W_FREQ * zs(freq) + W_RECENT * zs(recent) + W_DUE * zs(gap)


@lru_cache(maxsize=None)
def combo_index(m: int, k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(m), k)))


def smart_lines(prev: np.ndarray, g: Game, n: int = 1) -> list[list[int]]:
    """Best-scoring combination(s) that also look 'typical' (balanced odd/even, normal sum)."""
    score = blend_scores(prev, g)
    if g.ordered:
        return [(score.argmax(axis=1) + g.low).tolist()]
    k = g.picks
    order = np.argsort(-score)[:12]
    idx = combo_index(len(order), k)
    nums = order[idx] + g.low
    total = score[order][idx].sum(axis=1)
    sums = nums.sum(axis=1)
    odd = (nums % 2).sum(axis=1)
    lo, hi = np.percentile(prev.sum(axis=1), [10, 90])
    ok = (np.abs(odd - k / 2) <= 1) & (sums >= lo) & (sums <= hi)
    if not ok.any():
        ok[:] = True
    total = np.where(ok, total, -np.inf)
    best = np.argsort(-total)[:n]
    return [sorted(nums[i].tolist()) for i in best if np.isfinite(total[i])]


@st.cache_data(show_spinner="Back-testing...")
def backtest(arr: np.ndarray, game_key: str, tests: int, lookback: int):
    """Walk back through history: for each past draw, build the pick using only
    the draws BEFORE it, then count how many numbers it matched."""
    g = GAMES[game_key]
    n = max(0, min(tests, len(arr) - 40))
    matches = np.zeros(n, dtype=int)
    lines: list[list[int]] = []
    for t in range(n):
        prev = arr[t + 1 : t + 1 + lookback]
        line = smart_lines(prev, g)[0]
        actual = arr[t]
        matches[t] = int((np.array(line) == actual).sum()) if g.ordered else len(set(line) & set(actual.tolist()))
        lines.append(line)
    return matches, lines


def chance_model(g: Game) -> tuple[np.ndarray, float, float]:
    """Distribution of matches for a random pick: (probabilities, mean, variance)."""
    k, N = g.picks, g.size
    if g.ordered:
        p = 1 / N
        probs = np.array([math.comb(k, j) * p**j * (1 - p) ** (k - j) for j in range(k + 1)])
        return probs, k * p, k * p * (1 - p)
    probs = np.array([math.comb(k, j) * math.comb(N - k, k - j) / math.comb(N, k) for j in range(k + 1)])
    var = k * (k / N) * (1 - k / N) * (N - k) / (N - 1)
    return probs, k * k / N, var


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def fmt_num(n: int, g: Game) -> str:
    return f"{n:02d}" if g.wide else str(n)


def balls(row, g: Game, cls: str = "") -> str:
    inner = "".join(f"<span class='ball {cls}'>{fmt_num(int(n), g)}</span>" for n in row)
    return f"<div class='balls'>{inner}</div>"


def peso(v) -> str:
    return f"₱{v:,.0f}" if pd.notna(v) else "—"


def soften(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        legend=dict(orientation="h", y=1.12, x=0),
        barmode="group",
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
df_all, data_status = get_data()
if df_all.empty:
    st.error("No draw data could be loaded.")
    st.stop()

st.title("🎱 Lotto Lab")
st.markdown("<span class='small'>One smart pick, checked against real past results.</span>", unsafe_allow_html=True)

choice = st.pills("Game", list(GAMES), format_func=lambda k: GAMES[k].label, default="6-58", label_visibility="collapsed")
game = GAMES[choice or "6-58"]

with st.expander("Settings"):
    lookback = st.slider("Past draws used to score numbers", 100, 1000, 300, step=50)
    tests = st.slider("Past draws to test the pick on", 50, 400, 150, step=50)

hist = df_all[df_all["game"] == game.key]  # newest first
arr = picks(hist, game)
if len(arr) < 60:
    st.warning("Not enough history for this game yet.")
    st.stop()

latest = hist.iloc[0]
pick_lines = smart_lines(arr[:lookback], game, n=1 if game.ordered else 3)

left, right = st.columns(2)
with left:
    when = f"{latest['ts']:%a, %b %d %Y}" + (f" · {latest['draw_time']}" if latest["draw_time"] else "")
    prize = f"Top prize {peso(latest['jackpot_php'])}" if game.ordered else f"Jackpot {peso(latest['jackpot_php'])}"
    st.markdown(
        f"<div class='card'><div class='title'>Latest draw</div>{balls(arr[0], game, 'sage')}"
        f"<div class='small'>{when}<br>{prize} · {latest['winners']:,} winner{'' if latest['winners'] == 1 else 's'}</div></div>",
        unsafe_allow_html=True,
    )
with right:
    body = balls(pick_lines[0], game)
    runners = "".join(f"<div class='small'>or &nbsp;{' - '.join(fmt_num(n, game) for n in line)}</div>" for line in pick_lines[1:])
    st.markdown(
        f"<div class='card'><div class='title'>Smart pick</div>{body}{runners}"
        f"<div class='small' style='margin-top:.3rem'>Blend of frequent, trending &amp; overdue numbers</div></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"<span class='small'>Any single line has a 1 in {game.combos:,} chance at the top prize, "
    f"whichever numbers you choose.</span>",
    unsafe_allow_html=True,
)

# ---- Back-test ------------------------------------------------------------ #
st.subheader("How would it have done?")
matches, lines = backtest(arr, game.key, tests, lookback)
n = len(matches)
probs, exp_mean, var = chance_model(game)
mean = float(matches.mean())
z = (mean - exp_mean) / math.sqrt(var / n)

c1, c2, c3 = st.columns(3)
c1.metric("Smart pick avg matches", f"{mean:.2f}")
c2.metric("Pure chance expects", f"{exp_mean:.2f}")
c3.metric("Best single draw", f"{int(matches.max())} of {game.picks}")

if abs(z) < 2:
    verdict = f"Over the last {n} draws the smart pick landed within normal luck of random picks. That is what we'd expect: the draw has no memory."
elif z > 0:
    verdict = (
        f"The pick scored a little above chance over the last {n} draws, but with this many numbers to look at "
        "some streaks happen by luck and they don't carry forward."
    )
else:
    verdict = f"The pick scored a little below chance over the last {n} draws. Again, just luck of the window."
st.markdown(f"<div class='card small'>{verdict}</div>", unsafe_allow_html=True)

tally = np.bincount(matches, minlength=game.picks + 1) / n
fig = go.Figure()
fig.add_bar(x=list(range(game.picks + 1)), y=tally, name="Smart pick", marker_color=LAVENDER)
fig.add_bar(x=list(range(game.picks + 1)), y=probs, name="Pure chance", marker_color=PEACH)
fig.update_layout(xaxis_title="Numbers matched in a draw", yaxis_title="Share of draws", yaxis_tickformat=".0%")
st.plotly_chart(soften(fig), width="stretch")

st.markdown("**Recent draws vs the pick it would have made**")
recent = hist.head(8)
table = pd.DataFrame(
    {
        "Date": recent["draw_date"].dt.strftime("%b %d").to_numpy(),
        "Actual result": [" - ".join(fmt_num(int(x), game) for x in r) for r in arr[:8]],
        "Smart pick": [" - ".join(fmt_num(x, game) for x in ln) for ln in lines[:8]],
        "Matched": matches[:8],
    }
)
st.dataframe(table, hide_index=True, width="stretch")

st.divider()
col_a, col_b = st.columns([3, 1])
col_a.markdown(
    f"<span class='small'>Source: {data_status} · latest draw {df_all['draw_date'].max():%b %d, %Y} · "
    "PCSO results via lottotayo/pcso-lotto-results. Not affiliated with PCSO. For entertainment only; play responsibly.</span>",
    unsafe_allow_html=True,
)
if col_b.button("Refresh data"):
    get_data.clear()
    st.rerun()
