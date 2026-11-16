"""Lotto Lab - a PCSO lotto results explorer built with Streamlit.

Run locally:   streamlit run app.py
Data:          data/seed_draws.csv.gz (bundled) + live top-up from Hugging Face
               (lottotayo/pcso-lotto-results). If the live fetch fails the app
               keeps working from the bundled file.

Everything here is descriptive statistics. Every draw is independent, so past
results cannot predict future ones. Not affiliated with PCSO.
"""

from __future__ import annotations

import math
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Lotto Lab", page_icon="🎱", layout="wide")

st.markdown(
    """
    <style>
    .balls{display:flex;gap:.4rem;flex-wrap:wrap;margin:.35rem 0 .55rem}
    .ball{display:inline-flex;align-items:center;justify-content:center;
          width:2.4rem;height:2.4rem;border-radius:50%;font-weight:700;
          font-size:1rem;color:#2b1d00;box-shadow:0 2px 6px rgba(0,0,0,.35);
          background:radial-gradient(circle at 30% 30%,#ffe9a8,#f5b82e 60%,#c98a0a)}
    .ball.digit{color:#fff;
          background:radial-gradient(circle at 30% 30%,#c9ddff,#4f8cff 60%,#2457c5)}
    .meta{opacity:.7;font-size:.85rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Game definitions
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
    jackpot: bool  # True: rolling jackpot, False: fixed top prize

    @property
    def size(self) -> int:
        return self.high - self.low + 1

    @property
    def combos(self) -> int:
        return self.size**self.picks if self.ordered else math.comb(self.size, self.picks)

    @property
    def wide(self) -> bool:
        """Numbers with two digits (lotto balls / EZ2) vs single digits."""
        return self.high > 9


GAMES: dict[str, Game] = {
    g.key: g
    for g in [
        Game("6-58", "Ultra Lotto 6/58", 6, 1, 58, False, True),
        Game("6-55", "Grand Lotto 6/55", 6, 1, 55, False, True),
        Game("6-49", "Super Lotto 6/49", 6, 1, 49, False, True),
        Game("6-45", "Mega Lotto 6/45", 6, 1, 45, False, True),
        Game("6-42", "Lotto 6/42", 6, 1, 42, False, True),
        Game("6d", "6D Lotto", 6, 0, 9, True, False),
        Game("4d", "4D Lotto", 4, 0, 9, True, False),
        Game("swertres", "Swertres (3D)", 3, 0, 9, True, False),
        Game("ez2", "EZ2 (2D)", 2, 1, 31, True, False),
    ]
}

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw rows: types, valid ranges, de-duplication, newest first."""
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
        mask = df["game"] == key
        block = df.loc[mask, NUM_COLS[: g.picks]]
        ok = block.notna().all(axis=1) & ((block >= g.low) & (block <= g.high)).all(axis=1)
        valid.loc[ok.index[ok]] = True
    df = df[valid].copy()

    clock = df["draw_time"].where(df["draw_time"].str.match(r"^\d{2}:\d{2}$"), "00:00")
    df["ts"] = df["draw_date"] + pd.to_timedelta(clock + ":00")
    df["winners"] = df["winners"].fillna(0).astype(int)
    df = df.drop_duplicates(["game", "draw_date", "draw_time"], keep="last")
    return df.sort_values("ts", ascending=False).reset_index(drop=True)


def fetch_live(pages: int = 5) -> pd.DataFrame:
    """Newest rows from the public Hugging Face dataset (100 rows per page)."""
    rows: list[dict] = []
    for page in range(pages):
        resp = requests.get(
            HF_ROWS_URL,
            params={
                "dataset": HF_DATASET,
                "config": "default",
                "split": "train",
                "offset": page * 100,
                "length": 100,
            },
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
        status = "live feed"
    except Exception as exc:  # network down, rate limit, schema change...
        status = f"bundled data (live feed unavailable: {type(exc).__name__})"
    return normalise(pd.concat(frames, ignore_index=True)), status


def picks(df: pd.DataFrame, g: Game) -> np.ndarray:
    return df[NUM_COLS[: g.picks]].to_numpy(dtype=int)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def fmt_num(n: int, g: Game) -> str:
    return f"{n:02d}" if g.wide else str(n)


def fmt_draw(row, g: Game) -> str:
    return " - ".join(fmt_num(int(n), g) for n in row)


def balls_html(row, g: Game) -> str:
    cls = "ball" if g.wide and not g.ordered else "ball digit"
    inner = "".join(f"<span class='{cls}'>{fmt_num(int(n), g)}</span>" for n in row)
    return f"<div class='balls'>{inner}</div>"


def peso(value) -> str:
    return f"₱{value:,.0f}" if pd.notna(value) else "—"


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
    return fig


# --------------------------------------------------------------------------- #
# Sidebar & filtering
# --------------------------------------------------------------------------- #
df_all, data_status = get_data()
if df_all.empty:
    st.error("No draw data could be loaded.")
    st.stop()

st.sidebar.title("🎱 Lotto Lab")
game_key = st.sidebar.selectbox(
    "Game", list(GAMES), format_func=lambda k: GAMES[k].label, key="game"
)
game = GAMES[game_key]

RANGES = {"3 months": 90, "6 months": 182, "1 year": 365, "2 years": 730, "All time": None}
range_label = st.sidebar.select_slider("History window", list(RANGES), value="1 year")

game_all = df_all[df_all["game"] == game_key]
times = sorted(game_all["draw_time"].unique())
chosen_times = times
if len(times) > 1:
    chosen_times = st.sidebar.multiselect("Draw times", times, default=times) or times

view = game_all[game_all["draw_time"].isin(chosen_times)]
days = RANGES[range_label]
if days is not None and not view.empty:
    view = view[view["draw_date"] >= view["draw_date"].max() - pd.Timedelta(days=days)]

if st.sidebar.button("Refresh data", width="stretch"):
    get_data.clear()
    st.rerun()

st.sidebar.caption(
    f"Source: {data_status}. Latest draw in data: "
    f"{df_all['draw_date'].max():%b %d, %Y}."
)
st.sidebar.info(
    "Every draw is independent. These tools describe history; they can't "
    "predict the next draw. Play responsibly."
)

st.title("Lotto Lab")
st.caption("PCSO lotto results, statistics and odds - all in one place.")

if view.empty:
    st.warning("No draws match the current filters.")
    st.stop()

tab_latest, tab_stats, tab_prizes, tab_check, tab_pick = st.tabs(
    ["Latest results", "Statistics", "Prizes & winners", "Ticket checker", "Quick pick & odds"]
)

# --------------------------------------------------------------------------- #
# Tab 1 - latest results
# --------------------------------------------------------------------------- #
with tab_latest:
    newest = df_all.drop_duplicates("game")  # df_all is newest-first
    newest = newest.set_index("game")
    cols = st.columns(3)
    for i, (key, g) in enumerate(GAMES.items()):
        if key not in newest.index:
            continue
        row = newest.loc[key]
        with cols[i % 3], st.container(border=True):
            st.markdown(f"**{g.label}**")
            st.markdown(balls_html(picks(newest.loc[[key]], g)[0], g), unsafe_allow_html=True)
            when = f"{row['ts']:%a, %b %d %Y}" + (f" · {row['draw_time']}" if row["draw_time"] else "")
            st.markdown(f"<span class='meta'>{when}</span>", unsafe_allow_html=True)
            label = "Jackpot" if g.jackpot else "Top prize"
            if g.jackpot and row["winners"] == 0:
                st.markdown(f"{label} **{peso(row['jackpot_php'])}** · no winner, rolls over")
            else:
                st.markdown(
                    f"{label} **{peso(row['jackpot_php'])}** · {row['winners']:,} winner"
                    f"{'' if row['winners'] == 1 else 's'}"
                )

    st.subheader(f"History - {game.label}")
    limit = st.slider("Rows to show", 10, min(500, max(len(view), 11)), min(50, len(view)), step=10)
    shown = view.head(limit)
    table = pd.DataFrame(
        {
            "Date": shown["draw_date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "Time": shown["draw_time"].to_numpy(),
            "Result": [fmt_draw(r, game) for r in picks(shown, game)],
            "Jackpot" if game.jackpot else "Top prize": [peso(v) for v in shown["jackpot_php"]],
            "Winners": shown["winners"].to_numpy(),
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "Download filtered history (CSV)",
        view.drop(columns=["ts"]).to_csv(index=False).encode(),
        file_name=f"{game_key}_draws.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# Tab 2 - statistics
# --------------------------------------------------------------------------- #


def render_unordered_stats(v: pd.DataFrame, g: Game) -> None:
    arr = picks(v, g)
    draws, k = arr.shape
    values = np.arange(g.low, g.high + 1)
    counts = np.bincount((arr - g.low).ravel(), minlength=g.size)
    p = k / g.size
    expected = draws * p
    sd = math.sqrt(draws * p * (1 - p))
    z = (counts - expected) / sd if sd else np.zeros(g.size)
    present = (arr[:, :, None] == values[None, None, :]).any(axis=1)  # draws x numbers
    gap = np.where(present.any(axis=0), present.argmax(axis=0), draws)  # v is newest first

    stats = pd.DataFrame(
        {
            "Number": values,
            "Drawn": counts,
            "Expected": np.round(expected, 1),
            "z-score": np.round(z, 2),
            "Draws since last seen": gap,
        }
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Draws analysed", f"{draws:,}")
    m2.metric("Expected hits per number", f"{expected:.1f}")
    m3.metric("Numbers beyond ±2σ", f"{int((np.abs(z) > 2).sum())} of {g.size}")

    hot = set(stats.nlargest(6, "Drawn")["Number"])
    cold = set(stats.nsmallest(6, "Drawn")["Number"])
    colors = ["#f5b82e" if n in hot else "#4f8cff" if n in cold else "#6b7280" for n in values]
    fig = go.Figure(go.Bar(x=[fmt_num(n, g) for n in values], y=counts, marker_color=colors, name="Drawn"))
    fig.add_hline(y=expected, line_dash="dash", annotation_text="expected if fair")
    fig.update_layout(title="How often each number was drawn (gold = most, blue = least)")
    st.plotly_chart(style_fig(fig), width="stretch")
    st.caption(
        f"For a perfectly fair draw you'd expect about {0.0455 * g.size:.1f} of {g.size} numbers "
        "beyond ±2σ just by chance - so 'hot' and 'cold' numbers are mostly noise."
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Most drawn**")
    c1.dataframe(stats.nlargest(6, "Drawn")[["Number", "Drawn"]], hide_index=True)
    c2.markdown("**Least drawn**")
    c2.dataframe(stats.nsmallest(6, "Drawn")[["Number", "Drawn"]], hide_index=True)
    c3.markdown("**Longest since last seen**")
    c3.dataframe(stats.nlargest(6, "Draws since last seen")[["Number", "Draws since last seen"]], hide_index=True)

    left, right = st.columns(2)
    odd_total = int((values % 2).sum())
    observed = np.bincount((arr % 2).sum(axis=1), minlength=k + 1) / draws
    theory = [
        math.comb(odd_total, j) * math.comb(g.size - odd_total, k - j) / math.comb(g.size, k)
        for j in range(k + 1)
    ]
    fig_oe = go.Figure()
    fig_oe.add_bar(x=list(range(k + 1)), y=observed, name="Observed")
    fig_oe.add_bar(x=list(range(k + 1)), y=theory, name="Expected")
    fig_oe.update_layout(title="Odd numbers per draw", barmode="group", xaxis_title="Odd numbers in draw", yaxis_tickformat=".0%")
    left.plotly_chart(style_fig(fig_oe), width="stretch")
    fig_sum = px.histogram(x=arr.sum(axis=1), nbins=30, title="Sum of the drawn numbers")
    right.plotly_chart(style_fig(fig_sum), width="stretch")

    with st.expander("Full number table"):
        st.dataframe(stats, hide_index=True, width="stretch")


def render_ordered_stats(v: pd.DataFrame, g: Game) -> None:
    arr = picks(v, g)
    draws, k = arr.shape
    values = np.arange(g.low, g.high + 1)
    matrix = np.stack([np.bincount(arr[:, i] - g.low, minlength=g.size) for i in range(k)])
    repeats = float((np.array([len(set(r)) for r in arr]) < k).mean())
    expected_repeats = 1 - math.perm(g.size, k) / g.size**k

    m1, m2, m3 = st.columns(3)
    m1.metric("Draws analysed", f"{draws:,}")
    m2.metric("Expected hits per value / position", f"{draws / g.size:.1f}")
    m3.metric("Draws with a repeat", f"{repeats:.1%}", f"{repeats - expected_repeats:+.1%} vs fair ({expected_repeats:.1%})", delta_color="off")

    fig = px.imshow(
        matrix,
        x=[fmt_num(n, g) for n in values],
        y=[f"Position {i + 1}" for i in range(k)],
        color_continuous_scale="YlOrRd",
        aspect="auto",
        text_auto=g.size <= 10,
        title="How often each value appeared in each position",
    )
    st.plotly_chart(style_fig(fig, 300 if k <= 4 else 380), width="stretch")
    fig_sum = px.histogram(x=arr.sum(axis=1), nbins=30, title="Sum of the drawn numbers")
    st.plotly_chart(style_fig(fig_sum), width="stretch")


with tab_stats:
    st.subheader(f"{game.label} - {range_label.lower()}")
    if game.ordered:
        render_ordered_stats(view, game)
    else:
        render_unordered_stats(view, game)

# --------------------------------------------------------------------------- #
# Tab 3 - prizes & winners
# --------------------------------------------------------------------------- #
with tab_prizes:
    st.subheader(f"{game.label} - prizes & winners")
    chrono = view.sort_values("ts")
    if game.jackpot:
        wins = int((view["winners"] > 0).sum())
        latest = view.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest jackpot", peso(latest["jackpot_php"]))
        m2.metric("Highest in window", peso(view["jackpot_php"].max()))
        m3.metric("Jackpot won", f"{wins} time{'' if wins == 1 else 's'}")
        m4.metric("Avg draws per win", f"{len(view) / wins:.0f}" if wins else "—")
        fig = go.Figure(go.Scatter(x=chrono["ts"], y=chrono["jackpot_php"], mode="lines", name="Jackpot"))
        won = chrono[chrono["winners"] > 0]
        fig.add_trace(
            go.Scatter(x=won["ts"], y=won["jackpot_php"], mode="markers", name="Won", marker=dict(color="#ef4444", size=10))
        )
        fig.update_layout(title="Jackpot size over time (red = someone won)", yaxis_title="PHP")
        st.plotly_chart(style_fig(fig, 420), width="stretch")
    else:
        top = view["jackpot_php"].mode()
        m1, m2, m3 = st.columns(3)
        m1.metric("Typical top prize", peso(top.iloc[0]) if len(top) else "—")
        m2.metric("Avg winners per draw", f"{view['winners'].mean():,.1f}")
        m3.metric("Most winners in one draw", f"{int(view['winners'].max()):,}")
        monthly = (
            chrono.assign(month=chrono["draw_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month", as_index=False)["winners"]
            .mean()
        )
        fig = px.line(monthly, x="month", y="winners", markers=True, title="Average winners per draw, by month")
        st.plotly_chart(style_fig(fig, 420), width="stretch")

# --------------------------------------------------------------------------- #
# Tab 4 - ticket checker
# --------------------------------------------------------------------------- #


def parse_ticket(text: str, g: Game) -> list[int] | str:
    tokens = re.findall(r"\d+", text)
    if g.high == 9 and len(tokens) == 1 and len(tokens[0]) == g.picks:
        tokens = list(tokens[0])  # "448" -> 4 4 8
    nums = [int(t) for t in tokens]
    if len(nums) != g.picks:
        return f"Enter exactly {g.picks} numbers for {g.label}."
    if any(n < g.low or n > g.high for n in nums):
        return f"Numbers must be between {g.low} and {g.high}."
    if not g.ordered and len(set(nums)) != len(nums):
        return "Numbers can't repeat in this game."
    return nums


with tab_check:
    st.subheader(f"Check a ticket against all {len(game_all):,} {game.label} draws on file")
    example = " ".join(str(g) for g in range(game.low, game.low + game.picks))
    text = st.text_input("Your numbers", placeholder=f"e.g. {example}", key=f"ticket_{game_key}")
    if text:
        mine = parse_ticket(text, game)
        if isinstance(mine, str):
            st.error(mine)
        else:
            arr = picks(game_all, game)
            dates = game_all["draw_date"].dt.strftime("%Y-%m-%d") + " " + game_all["draw_time"]
            if game.ordered:
                exact = (arr == np.array(mine)).all(axis=1)
                anyorder = (np.sort(arr, axis=1) == np.sort(np.array(mine))).all(axis=1)
                c1, c2 = st.columns(2)
                c1.metric("Exact-order hits", int(exact.sum()))
                c2.metric("Any-order (rambolito) hits", int(anyorder.sum()))
                if anyorder.any():
                    st.dataframe(
                        pd.DataFrame({"Draw": dates[anyorder].to_numpy(), "Result": [fmt_draw(r, game) for r in arr[anyorder]]}),
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.info("That combination has never come up in the data.")
            else:
                k = game.picks
                matches = np.isin(arr, mine).sum(axis=1)
                tally = np.bincount(matches, minlength=k + 1)
                theory = [math.comb(k, j) * math.comb(game.size - k, k - j) / math.comb(game.size, k) for j in range(k + 1)]
                result = pd.DataFrame(
                    {
                        "Numbers matched": range(k + 1),
                        "Draws": tally,
                        "Share": [f"{t / len(arr):.2%}" for t in tally],
                        "Expected share": [f"{t:.2%}" for t in theory],
                    }
                )
                st.dataframe(result, hide_index=True, width="stretch")
                if tally[k]:
                    st.success(f"Jackpot match on: {', '.join(dates[matches == k])}")
                else:
                    st.info("This exact combination has never won - and it's exactly as likely as any other next draw.")
                best = matches >= k - 2
                if best.any():
                    near = pd.DataFrame(
                        {
                            "Draw": dates[best].to_numpy(),
                            "Matched": matches[best],
                            "Result": [fmt_draw(sorted(r), game) for r in arr[best]],
                        }
                    ).head(25)
                    st.markdown(f"**Closest draws ({k - 2}+ matches)**")
                    st.dataframe(near, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Tab 5 - quick pick & odds
# --------------------------------------------------------------------------- #


def make_line(g: Game, rng: secrets.SystemRandom, sums: tuple[float, float] | None) -> list[int]:
    if g.ordered:
        return [rng.randint(g.low, g.high) for _ in range(g.picks)]
    for _ in range(2000):
        line = sorted(rng.sample(range(g.low, g.high + 1), g.picks))
        if sums is None:
            return line
        odd = sum(n % 2 for n in line)
        if abs(odd - g.picks / 2) <= 1 and sums[0] <= sum(line) <= sums[1]:
            return line
    return line


with tab_pick:
    st.subheader(f"Quick pick - {game.label}")
    c1, c2 = st.columns([1, 2])
    n_lines = c1.number_input("Lines", 1, 10, 3)
    balanced = False
    if not game.ordered:
        balanced = c2.checkbox(
            "Balanced lines (near-even odd/even split, typical sum)",
            help="Cosmetic only: every combination is equally likely to win.",
        )
    if st.button("Generate", type="primary"):
        sums = None
        if balanced:
            history_sums = picks(game_all, game).sum(axis=1)
            sums = (float(np.percentile(history_sums, 10)), float(np.percentile(history_sums, 90)))
        rng = secrets.SystemRandom()
        st.session_state["quick_pick"] = (game_key, [make_line(game, rng, sums) for _ in range(int(n_lines))])
    saved = st.session_state.get("quick_pick")
    if saved and saved[0] == game_key:
        for line in saved[1]:
            st.markdown(balls_html(line, game), unsafe_allow_html=True)

    st.subheader("Odds of winning the top prize")
    odds = pd.DataFrame(
        [
            {
                "Game": g.label,
                "Rule": f"Pick {g.picks} of {g.low}-{g.high}" + (" in order" if g.ordered else ""),
                "Combinations": f"{g.combos:,}",
                "Chance": f"1 in {g.combos:,}",
            }
            for g in GAMES.values()
        ]
    )
    st.dataframe(odds, hide_index=True, width="stretch")
    st.caption("Digit games also pay for any-order (rambolito) matches at smaller prizes.")

st.divider()
st.caption(
    "Results from PCSO draws via the public lottotayo/pcso-lotto-results dataset. "
    "Not affiliated with PCSO. For information and entertainment only."
)
