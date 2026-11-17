"""Lotto Lab - machine-learning picks for every PCSO game.

Run locally:   streamlit run app.py

For every game and draw time the page shows
  1. the latest official result,
  2. the 5 combinations we made BEFORE that draw, scored against it,
  3. 5 fresh combinations for the NEXT draw,
  4. an honest "model check": how the model did on past draws vs plain chance.

The models (gradient boosting + logistic regression on ~25 history features)
live in lotto_model.py; data loading lives in lotto_core.py. Picks are a pure
function of the results, so they only change when a new result arrives.

Lottery draws are random. Not affiliated with PCSO.
"""

from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st

from lotto_core import GAMES, Game, load_draws, slots_for
from lotto_model import GameReport, SlotPicks, analyze_game

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Lotto Lab", page_icon="🎱", layout="wide", initial_sidebar_state="collapsed")

INK, MUTED, LINE = "#3d4150", "#767a8a", "#e8e3d9"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html {{font-size: 18px;}}
    html, body, [class*="css"], .stMarkdown, button {{font-family: 'Inter', system-ui, sans-serif !important;}}
    .block-container {{max-width: 1080px; padding-top: 2.2rem; padding-bottom: 3rem;}}
    #MainMenu, footer, header[data-testid="stHeader"] {{visibility: hidden; height: 0;}}
    .hero h1 {{margin:0; font-size:2.5rem; font-weight:800; letter-spacing:-.03em; color:{INK};}}
    .hero p {{margin:.3rem 0 1.1rem; color:{MUTED}; font-size:1.05rem;}}
    .stats {{display:grid; grid-template-columns:repeat(3,1fr); gap:.9rem; margin:.4rem 0 1.4rem;}}
    .stat {{background:#fffefb; border:1px solid {LINE}; border-radius:18px; padding:.9rem 1.1rem;
            box-shadow:0 1px 2px rgba(60,50,30,.04);}}
    .stat .k {{color:{MUTED}; font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.07em;}}
    .stat .v {{color:{INK}; font-size:1.45rem; font-weight:700; margin-top:.1rem;}}
    .card {{background:#fffefb; border:1px solid {LINE}; border-radius:24px; padding:1.5rem 1.7rem; margin-bottom:1.3rem;
            box-shadow:0 1px 2px rgba(60,50,30,.04), 0 10px 30px rgba(60,50,30,.06);}}
    .card .top {{display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap;}}
    .card .name {{font-size:1.55rem; font-weight:800; color:{INK}; letter-spacing:-.02em;}}
    .card .sub {{color:{MUTED}; font-size:.98rem; margin-top:.1rem;}}
    .pill {{display:inline-block; padding:.28rem .8rem; border-radius:999px; background:#eef0fb; color:#5560b8;
            font-size:.85rem; font-weight:600;}}
    .label {{color:{MUTED}; font-size:.8rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin:1.2rem 0 .35rem;}}
    .cols {{display:grid; grid-template-columns:1fr 1.25fr; gap:1.8rem;}}
    @media (max-width: 760px) {{.cols {{grid-template-columns:1fr;}} .stats {{grid-template-columns:1fr;}}}}
    .balls {{display:flex; gap:.5rem; flex-wrap:wrap;}}
    .ball {{display:inline-flex; align-items:center; justify-content:center; width:3.1rem; height:3.1rem; border-radius:50%;
            font-weight:700; font-size:1.45rem; color:{INK}; border:1px solid rgba(0,0,0,.07);}}
    .ball.result {{background:radial-gradient(circle at 30% 25%, #f7f8ff, #c5cbf0);}}
    .ball.pick {{background:radial-gradient(circle at 30% 25%, #fff8e9, #f2cd9f);}}
    .ball.small {{width:2.35rem; height:2.35rem; font-size:1.12rem;}}
    .ball.hit {{background:radial-gradient(circle at 30% 25%, #f2fbf6, #a6d4c0); border-color:#8dbba7;}}
    .ball.miss {{background:#f3f0ea; color:#a1a4b1;}}
    .row {{display:flex; align-items:center; gap:.8rem; margin:.42rem 0;}}
    .idx {{flex:0 0 1.6rem; height:1.6rem; border-radius:8px; background:#f1eee7; color:{MUTED}; font-size:.82rem;
           font-weight:700; display:inline-flex; align-items:center; justify-content:center;}}
    .score {{margin-left:.2rem; color:{MUTED}; font-weight:600; font-size:.98rem;}}
    .note {{color:{MUTED}; font-size:.95rem; margin-top:.45rem;}}
    .check {{margin-top:1.1rem; padding:.75rem 1rem; border-radius:14px; background:#f7f5ef; color:{MUTED}; font-size:.92rem;}}
    .check b {{color:{INK};}}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Training (cached; picks only change when new results arrive)
# --------------------------------------------------------------------------- #
@st.cache_resource(ttl=1800, show_spinner=False)
def build_reports() -> tuple[pd.DataFrame, str, list[GameReport], dict[str, str]]:
    df, status = load_draws()
    reports: list[GameReport] = []
    errors: dict[str, str] = {}
    for g in GAMES:
        try:
            slots = slots_for(df, g)
            if slots:
                reports.append(analyze_game(g, slots))
        except Exception as exc:  # one game failing must not take the page down
            errors[g.label] = f"{type(exc).__name__}: {exc}"
    return df, status, reports, errors


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def fmt(n: int, g: Game) -> str:
    return f"{n:02d}" if g.wide else str(n)


def balls(line, g: Game, kind: str = "", hits: list[bool] | None = None, small: bool = False) -> str:
    out = []
    for i, n in enumerate(line):
        cls = kind if hits is None else ("hit" if hits[i] else "miss")
        out.append(f"<span class='ball {cls}{' small' if small else ''}'>{fmt(int(n), g)}</span>")
    return "<div class='balls'>" + "".join(out) + "</div>"


def hit_flags(combo, latest, g: Game) -> list[bool]:
    if g.ordered:
        return [int(a) == int(b) for a, b in zip(combo, latest)]
    drawn = {int(x) for x in latest}
    return [int(a) in drawn for a in combo]


def scored_rows(combos, latest, g: Game) -> str:
    rows = []
    for i, c in enumerate(combos, 1):
        flags = hit_flags(c, latest, g)
        rows.append(
            f"<div class='row'><span class='idx'>{i}</span>{balls(c, g, hits=flags, small=True)}"
            f"<span class='score'>{sum(flags)}/{g.picks}</span></div>"
        )
    return "".join(rows)


def new_rows(combos, g: Game) -> str:
    return "".join(f"<div class='row'><span class='idx'>{i}</span>{balls(c, g, 'pick')}</div>" for i, c in enumerate(combos, 1))


def peso(v: float) -> str:
    return f"₱{v:,.0f}" if v == v else "—"


def slot_card(rep: GameReport, sp: SlotPicks) -> str:
    g = rep.game
    when = pd.Timestamp(sp.latest_date).strftime("%a, %b %d, %Y")
    prize = f"{'Top prize' if g.ordered else 'Jackpot'} {peso(sp.latest_prize)}"
    winners = f"{sp.latest_winners:,} winner{'' if sp.latest_winners == 1 else 's'}"
    best = max(sp.previous_matches)
    check = (
        f"<div class='check'><b>Model check</b> · on the {rep.holdout_draws} most recent draws it averaged "
        f"<b>{rep.model_matches:.2f}</b> matches per combination; random picks average <b>{rep.chance_matches:.2f}</b>. "
        f"{rep.verdict}.</div>"
    )
    return (
        "<div class='card'>"
        f"<div class='top'><div><div class='name'>{sp.name}</div>"
        f"<div class='sub'>Latest draw · {when}</div></div><span class='pill'>Best last pick {best}/{g.picks}</span></div>"
        "<div class='cols'><div>"
        "<div class='label'>Latest result</div>"
        f"{balls(sp.latest, g, 'result')}"
        f"<div class='note'>{prize} · {winners}</div>"
        "<div class='label'>Our last 5 picks, scored</div>"
        f"{scored_rows(sp.previous, sp.latest, g)}"
        "</div><div>"
        "<div class='label'>New picks for the next draw</div>"
        f"{new_rows(sp.upcoming, g)}"
        "</div></div>"
        f"{check}</div>"
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
with st.spinner("Training models on the full draw history (first load takes a few seconds)..."):
    df_all, data_status, reports, errors = build_reports()

if not reports:
    st.error("The models could not be built. " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
    st.stop()

st.markdown(
    "<div class='hero'><h1>🎱 Lotto Lab</h1>"
    "<p>Machine-learning picks for every PCSO game. Last picks are scored against the newest results, then fresh ones are made.</p></div>",
    unsafe_allow_html=True,
)

best_pick = max(
    ((max(sp.previous_matches), sp.name, rep.game.picks) for rep in reports for sp in rep.slots),
    key=lambda t: t[0] / t[2],
)
st.markdown(
    "<div class='stats'>"
    f"<div class='stat'><div class='k'>Draws learned from</div><div class='v'>{len(df_all):,}</div></div>"
    f"<div class='stat'><div class='k'>Newest result</div><div class='v'>{df_all['draw_date'].max():%b %d, %Y}</div></div>"
    f"<div class='stat'><div class='k'>Best last pick</div><div class='v'>{best_pick[0]}/{best_pick[2]} · {best_pick[1].split(' · ')[0]}</div></div>"
    "</div>",
    unsafe_allow_html=True,
)

labels = {r.game.key: r.game.label for r in reports}
choice = st.pills("Game", ["All games"] + list(labels.values()), default="All games", label_visibility="collapsed")
shown = [r for r in reports if choice in (None, "All games", r.game.label)]

for rep in shown:
    for sp in rep.slots:
        st.markdown(slot_card(rep, sp), unsafe_allow_html=True)

for name, err in errors.items():
    st.warning(f"{name} is temporarily unavailable ({err}).")

with st.expander("How the picks are made"):
    st.markdown(
        "1. **Learn from history.** All past draws for a game (back to 2016) become training data.\n"
        "2. **Score every number.** About 25 signals per number - long-run and recent rates, how overdue it is, "
        "whether it appeared in the last few draws, which numbers it tends to appear with - feed a gradient-boosting "
        "model and a logistic-regression model. Their average is each number's probability.\n"
        "3. **Build 5 different combinations.** Highest joint probability, kept balanced (odd/even, high/low, typical total) "
        "and different from each other.\n"
        "4. **Check against reality.** The model is trained only on the past and tested on the most recent draws, "
        "and the result is shown honestly on each card.\n\n"
        "Lottery draws are random, so expect the model to perform like random picks. Use it for fun and play responsibly."
    )

st.divider()
left, right = st.columns([5, 1])
left.markdown(
    f"<span class='note'>Data: {data_status}, built from PCSO results (your spreadsheet plus the public "
    "lottotayo/pcso-lotto-results dataset). Not affiliated with PCSO. For entertainment only.</span>",
    unsafe_allow_html=True,
)
if right.button("Refresh"):
    build_reports.clear()
    st.rerun()
