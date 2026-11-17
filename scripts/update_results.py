"""Bring Lotto Lab up to date with the newest PCSO results.

    python scripts/update_results.py              # on your computer: also updates Lotto results.xlsx
    python scripts/update_results.py --no-xlsx    # what the scheduled GitHub job runs
    python scripts/update_results.py --dry-run    # just report what is new

Steps
  1. Fetch the newest results and keep only draws we don't have yet.
  2. Add them to data/draws.csv.gz (the file the app trains on) and, unless
     --no-xlsx is given, to the top of `Lotto results.xlsx` in its existing format.
  3. Score the picks we logged earlier against the draws that have now happened,
     and log fresh picks for the next draws (data/pick_log.csv). That log is the
     app's honest, out-of-sample track record.

When the data file changes and is pushed, Streamlit Cloud redeploys and the
app retrains on the new results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import build_data  # noqa: E402
from lotto_core import BY_KEY, GAMES, NUM_COLS, fetch_live, load_draws, normalise, picks_matrix, slots_for  # noqa: E402

warnings.filterwarnings("ignore")

XLSX = build_data.XLSX
CLEAN = build_data.OUT
LEDGER = ROOT / "data" / "pick_log.csv"
SHEET = "Lotto Results"
NAME_FOR = {(key, time): name for name, (key, time, *_rest) in build_data.GAME_MAP.items()}
LEDGER_COLS = ["game", "draw_time", "after_date", "combos", "scored_date", "matches", "avg_matches", "chance_avg"]


# --------------------------------------------------------------------------- #
# 1. find new draws
# --------------------------------------------------------------------------- #
def find_new(pages: int) -> pd.DataFrame:
    live = None
    for attempt in range(4):  # the free feed has the occasional 5xx; wait and retry
        try:
            live = normalise(fetch_live(pages))
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * 3**attempt)
    clean = pd.read_csv(CLEAN)
    have = set(zip(clean["game"], clean["draw_date"].astype(str).str[:10], clean["draw_time"].astype(str).str[:5]))
    day = live["draw_date"].dt.strftime("%Y-%m-%d")
    keep = [((g, d, t) not in have) and ((g, t) in NAME_FOR) for g, d, t in zip(live["game"], day, live["draw_time"])]
    return live[keep].copy()


# --------------------------------------------------------------------------- #
# 2. write them down
# --------------------------------------------------------------------------- #
def merge_clean(new: pd.DataFrame) -> None:
    cols = ["game", "draw_date", "draw_time"] + NUM_COLS + ["jackpot_php", "winners"]
    add = new[cols].copy()
    add["draw_date"] = add["draw_date"].dt.strftime("%Y-%m-%d")
    clean = pd.read_csv(CLEAN)
    merged = pd.concat([clean[cols], add], ignore_index=True)
    merged = merged.drop_duplicates(["game", "draw_date", "draw_time"], keep="last")
    merged = merged.sort_values(["draw_date", "game", "draw_time"], ascending=[False, True, True]).reset_index(drop=True)
    for c in NUM_COLS:
        merged[c] = merged[c].astype("Int64")
    merged["winners"] = merged["winners"].fillna(0).astype(int)
    merged.to_csv(CLEAN, index=False, compression="gzip")
    print(f"data/draws.csv.gz now has {len(merged):,} draws (newest {merged['draw_date'].max()})")


def combo_text(row: pd.Series) -> str:
    g = BY_KEY[row["game"]]
    nums = [int(row[f"n{i}"]) for i in range(1, g.picks + 1)]
    return "-".join(f"{n:02d}" if g.wide else str(n) for n in nums)


def append_to_xlsx(new: pd.DataFrame) -> None:
    """Insert new rows under the header, newest first, matching the existing cell formats."""
    import openpyxl

    wb = openpyxl.load_workbook(XLSX)
    ws = wb[SHEET] if SHEET in wb.sheetnames else wb.active
    rows = new.sort_values(["draw_date", "game", "draw_time"], ascending=[False, True, True])
    ws.insert_rows(2, amount=len(rows))
    for offset, (_, r) in enumerate(rows.iterrows()):
        i = 2 + offset
        ws.cell(i, 1, NAME_FOR[(r["game"], r["draw_time"])])
        ws.cell(i, 2, combo_text(r))
        date = ws.cell(i, 3, pd.Timestamp(r["draw_date"]).to_pydatetime())
        date.number_format = "mm-dd-yy"
        prize = ws.cell(i, 4, None if pd.isna(r["jackpot_php"]) else float(r["jackpot_php"]))
        prize.number_format = "#,##0.00"
        ws.cell(i, 5, int(r["winners"]))
    tmp = XLSX.with_suffix(".tmp.xlsx")
    wb.save(tmp)
    try:
        tmp.replace(XLSX)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise SystemExit("Could not save 'Lotto results.xlsx' - close it in Excel and run this again.")
    print(f"Lotto results.xlsx: added {len(rows)} rows at the top")


# --------------------------------------------------------------------------- #
# 3. pick log = honest track record
# --------------------------------------------------------------------------- #
def update_ledger() -> None:
    from lotto_model import analyze_game, chance_stats, count_matches

    df, _ = load_draws(live=False)
    if LEDGER.exists():
        ledger = pd.read_csv(LEDGER, dtype=str)
    else:
        ledger = pd.DataFrame(columns=LEDGER_COLS, dtype=str)

    # score earlier picks against the first draw that happened after they were made
    scored = 0
    for idx, row in ledger[ledger["scored_date"].isna()].iterrows():
        g = BY_KEY[row["game"]]
        later = df[(df["game"] == row["game"]) & (df["draw_time"] == row["draw_time"]) & (df["draw_date"] > pd.Timestamp(row["after_date"]))]
        if later.empty:
            continue
        nxt = later.sort_values("ts").head(1)
        actual = picks_matrix(nxt, g)[0]
        m = [count_matches(c, actual, g) for c in json.loads(row["combos"])]
        ledger.loc[idx, "scored_date"] = f"{nxt.iloc[0]['draw_date']:%Y-%m-%d}"
        ledger.loc[idx, "matches"] = json.dumps(m)
        ledger.loc[idx, "avg_matches"] = f"{np.mean(m):.4f}"
        ledger.loc[idx, "chance_avg"] = f"{chance_stats(g)[0]:.4f}"
        scored += 1

    # log picks for the next draw of every slot we haven't logged yet
    have = set(zip(ledger["game"], ledger["draw_time"], ledger["after_date"]))
    logged = 0
    for g in GAMES:
        slots = slots_for(df, g)
        latest = {s.time: f"{s.frame.iloc[0]['draw_date']:%Y-%m-%d}" for s in slots}
        if not slots or all((g.key, t, d) in have for t, d in latest.items()):
            continue
        rep = analyze_game(g, slots)
        for sp in rep.slots:
            if (g.key, sp.time, sp.latest_date) in have:
                continue
            ledger.loc[len(ledger)] = [g.key, sp.time, sp.latest_date, json.dumps(sp.upcoming), None, None, None, None]
            logged += 1

    ledger.to_csv(LEDGER, index=False)
    total = int(ledger["scored_date"].notna().sum())
    print(f"pick log: scored {scored} earlier picks, logged {logged} new ones ({total} scored in total)")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=8, help="pages of 100 live rows to look through (default 8)")
    ap.add_argument("--no-xlsx", action="store_true", help="don't touch Lotto results.xlsx")
    ap.add_argument("--dry-run", action="store_true", help="only report what is new")
    args = ap.parse_args()

    try:
        new = find_new(args.pages)
    except Exception as exc:
        print(f"Could not reach the live results feed: {type(exc).__name__}: {exc}")
        return 1

    if new.empty:
        print("No new results - already up to date.")
    else:
        print(f"{len(new)} new results:")
        print(new.groupby("game")["draw_date"].agg(["count", "min", "max"]).to_string())
        if args.dry_run:
            return 0
        merge_clean(new)
        if not args.no_xlsx and XLSX.exists():
            append_to_xlsx(new)

    if not args.dry_run:
        update_ledger()
    return 0


if __name__ == "__main__":
    sys.exit(main())
