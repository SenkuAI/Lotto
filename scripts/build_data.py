"""Turn `Lotto results.xlsx` into a clean `data/draws.csv.gz`.

Excel silently converted some 2D/3D combinations into dates, e.g. the 3D draw
"5-2-8" became 2008-05-02 and the 2D draw "09-06" became 2026-09-06. This
script reverses that, validates every row, merges with the older seed file,
and writes one tidy table (same schema the app already uses).

Usage:  python scripts/build_data.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "Lotto results.xlsx"
SEED = ROOT / "data" / "seed_draws.csv.gz"
OUT = ROOT / "data" / "draws.csv.gz"

# workbook name -> (game key, draw time, numbers per draw, low, high)
GAME_MAP = {
    "Ultra Lotto 6/58": ("6-58", "21:00", 6, 1, 58),
    "Grand Lotto 6/55": ("6-55", "21:00", 6, 1, 55),
    "Superlotto 6/49": ("6-49", "21:00", 6, 1, 49),
    "Megalotto 6/45": ("6-45", "21:00", 6, 1, 45),
    "Lotto 6/42": ("6-42", "21:00", 6, 1, 42),
    "6D Lotto": ("6d", "21:00", 6, 0, 9),
    "4D Lotto": ("4d", "21:00", 4, 0, 9),
    "3D Lotto 2PM": ("swertres", "14:00", 3, 0, 9),
    "3D Lotto 5PM": ("swertres", "17:00", 3, 0, 9),
    "3D Lotto 9PM": ("swertres", "21:00", 3, 0, 9),
    "2D Lotto 2PM": ("ez2", "14:00", 2, 1, 31),
    "2D Lotto 5PM": ("ez2", "17:00", 2, 1, 31),
    "2D Lotto 9PM": ("ez2", "21:00", 2, 1, 31),
}


def parse_combo(value, key: str, picks: int, conv_year: int):
    """Return a list of ints, or None if the cell can't be trusted."""
    if isinstance(value, (pd.Timestamp,)) or hasattr(value, "year"):
        if key == "swertres":  # "M-D-Y" with Y a single digit -> year 2000+Y
            nums = [value.month, value.day, value.year - 2000]
        elif key == "ez2":  # "M-D" -> current year at conversion time
            # "M-D" became that date in the conversion year; if the day was invalid for the
            # month (e.g. "2-29"), Excel read it as month-year instead ("Feb 2029").
            nums = [value.month, value.day] if value.year == conv_year else [value.month, value.year % 100]
        else:
            return None
    else:
        nums = [int(t) for t in re.findall(r"\d+", str(value))]
    return nums if len(nums) == picks else None


def load_xlsx() -> pd.DataFrame:
    raw = pd.read_excel(XLSX)
    raw.columns = ["name", "combo", "date", "jackpot_php", "winners"]
    is_2d = raw["name"].str.startswith("2D") & raw["combo"].map(lambda v: hasattr(v, "year"))
    conv_year = int(raw.loc[is_2d, "combo"].map(lambda v: v.year).mode().iloc[0])
    rows, dropped, bad = [], 0, []
    for r in raw.itertuples(index=False):
        if r.name not in GAME_MAP:
            continue
        key, time, picks, low, high = GAME_MAP[r.name]
        nums = parse_combo(r.combo, key, picks, conv_year)
        if nums is None or any(n < low or n > high for n in nums):
            dropped += 1
            bad.append((r.name, str(r.date)[:10], r.combo))
            continue
        rec = {"game": key, "draw_date": pd.Timestamp(r.date).strftime("%Y-%m-%d"), "draw_time": time}
        rec.update({f"n{i + 1}": (nums[i] if i < picks else None) for i in range(6)})
        rec["jackpot_php"] = r.jackpot_php
        rec["winners"] = r.winners
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"xlsx: {len(df):,} usable rows, {dropped} dropped as unreadable (conversion year {conv_year})")
    for item in bad[:8]:
        print("   dropped:", item)
    return df


def agreement_with_seed(xl: pd.DataFrame, seed: pd.DataFrame) -> None:
    cols = [f"n{i}" for i in range(1, 7)]
    m = xl.merge(seed, on=["game", "draw_date", "draw_time"], suffixes=("_x", "_s"))
    same = pd.Series(True, index=m.index)
    for c in cols:
        a, b = m[f"{c}_x"], m[f"{c}_s"]
        same &= (a == b) | (a.isna() & b.isna())
    print(f"overlap with seed file: {len(m):,} draws, numbers identical in {same.mean():.2%}")
    bad = m[~same]
    if len(bad):
        print(bad[["game", "draw_date", "draw_time"] + [f"{c}_x" for c in cols[:3]] + [f"{c}_s" for c in cols[:3]]].head(10).to_string())


def main() -> int:
    xl = load_xlsx()
    seed = pd.read_csv(SEED)
    seed["draw_date"] = pd.to_datetime(seed["draw_date"]).dt.strftime("%Y-%m-%d")
    agreement_with_seed(xl, seed)

    cols = ["game", "draw_date", "draw_time"] + [f"n{i}" for i in range(1, 7)] + ["jackpot_php", "winners"]
    # seed rows are listed last so they win ties (they came straight from the source, not via Excel)
    merged = pd.concat([xl[cols], seed[cols]], ignore_index=True)
    merged = merged.drop_duplicates(["game", "draw_date", "draw_time"], keep="last")
    merged = merged.sort_values(["draw_date", "game", "draw_time"], ascending=[False, True, True]).reset_index(drop=True)
    for c in [f"n{i}" for i in range(1, 7)]:
        merged[c] = merged[c].astype("Int64")
    merged["winners"] = merged["winners"].fillna(0).astype(int)
    merged.to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(merged):,} draws, {merged['draw_date'].min()} to {merged['draw_date'].max()}")
    print(merged.groupby("game").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
