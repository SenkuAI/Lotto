"""Data layer for Lotto Lab: game definitions, loading, cleaning, draw slots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "draws.csv.gz"
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
    times: tuple[str, ...]  # draw times in a day

    @property
    def size(self) -> int:
        return self.high - self.low + 1

    @property
    def wide(self) -> bool:
        return self.high > 9

    @property
    def odds(self) -> int:
        return self.size**self.picks if self.ordered else math.comb(self.size, self.picks)


NIGHT = ("21:00",)
GAMES: list[Game] = [
    Game("6-58", "Ultra Lotto 6/58", 6, 1, 58, False, NIGHT),
    Game("6-55", "Grand Lotto 6/55", 6, 1, 55, False, NIGHT),
    Game("6-49", "Super Lotto 6/49", 6, 1, 49, False, NIGHT),
    Game("6-45", "Mega Lotto 6/45", 6, 1, 45, False, NIGHT),
    Game("6-42", "Lotto 6/42", 6, 1, 42, False, NIGHT),
    Game("6d", "6D Lotto", 6, 0, 9, True, NIGHT),
    Game("4d", "4D Lotto", 4, 0, 9, True, NIGHT),
    Game("swertres", "Swertres 3D", 3, 0, 9, True, ("14:00", "17:00", "21:00")),
    Game("ez2", "EZ2 2D", 2, 1, 31, True, ("14:00", "17:00", "21:00")),
]
BY_KEY = {g.key: g for g in GAMES}


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Types, valid ranges, de-duplication; newest draw first."""
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
    """Newest rows from the public Hugging Face mirror of PCSO results."""
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


def load_draws(live: bool = True) -> tuple[pd.DataFrame, str]:
    """Bundled history (2016+) plus, when reachable, the newest live rows."""
    frames = [pd.read_csv(DATA_PATH)]
    status = "bundled data"
    if live:
        try:
            frames.append(fetch_live())
            status = "live data"
        except Exception as exc:  # offline, rate-limited, schema change...
            status = f"bundled data (live feed unavailable: {type(exc).__name__})"
    return normalise(pd.concat(frames, ignore_index=True)), status


@dataclass
class Slot:
    """One game at one draw time, e.g. Swertres 5 PM."""

    game: Game
    time: str
    frame: pd.DataFrame  # newest first
    arr: np.ndarray  # chronological (oldest first), shape (draws, picks)

    @property
    def name(self) -> str:
        return self.game.label + (f" · {clock12(self.time)}" if len(self.game.times) > 1 else "")


def picks_matrix(df: pd.DataFrame, g: Game) -> np.ndarray:
    return df[NUM_COLS[: g.picks]].to_numpy(dtype=int)


def slots_for(df: pd.DataFrame, g: Game) -> list[Slot]:
    hist = df[df["game"] == g.key]
    out = []
    for t in g.times:
        frame = hist[hist["draw_time"] == t]
        if len(frame) < 400:
            continue
        out.append(Slot(g, t, frame, picks_matrix(frame, g)[::-1].copy()))
    return out


def clock12(t: str) -> str:
    try:
        return pd.to_datetime(t, format="%H:%M").strftime("%I:%M %p").lstrip("0")
    except Exception:
        return t
