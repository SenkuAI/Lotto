"""Machine-learning engine for Lotto Lab.

For every game we model "will number v appear in this draw?" as a binary
classification problem and learn it from the full draw history.

Features (all computed ONLY from draws before the one being predicted):
  * Bayesian-shrunk long-run rate and its z-score
  * rolling rates over 10 / 25 / 50 / 100 / 250 draws
  * exponentially weighted rates (half-lives 8 / 30 / 120 draws)
  * overdue gap (draws since last seen), relative to its expected and average gap
  * lags 1-3 (did it appear in the previous draws?)
  * pair affinity (how often v co-occurs with the numbers of the last draw)
  * value position, parity, and (digit games) the digit position

Models: gradient boosting + L2 logistic regression, averaged. Probabilities
are re-normalised so a draw's probabilities add up to the numbers drawn.

A pick is then the set of combinations with the highest joint probability that
also pass a "typical draw" filter (odd/even, high/low, sum, no long runs), and
the 5 combinations are chosen to differ from each other.

In the jackpot games (lines of 6 numbers) a "popularity" penalty also steers away
from lines many players are likely to share: mostly birthday numbers (<= 31),
runs of consecutive numbers, and arithmetic sequences. This does NOT change the
chance of winning - every line is equally likely - but if a shared line wins, the
jackpot is split, so a rarer line has a higher expected payout.

Honesty: a walk-forward hold-out test (train on the past, score the most recent
draws) is run for every game, so the app can show whether the model beats plain
chance. Lottery draws are random, so expect it not to.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lotto_core import Game, Slot

WARMUP = 80  # draws of history needed before a row is used for training
MAX_DRAWS = 600  # most recent draws per slot used for training
HOLDOUT = 200  # most recent draws kept aside for the honesty test
N_COMBOS = 5
POOL = 16  # lotto games: combinations are built from the 16 most likely numbers
POPULARITY_WEIGHT = 1.0  # strength of the "avoid lines other players share" penalty (0 = off)
BIRTHDAY_MAX = 31  # numbers up to this are over-picked because they are birthdays / days of the month


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def stream_features(X: np.ndarray, k_eff: int, k_draw: int, pos: int, n_pos: int, pair: bool) -> np.ndarray:
    """X: (D, N) 0/1 matrix, oldest draw first. Returns (D+1, N, F).

    Row t describes the situation just BEFORE draw t (draws 0..t-1 are known);
    row D is the situation for the upcoming, unknown draw.
    """
    D, N = X.shape
    p = k_eff / N
    t_idx = np.arange(D + 1)
    tcol = t_idx[:, None].astype(float)
    cs = np.vstack([np.zeros((1, N)), np.cumsum(X, axis=0)])  # cs[t] = counts in draws < t
    f: list[np.ndarray] = []

    f.append((cs + 30 * p) / (np.maximum(tcol, 1) + 30) / p)  # shrunk long-run rate (relative)
    f.append((cs - tcol * p) / np.sqrt(np.maximum(tcol * p * (1 - p), 1e-9)))  # z-score
    for w in (10, 25, 50, 100, 250):
        lo = np.maximum(t_idx - w, 0)
        f.append((cs - cs[lo]) / np.maximum(t_idx - lo, 1)[:, None] / p)
    for half_life in (8, 30, 120):
        a = 1 - 0.5 ** (1 / half_life)
        E = np.empty((D + 1, N))
        E[0] = p
        for t in range(1, D + 1):
            E[t] = E[t - 1] * (1 - a) + a * X[t - 1]
        f.append(E / p)

    gap = np.empty((D + 1, N))
    last = np.full(N, -1.0)
    for t in range(D + 1):
        gap[t] = t - last - 1
        if t < D:
            last = np.where(X[t] > 0, t, last)
    f.append(gap * p)  # gap relative to the expected gap
    f.append(np.log1p(gap))
    f.append(gap / (np.maximum(tcol, 1) / (cs + 1)))  # gap relative to this number's own average gap

    for lag in (1, 2, 3):
        L = np.zeros((D + 1, N))
        L[lag:] = X[: D + 1 - lag]
        f.append(L)

    aff = np.zeros((D + 1, N))
    if pair:
        C = np.zeros((N, N))
        norm = k_draw * (k_draw - 1) / (N - 1)
        for t in range(1, D + 1):
            x = X[t - 1]
            aff[t] = (C @ x) / (cs[t - 1] + 1) / norm
            C += np.outer(x, x)
            np.fill_diagonal(C, 0)
    f.append(aff)

    v = np.arange(N)
    f.append(np.broadcast_to(v / N, (D + 1, N)))
    f.append(np.broadcast_to((v % 2).astype(float), (D + 1, N)))
    f.append(np.full((D + 1, N), pos / max(n_pos - 1, 1)))
    return np.stack(f, axis=2).astype(np.float32)


@dataclass
class Stream:
    X: np.ndarray  # (D, N) indicators
    F: np.ndarray  # (D+1, N, nf) features


@dataclass
class SlotData:
    slot: Slot
    streams: list[Stream]

    @property
    def D(self) -> int:
        return len(self.slot.arr)


def build_slot(slot: Slot) -> SlotData:
    g, arr = slot.game, slot.arr
    D, N, k = len(arr), slot.game.size, g.picks
    streams: list[Stream] = []
    if g.ordered:
        for i in range(k):
            X = (arr[:, i : i + 1] - g.low == np.arange(N)[None, :]).astype(float)
            streams.append(Stream(X, stream_features(X, 1, 1, i, k, pair=False)))
    else:
        X = np.zeros((D, N))
        X[np.arange(D)[:, None], arr - g.low] = 1
        streams.append(Stream(X, stream_features(X, k, k, 0, 1, pair=True)))
    return SlotData(slot, streams)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def training_matrix(sds: list[SlotData], t_hi: dict[int, int]):
    Xs, ys = [], []
    for j, sd in enumerate(sds):
        hi = t_hi[j]
        lo = max(WARMUP, hi - MAX_DRAWS)
        for s in sd.streams:
            nf = s.F.shape[2]
            Xs.append(s.F[lo:hi].reshape(-1, nf))
            ys.append(s.X[lo:hi].reshape(-1))
    return np.concatenate(Xs), np.concatenate(ys).astype(int)


class Ensemble:
    def __init__(self) -> None:
        self.gb = HistGradientBoostingClassifier(
            max_iter=70, learning_rate=0.05, max_depth=3, min_samples_leaf=300, l2_regularization=3.0, random_state=0
        )
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=400))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ensemble":
        self.gb.fit(X, y)
        self.lr.fit(X, y)
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        return 0.5 * (self.gb.predict_proba(X)[:, 1] + self.lr.predict_proba(X)[:, 1])


def predict_rows(model: Ensemble, sd: SlotData, t0: int, t1: int) -> list[np.ndarray]:
    """Per stream: probabilities of shape (t1-t0, N), normalised per draw."""
    g = sd.slot.game
    out = []
    for s in sd.streams:
        T, N, nf = t1 - t0, s.F.shape[1], s.F.shape[2]
        p = model.proba(s.F[t0:t1].reshape(-1, nf)).reshape(T, N)
        p = np.clip(p, 1e-6, None)
        total = 1.0 if g.ordered else float(g.picks)
        out.append(p * (total / p.sum(axis=1, keepdims=True)))
    return out


# --------------------------------------------------------------------------- #
# Turning probabilities into combinations
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _combo_idx(m: int, k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(m), k)), dtype=np.int16)


@lru_cache(maxsize=None)
def _product_idx(m: int, k: int) -> np.ndarray:
    return np.array(list(itertools.product(range(m), repeat=k)), dtype=np.int16)


def popularity(nums: np.ndarray) -> np.ndarray:
    """How likely other players are to hold a similar line (higher = more shared). nums: (C, k)."""
    ordered = np.sort(nums, axis=1)
    steps = np.diff(ordered, axis=1)
    birthday = (nums <= BIRTHDAY_MAX).sum(axis=1)
    runs = (steps == 1).sum(axis=1)
    sequence = (steps == steps[:, :1]).all(axis=1)  # e.g. 5-10-15-20-25-30
    return 0.35 * birthday + 0.5 * runs + 2.0 * sequence


def combos_unordered(p: np.ndarray, g: Game, sums_ref: tuple[float, float], n: int = N_COMBOS) -> list[list[int]]:
    k, N = g.picks, g.size
    pool = np.argsort(-p)[:POOL]
    vals = pool[_combo_idx(len(pool), k)].astype(int)  # (C, k) zero-based values
    nums = vals + g.low
    score = np.log(p)[vals].sum(axis=1) - POPULARITY_WEIGHT * popularity(nums)
    C = len(nums)
    odd = (nums % 2).sum(axis=1)
    high = (nums > (g.low + g.high) / 2).sum(axis=1)
    sums = nums.sum(axis=1)
    consec = (np.diff(np.sort(nums, axis=1), axis=1) == 1).sum(axis=1)
    lo, hi = sums_ref
    masks = [
        (np.abs(odd - k / 2) <= 1) & (np.abs(high - k / 2) <= 1) & (sums >= lo) & (sums <= hi) & (consec <= 1),
        (np.abs(odd - k / 2) <= 1) & (sums >= lo) & (sums <= hi),
        np.ones(C, dtype=bool),
    ]
    member = np.zeros((C, N), dtype=np.float32)
    member[np.arange(C)[:, None], vals] = 1
    taken = np.zeros(C, dtype=bool)
    overlap = np.zeros(C)
    chosen: list[int] = []
    for _ in range(n):
        picked = None
        for max_overlap in (k - 2, k - 1, k):
            for mask in masks:
                ok = mask & ~taken & (overlap <= max_overlap)
                if ok.any():
                    picked = int(np.argmax(np.where(ok, score, -np.inf)))
                    break
            if picked is not None:
                break
        if picked is None:
            break
        chosen.append(picked)
        taken[picked] = True
        overlap = np.maximum(overlap, member @ member[picked])
    return [sorted(nums[i].tolist()) for i in chosen]


def combos_ordered(P: np.ndarray, g: Game, n: int = N_COMBOS) -> list[list[int]]:
    k, N = P.shape
    m = min(N, 6 if k <= 2 else 4)
    top = np.argsort(-P, axis=1)[:, :m]  # (k, m)
    grid = _product_idx(m, k).astype(int)
    cols = np.arange(k)[None, :]
    vals = top[cols, grid]  # (C, k)
    score = np.log(P[cols, vals]).sum(axis=1)
    cands = vals + g.low
    min_dist = 2 if k >= 3 else 1
    taken = np.zeros(len(cands), dtype=bool)
    dist = np.full(len(cands), k)
    chosen: list[int] = []
    for _ in range(n):
        picked = None
        for need in (min_dist, 1):
            ok = ~taken & (dist >= need)
            if ok.any():
                picked = int(np.argmax(np.where(ok, score, -np.inf)))
                break
        if picked is None:
            break
        chosen.append(picked)
        taken[picked] = True
        dist = np.minimum(dist, (cands != cands[picked]).sum(axis=1))
    return [cands[i].tolist() for i in chosen]


def combos_for(probs: list[np.ndarray], row: int, g: Game, sums_ref: tuple[float, float]) -> list[list[int]]:
    if g.ordered:
        return combos_ordered(np.stack([p[row] for p in probs]), g)
    return combos_unordered(probs[0][row], g, sums_ref)


def count_matches(combo: list[int], actual: np.ndarray, g: Game) -> int:
    if g.ordered:
        return int(sum(int(a) == int(b) for a, b in zip(combo, actual)))
    return len(set(combo) & set(int(x) for x in actual))


# --------------------------------------------------------------------------- #
# Putting it together
# --------------------------------------------------------------------------- #
@dataclass
class SlotPicks:
    name: str
    time: str
    draws: int
    latest: list[int]
    latest_date: str
    latest_prize: float
    latest_winners: int
    previous: list[list[int]]  # 5 combinations made BEFORE the latest draw
    previous_matches: list[int]
    upcoming: list[list[int]]  # 5 combinations for the NEXT draw


@dataclass
class GameReport:
    game: Game
    slots: list[SlotPicks] = field(default_factory=list)
    train_rows: int = 0
    holdout_draws: int = 0
    model_matches: float = 0.0  # avg matches per combination on the hold-out
    chance_matches: float = 0.0
    z_score: float = 0.0
    logloss_gain_pct: float = 0.0  # >0 means better than a constant-rate baseline
    verdict: str = ""


def chance_stats(g: Game) -> tuple[float, float]:
    """Expected matches per combination for a random pick, and its variance."""
    k, N = g.picks, g.size
    if g.ordered:
        return k / N, k * (1 / N) * (1 - 1 / N)
    return k * k / N, k * (k / N) * (1 - k / N) * (N - k) / (N - 1)


def _sums_ref(arr: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(arr.sum(axis=1), [10, 90])
    return float(lo), float(hi)


def analyze_game(g: Game, slots: list[Slot]) -> GameReport:
    sds = [build_slot(s) for s in slots]
    report = GameReport(g)

    # ---- production model: learns from every draw except the very latest -------------
    X, y = training_matrix(sds, {j: sd.D - 1 for j, sd in enumerate(sds)})
    prod = Ensemble().fit(X, y)
    report.train_rows = len(y)

    for sd in sds:
        D, s = sd.D, sd.slot
        probs_prev = predict_rows(prod, sd, D - 1, D)  # situation before the latest draw
        probs_next = predict_rows(prod, sd, D, D + 1)  # situation before the next draw
        ref_prev, ref_next = _sums_ref(s.arr[: D - 1]), _sums_ref(s.arr)
        prev = combos_for(probs_prev, 0, g, ref_prev)
        latest = s.arr[D - 1]
        row = s.frame.iloc[0]
        report.slots.append(
            SlotPicks(
                name=s.name,
                time=s.time,
                draws=D,
                latest=[int(x) for x in latest],
                latest_date=f"{row['draw_date']:%Y-%m-%d}",
                latest_prize=float(row["jackpot_php"]) if row["jackpot_php"] == row["jackpot_php"] else float("nan"),
                latest_winners=int(row["winners"]),
                previous=prev,
                previous_matches=[count_matches(c, latest, g) for c in prev],
                upcoming=combos_for(probs_next, 0, g, ref_next),
            )
        )

    # ---- honesty test: train on the past, score the most recent draws ------------------
    H = min(HOLDOUT, int(0.2 * min(sd.D for sd in sds)))
    report.holdout_draws = H
    Xh, yh = training_matrix(sds, {j: sd.D - H for j, sd in enumerate(sds)})
    hold = Ensemble().fit(Xh, yh)
    base_p = float(yh.mean())
    ll_model, ll_base, rows = 0.0, 0.0, 0
    matches: list[int] = []
    for sd in sds:
        D, s = sd.D, sd.slot
        raw = []
        for st in sd.streams:
            nf = st.F.shape[2]
            raw.append(hold.proba(st.F[D - H : D].reshape(-1, nf)))
        for st, pr in zip(sd.streams, raw):
            actual = st.X[D - H : D].reshape(-1)
            pr = np.clip(pr, 1e-6, 1 - 1e-6)
            ll_model += float(-(actual * np.log(pr) + (1 - actual) * np.log(1 - pr)).sum())
            ll_base += float(-(actual * math.log(base_p) + (1 - actual) * math.log(1 - base_p)).sum())
            rows += len(actual)
        probs = predict_rows(hold, sd, D - H, D)
        ref = _sums_ref(s.arr[: D - H])
        for i in range(H):
            for combo in combos_for(probs, i, g, ref):
                matches.append(count_matches(combo, s.arr[D - H + i], g))
    exp, var = chance_stats(g)
    mean = float(np.mean(matches))
    # draws within a slot are independent; combos of one draw are correlated, so use draws as the unit
    n_eff = H * len(sds)
    report.model_matches = mean
    report.chance_matches = exp
    report.z_score = (mean - exp) / math.sqrt(var / n_eff)
    report.logloss_gain_pct = 100 * (ll_base - ll_model) / ll_base
    if abs(report.z_score) < 2:
        report.verdict = "No measurable edge over random picks"
    elif report.z_score > 0:
        report.verdict = "Slightly above chance on the test window (likely luck)"
    else:
        report.verdict = "Slightly below chance on the test window"
    return report
