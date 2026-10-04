"""Experiment engine: budgeted screening of solar-absorber candidates.

The pool is JARVIS-DFT materials that have an SLME value. Cheap columns
(composition descriptors, OptB88vdW gap, ehull, formation energy) are free.
SLME and the MBJ gap are "expensive": revealing them costs one unit of the
calculation budget, standing in for an MBJ + optics DFT run.

A hit is a stable (ehull <= 0.05 eV/atom), non-toxic material whose SLME
meets the campaign's efficiency target.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "pool.csv"

STABLE_EHULL = 0.05
EXPENSIVE = ["slme", "mbj_bandgap"]
NON_FEATURES = {"jid", "formula", "spg_symbol", "elements", "toxic", *EXPENSIVE}


@lru_cache(maxsize=1)
def load_pool() -> pd.DataFrame:
    return pd.read_csv(POOL)


def candidates(prefilter: dict | None = None) -> pd.DataFrame:
    """Candidate pool: stable and non-toxic, then an optional cheap pre-filter.

    ``prefilter`` keys (all optional, all based on free information):
    - ``gap_min`` / ``gap_max``: OptB88vdW band-gap window in eV.
    - ``any_of_elements``: keep materials containing at least one of these.
    - ``exclude_elements``: drop materials containing any of these.
    """
    df = load_pool()
    cand = df[(df["ehull"] <= STABLE_EHULL) & ~df["toxic"]]
    return apply_prefilter(cand, prefilter)


PREFILTER_KEYS = {"gap_min", "gap_max", "any_of_elements", "exclude_elements"}


def apply_prefilter(cand: pd.DataFrame, prefilter: dict | None) -> pd.DataFrame:
    if not prefilter:
        return cand
    unknown = set(prefilter) - PREFILTER_KEYS
    if unknown:
        raise ValueError(f"unsupported pre-filter keys {sorted(unknown)}; supported: {sorted(PREFILTER_KEYS)}")
    gap = cand["optb88vdw_bandgap"]
    mask = pd.Series(True, index=cand.index)
    if prefilter.get("gap_min") is not None:
        mask &= gap >= float(prefilter["gap_min"])
    if prefilter.get("gap_max") is not None:
        mask &= gap <= float(prefilter["gap_max"])
    els = cand["elements"].map(lambda s: set(s.split()))
    if prefilter.get("any_of_elements"):
        want = set(prefilter["any_of_elements"])
        mask &= els.map(lambda e: bool(e & want))
    if prefilter.get("exclude_elements"):
        drop = set(prefilter["exclude_elements"])
        mask &= els.map(lambda e: not e & drop)
    return cand[mask]


def prior_data(leakage_guard: bool = False) -> pd.DataFrame:
    """Materials outside the candidate set whose SLME is already known.

    With ``leakage_guard``, drop any prior row whose formula also appears among
    the candidates (same composition in another crystal structure).
    """
    df = load_pool()
    cand = candidates()
    prior = df.drop(index=cand.index)  # base candidates; a pre-filter never adds prior rows
    if leakage_guard:
        prior = prior[~prior["formula"].isin(set(cand["formula"]))]
    return prior


@lru_cache(maxsize=1)
def feature_cols() -> list[str]:
    df = load_pool()
    return [c for c in df.columns if c not in NON_FEATURES and df[c].dtype != object]


def is_hit(rows: pd.DataFrame, min_slme: float) -> pd.Series:
    return (rows["slme"] >= min_slme) & (rows["ehull"] <= STABLE_EHULL) & ~rows["toxic"]


# ── selection strategies ────────────────────────────────────────────────
# Each takes (candidate frame, measured index list, batch size, rng, **params)
# and returns the index labels of the next batch to "calculate".


def pick_random(cand: pd.DataFrame, measured: list, k: int, rng: np.random.Generator, **_) -> list:
    pool = cand.index.difference(measured)
    return list(rng.choice(pool, size=min(k, len(pool)), replace=False))


def pick_heuristic(cand: pd.DataFrame, measured: list, k: int, rng: np.random.Generator, target_gap: float = 1.1, **_) -> list:
    """Textbook rule: rank by closeness of the cheap OptB88vdW gap to a target.

    OptB88vdW underestimates gaps, so the default target sits below the
    Shockley-Queisser optimum of ~1.34 eV.
    """
    pool = cand.drop(index=measured, errors="ignore")
    score = (pool["optb88vdw_bandgap"] - target_gap).abs() + rng.uniform(0, 1e-6, len(pool))
    return list(score.nsmallest(k).index)


def fit_surrogate(train: pd.DataFrame, seed: int) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    model.fit(train[feature_cols()].fillna(0).values, train["slme"].values)
    return model


def predict(model: RandomForestRegressor, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    per_tree = np.stack([t.predict(rows[feature_cols()].fillna(0).values) for t in model.estimators_])
    return per_tree.mean(0), per_tree.std(0)


def pick_surrogate(
    cand: pd.DataFrame,
    measured: list,
    k: int,
    rng: np.random.Generator,
    explore: float = 0.5,
    warm_start: bool = True,
    leakage_guard: bool = False,
    exclude_elements: tuple[str, ...] = (),
    **_,
) -> list:
    """Active learning: random forest on known SLME, pick by upper confidence bound.

    ``explore`` scales the uncertainty bonus (0 = pure exploitation).
    ``warm_start`` adds prior calculations on materials outside the candidate set.
    """
    train = cand.loc[measured]
    if warm_start:
        train = pd.concat([train, prior_data(leakage_guard)])
    pool = cand.drop(index=measured, errors="ignore")
    if exclude_elements:
        pool = pool[pool["elements"].map(lambda s: not set(s.split()) & set(exclude_elements))]
    if len(train) < 5:
        return pick_random(cand, measured, k, rng)
    mean, std = predict(fit_surrogate(train, int(rng.integers(1_000_000_000))), pool)
    return list(pool.index[np.argsort(-(mean + explore * std))[:k]])


STRATEGIES = {"random": pick_random, "heuristic": pick_heuristic, "surrogate": pick_surrogate}


# ── retrospective benchmark ─────────────────────────────────────────────


@dataclass
class Trace:
    strategy: str
    seed: int
    calcs_to_target: int | None  # None = target not reached within budget


def simulate(strategy: str, seed: int, min_slme: float, target_hits: int, batch: int, budget: int, prefilter: dict | None = None, **params) -> Trace:
    """Run one screening campaign against the hidden SLME values.

    Every strategy spends its first two batches on the same random picks.
    """
    cand = candidates(prefilter)
    hit = is_hit(cand, min_slme)
    rng = np.random.default_rng(seed)
    measured: list = []
    hits = 0
    while len(measured) < min(budget, len(cand)):
        picker = pick_random if len(measured) < batch * 2 else STRATEGIES[strategy]
        for idx in picker(cand, measured, batch, rng, **params):
            measured.append(idx)
            hits += int(hit.at[idx])
            if hits >= target_hits:
                return Trace(strategy, seed, len(measured))
    return Trace(strategy, seed, None)


def benchmark(
    strategies: list[str],
    seeds: list[int],
    min_slme: float = 30.0,
    target_hits: int = 10,
    batch: int = 10,
    budget: int = 600,
    prefilter: dict | None = None,
    **params,
) -> dict:
    """Matched-conditions comparison: same pool, seeds, batch size and budget.

    With a ``prefilter``, every listed strategy screens the filtered pool, and
    a random baseline on the full pool is always added so speedups stay
    comparable across filters. The filter's recall (hits it keeps) is reported:
    a filter that throws away discoveries is not free.
    """
    base = candidates()
    cand = candidates(prefilter)
    base_hit = is_hit(base, min_slme)
    hit = is_hit(cand, min_slme)
    out: dict = {
        "conditions": {"min_slme": min_slme, "target_hits": target_hits, "batch": batch, "budget": budget, "seeds": seeds, "prefilter": prefilter, "surrogate_params": params},
        "pool": {"candidates": int(len(cand)), "hits": int(hit.sum()), "base_rate": round(float(hit.mean()), 4) if len(cand) else 0.0},
    }
    if prefilter:
        out["prefilter_effect"] = {
            "full_pool": int(len(base)),
            "kept": int(len(cand)),
            "hits_kept": int(hit.sum()),
            "hits_total": int(base_hit.sum()),
            "hit_recall": round(float(hit.sum() / base_hit.sum()), 3),
            "non_hits_removed": round(float(1 - (~hit).sum() / (~base_hit).sum()), 3),
        }
        if hit.sum() < target_hits:
            out["error"] = f"pre-filtered pool has only {int(hit.sum())} hits, fewer than target_hits={target_hits}"
            return out
    results = {}
    if prefilter:
        traces = [simulate("random", seed, min_slme, target_hits, batch, budget) for seed in seeds]
        reached = [t.calcs_to_target for t in traces if t.calcs_to_target is not None]
        results["random_full_pool"] = {
            "calcs_to_target": [t.calcs_to_target for t in traces],
            "median": float(np.median(reached)) if reached else None,
            "mean": round(float(np.mean(reached)), 1) if reached else None,
            "reached": f"{len(reached)}/{len(traces)}",
        }
    for s in strategies:
        traces = [simulate(s, seed, min_slme, target_hits, batch, budget, prefilter, **(params if s == "surrogate" else {})) for seed in seeds]
        reached = [t.calcs_to_target for t in traces if t.calcs_to_target is not None]
        results[s] = {
            "calcs_to_target": [t.calcs_to_target for t in traces],
            "median": float(np.median(reached)) if reached else None,
            "mean": round(float(np.mean(reached)), 1) if reached else None,
            "reached": f"{len(reached)}/{len(traces)}",
        }
    out["results"] = results
    ref_name = "random_full_pool" if prefilter else "random"
    ref = results.get(ref_name, {}).get("median")
    if ref:
        out["speedup_vs_random"] = {s: round(ref / r["median"], 2) for s, r in results.items() if r["median"]}
        out["speedup_reference"] = f"{ref_name} (median calculations)"
    return out
