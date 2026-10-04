"""Build the lab's candidate pool and cheap descriptors from JARVIS-DFT.

Run once with the lab venv (needs jarvis-tools); scripts/setup.sh does this:
    .venv/bin/python scripts/build_features.py

Downloads JARVIS-DFT dft_3d (~48 MB, figshare) into data/ and writes
data/pool.csv: one row per material that has an SLME value.
- Cheap columns (free to the agents): composition descriptors, the
  OptB88vdW band gap, ehull, formation energy, density, atom count.
- Expensive columns (hidden behind the run_calculations budget):
  slme, mbj_bandgap.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from jarvis.core.specie import Specie

ROOT = Path(__file__).resolve().parents[1]
PROPS = ["Z", "X", "atomic_mass", "atomic_rad", "row", "first_ion_en", "elec_aff", "mol_vol", "polzbl", "nfunfill"]
TOXIC = {"Pb", "Cd", "Hg", "Tl", "As"}


def parse_formula(formula: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula):
        counts[el] = counts.get(el, 0.0) + (float(n) if n else 1.0)
    return counts


_cache: dict[str, dict[str, float]] = {}


def element_props(el: str) -> dict[str, float]:
    if el not in _cache:
        s = Specie(el)
        vals = {}
        for p in PROPS:
            try:
                vals[p] = float(s.element_property(p))
            except Exception:
                vals[p] = np.nan
        _cache[el] = vals
    return _cache[el]


def descriptors(formula: str) -> dict[str, float]:
    comp = parse_formula(formula)
    total = sum(comp.values())
    fracs = {el: n / total for el, n in comp.items()}
    out: dict[str, float] = {"n_elements": len(comp)}
    for p in PROPS:
        vals = np.array([element_props(el)[p] for el in fracs])
        w = np.array(list(fracs.values()))
        out[f"{p}_mean"] = float(np.nansum(vals * w))
        out[f"{p}_min"] = float(np.nanmin(vals))
        out[f"{p}_max"] = float(np.nanmax(vals))
        out[f"{p}_range"] = out[f"{p}_max"] - out[f"{p}_min"]
    out["halide_frac"] = sum(f for el, f in fracs.items() if el in {"F", "Cl", "Br", "I"})
    out["chalcogen_frac"] = sum(f for el, f in fracs.items() if el in {"O", "S", "Se", "Te"})
    return out


COLUMNS = ["jid", "formula", "spg_symbol", "nat", "density", "optb88vdw_bandgap", "mbj_bandgap", "slme", "ehull", "formation_energy_peratom"]


def load_jarvis() -> pd.DataFrame:
    from jarvis.db.figshare import data

    (ROOT / "data").mkdir(exist_ok=True)
    df = pd.DataFrame(data("dft_3d", store_dir=str(ROOT / "data")))[COLUMNS]
    for c in COLUMNS[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # JARVIS uses "na" for missing
    return df


def main() -> None:
    df = load_jarvis()
    pool = df[df["slme"].notna()].reset_index(drop=True)
    feats = pd.DataFrame([descriptors(f) for f in pool["formula"]])
    pool["toxic"] = pool["formula"].map(lambda f: bool(TOXIC & set(parse_formula(f))))
    pool["elements"] = pool["formula"].map(lambda f: " ".join(sorted(parse_formula(f))))
    out = pd.concat([pool, feats], axis=1)
    out.to_csv(ROOT / "data" / "pool.csv", index=False)
    print(f"pool: {len(out)} materials, {feats.shape[1]} descriptors, {int(out['toxic'].sum())} toxic")


if __name__ == "__main__":
    main()
