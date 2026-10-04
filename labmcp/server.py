"""MCP server exposing the lab's tools to Omnigent agents.

Databricks-hosted Omnigent rejects bundled Python tool files, so the lab's
tools run here instead, as a stdio MCP server on the host machine.

State lives on disk so every agent (each with its own MCP process) sees the
same lab:
- runs/campaigns/<name>.json  live screening campaigns (budget, results)
- runs/record.jsonl           append-only research record

Run standalone for a smoke test:
    .venv/bin/python -m labmcp.server
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from mcp.server.fastmcp import FastMCP

from labmcp import engine

RUNS = engine.ROOT / "runs"
CAMPAIGNS = RUNS / "campaigns"
RECORD = RUNS / "record.jsonl"
CAMPAIGNS.mkdir(parents=True, exist_ok=True)

JARVIS = "JARVIS-DFT dft_3d (Choudhary et al., doi:10.1016/j.commatsci.2025.114063)"
MAX_BATCH = 20

mcp = FastMCP("lab")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _append_record(entry: dict) -> dict:
    entry = {"id": uuid.uuid4().hex[:8], "time": _now(), **entry}
    with RECORD.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def _campaign_path(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in "-_")
    if not safe:
        raise ValueError("campaign name must contain letters or digits")
    return CAMPAIGNS / f"{safe}.json"


def _load_campaign(name: str) -> dict:
    path = _campaign_path(name)
    if not path.exists():
        existing = sorted(f.stem for f in CAMPAIGNS.glob("*.json"))
        raise ValueError(f"no campaign named {name!r}. Existing campaigns: {existing or 'none'}. Start one with start_campaign.")
    return json.loads(path.read_text())


def _save_campaign(c: dict) -> None:
    _campaign_path(c["name"]).write_text(json.dumps(c, indent=1))


def _prefilter(gap_min, gap_max, any_of_elements, exclude_elements) -> dict | None:
    pf = {
        "gap_min": gap_min,
        "gap_max": gap_max,
        "any_of_elements": list(any_of_elements) if any_of_elements else None,
        "exclude_elements": list(exclude_elements) if exclude_elements else None,
    }
    pf = {k: v for k, v in pf.items() if v is not None}
    return pf or None


# ── orientation ─────────────────────────────────────────────────────────


@mcp.tool()
def dataset_summary(min_slme: float = 30.0) -> dict:
    """Describe the candidate pool and how rare hits are at an efficiency target.

    Args:
        min_slme: Efficiency target: minimum SLME (spectroscopic limited maximum efficiency), in percent.
    """
    df = engine.load_pool()
    cand = engine.candidates()
    hits = engine.is_hit(cand, min_slme)
    return {
        "source": JARVIS,
        "materials_with_slme": int(len(df)),
        "candidates_stable_non_toxic": int(len(cand)),
        "hits_at_target": int(hits.sum()),
        "base_rate": round(float(hits.mean()), 4),
        "prior_calculations_available": int(len(engine.prior_data())),
        "definitions": {
            "hit": f"SLME >= {min_slme}%, ehull <= {engine.STABLE_EHULL} eV/atom, no Pb/Cd/Hg/Tl/As",
            "cheap_info": "composition descriptors, OptB88vdW band gap, ehull, formation energy",
            "expensive_info": "SLME and MBJ band gap: one budget unit per material",
        },
    }


@mcp.tool()
def literature_search(query: str, max_results: int = 5) -> dict:
    """Search OpenAlex for papers. Returns titles, years, DOIs, citation counts and abstract excerpts.

    Args:
        query: Search terms, e.g. "lead-free halide perovskite photovoltaic absorber".
        max_results: Number of papers to return (1-10).
    """
    max_results = max(1, min(int(max_results), 10))
    resp = requests.get(
        "https://api.openalex.org/works",
        params={"search": query, "per_page": max_results, "sort": "relevance_score:desc"},
        headers={"User-Agent": "omnigent-lab (hackathon research prototype)"},
        timeout=30,
    )
    resp.raise_for_status()
    papers = []
    for w in resp.json().get("results", []):
        abstract = ""
        inv = w.get("abstract_inverted_index") or {}
        if inv:
            words = sorted((pos, word) for word, positions in inv.items() for pos in positions)
            abstract = " ".join(word for _, word in words)[:600]
        papers.append({
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "doi": w.get("doi"),
            "cited_by": w.get("cited_by_count"),
            "abstract_excerpt": abstract,
        })
    return {"query": query, "source": "OpenAlex (api.openalex.org)", "papers": papers}


# ── live screening campaign ─────────────────────────────────────────────


@mcp.tool()
def start_campaign(name: str, budget: int = 100, min_slme: float = 30.0) -> dict:
    """Start a screening campaign with a fixed budget of expensive calculations.

    Args:
        name: Short campaign name, e.g. "c1".
        budget: Maximum number of materials whose SLME may be revealed.
        min_slme: Efficiency target for a hit, in percent.
    """
    path = _campaign_path(name)
    if path.exists():
        raise ValueError(f"campaign {name!r} already exists")
    c = {"name": name, "budget": int(budget), "min_slme": float(min_slme), "started": _now(), "measured": [], "batches": []}
    _save_campaign(c)
    _append_record({"kind": "campaign_started", "agent": "tool", "summary": f"Campaign {name}: budget {budget}, target SLME >= {min_slme}%"})
    return {"campaign": name, "budget": budget, "min_slme": min_slme, "candidates": int(len(engine.candidates()))}


@mcp.tool()
def select_candidates(
    campaign: str,
    strategy: str = "surrogate",
    k: int = 10,
    explore: float = 0.5,
    warm_start: bool = True,
    leakage_guard: bool = True,
    exclude_elements: list[str] | None = None,
    gap_min: float | None = None,
    gap_max: float | None = None,
    any_of_elements: list[str] | None = None,
) -> dict:
    """Propose the next materials to calculate. Free: reveals no SLME values.

    Args:
        campaign: Campaign name.
        strategy: "random", "heuristic" (cheap band gap near 1.1 eV) or "surrogate" (active learning).
        k: Number of materials to propose (max 20).
        explore: Surrogate only: weight on model uncertainty (0 = exploit best predictions).
        warm_start: Surrogate only: also learn from prior calculations outside the candidate set.
        leakage_guard: Surrogate only: drop prior rows sharing a formula with any candidate.
        exclude_elements: Element symbols to avoid, e.g. ["Se"].
        gap_min: Pre-filter: minimum cheap OptB88vdW band gap, eV.
        gap_max: Pre-filter: maximum cheap OptB88vdW band gap, eV.
        any_of_elements: Pre-filter: keep only materials containing at least one of these elements.
    """
    if strategy not in engine.STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(engine.STRATEGIES)}")
    c = _load_campaign(campaign)
    cand = engine.candidates(_prefilter(gap_min, gap_max, any_of_elements, exclude_elements))
    measured = [i for i in cand.index if cand.at[i, "jid"] in set(c["measured"])]
    rng = np.random.default_rng(len(measured))
    k = max(1, min(int(k), MAX_BATCH))
    picks = engine.STRATEGIES[strategy](
        cand, measured, k, rng,
        explore=explore, warm_start=warm_start, leakage_guard=leakage_guard,
    )
    rows = cand.loc[picks]
    c.setdefault("selections", []).append({
        "time": _now(), "strategy": strategy, "pool_size": int(len(cand)),
        "prefilter": _prefilter(gap_min, gap_max, any_of_elements, exclude_elements),
    })
    _save_campaign(c)
    out = []
    if strategy == "surrogate":
        train = cand.loc[measured]
        if warm_start:
            train = engine.pd.concat([train, engine.prior_data(leakage_guard)])
        mean, std = engine.predict(engine.fit_surrogate(train, 0), rows)
    for n, (_, r) in enumerate(rows.iterrows()):
        item = {"jid": r["jid"], "formula": r["formula"], "optb88vdw_gap_eV": round(float(r["optb88vdw_bandgap"]), 3), "ehull": round(float(r["ehull"]), 4)}
        if strategy == "surrogate":
            item["predicted_slme"] = round(float(mean[n]), 1)
            item["uncertainty"] = round(float(std[n]), 1)
        out.append(item)
    return {"campaign": campaign, "strategy": strategy, "proposed": out, "budget_left": c["budget"] - len(c["measured"])}


@mcp.tool()
def run_calculations(campaign: str, jids: list[str], requested_by: str = "experiment_runner") -> dict:
    """Run the expensive calculation (reveal SLME and MBJ gap) for the given materials. Costs budget.

    Args:
        campaign: Campaign name.
        jids: JARVIS IDs from select_candidates, at most 20 per call.
        requested_by: Agent making the request, for the research record.
    """
    c = _load_campaign(campaign)
    if len(jids) > MAX_BATCH:
        raise ValueError(f"at most {MAX_BATCH} calculations per call")
    new = [j for j in dict.fromkeys(jids) if j not in set(c["measured"])]
    if not new:
        raise ValueError("all of these materials were already calculated in this campaign")
    left = c["budget"] - len(c["measured"])
    if len(new) > left:
        raise ValueError(f"budget exceeded: {len(new)} requested, {left} left. Ask the scientist to approve more budget.")
    df = engine.load_pool().set_index("jid")
    unknown = [j for j in new if j not in df.index]
    if unknown:
        raise ValueError(f"unknown JARVIS IDs: {unknown}")
    rows = df.loc[new]
    hits = engine.is_hit(rows, c["min_slme"])
    results = [
        {
            "jid": j,
            "formula": rows.at[j, "formula"],
            "slme_percent": round(float(rows.at[j, "slme"]), 2),
            "mbj_gap_eV": None if np.isnan(rows.at[j, "mbj_bandgap"]) else round(float(rows.at[j, "mbj_bandgap"]), 3),
            "hit": bool(hits.at[j]),
            "toxic_flag": bool(rows.at[j, "toxic"]),
        }
        for j in new
    ]
    c["measured"].extend(new)
    c["batches"].append({"time": _now(), "requested_by": requested_by, "jids": new, "hits": int(hits.sum())})
    _save_campaign(c)
    total_hits = int(engine.is_hit(df.loc[c["measured"]], c["min_slme"]).sum())
    _append_record({
        "kind": "experiment_result",
        "agent": requested_by,
        "summary": f"{campaign}: ran {len(new)} calculations, {int(hits.sum())} hits (total {total_hits} in {len(c['measured'])})",
        "evidence": {"source": JARVIS, "jids": new},
    })
    return {
        "campaign": campaign,
        "results": results,
        "batch_hits": int(hits.sum()),
        "total_hits": total_hits,
        "calculations_used": len(c["measured"]),
        "budget_left": c["budget"] - len(c["measured"]),
        "source": JARVIS,
    }


@mcp.tool()
def campaign_status(campaign: str) -> dict:
    """Show a campaign's budget, hits so far, and the best materials found.

    Args:
        campaign: Campaign name.
    """
    c = _load_campaign(campaign)
    df = engine.load_pool().set_index("jid")
    done = df.loc[c["measured"]] if c["measured"] else df.iloc[:0]
    hits = engine.is_hit(done, c["min_slme"]) if len(done) else []
    best = done[hits].sort_values("slme", ascending=False).head(10) if len(done) else done
    return {
        "campaign": campaign,
        "min_slme": c["min_slme"],
        "calculations_used": len(c["measured"]),
        "budget_left": c["budget"] - len(c["measured"]),
        "hits": int(sum(hits)) if len(done) else 0,
        "batches": [{"requested_by": b["requested_by"], "n": len(b["jids"]), "hits": b["hits"]} for b in c["batches"]],
        "selections": c.get("selections", []),
        "best_hits": [{"jid": j, "formula": r["formula"], "slme_percent": round(float(r["slme"]), 2)} for j, r in best.iterrows()],
    }


# ── retrospective experiments ───────────────────────────────────────────


@mcp.tool()
def compare_strategies(
    strategies: list[str],
    seeds: int = 5,
    min_slme: float = 30.0,
    target_hits: int = 10,
    batch: int = 10,
    explore: float = 0.5,
    warm_start: bool = True,
    leakage_guard: bool = False,
    gap_min: float | None = None,
    gap_max: float | None = None,
    any_of_elements: list[str] | None = None,
    exclude_elements: list[str] | None = None,
) -> dict:
    """Matched-conditions benchmark: how many calculations each strategy needs to find target_hits hits.

    Replays full campaigns against the hidden values, same pool, seeds and batch
    size for every strategy. Speedup is relative to random screening.

    Args:
        strategies: Any of "random", "heuristic", "surrogate". Include "random" to get speedups.
        seeds: Number of repeated campaigns per strategy (1-10).
        min_slme: Efficiency target for a hit, in percent.
        target_hits: Hits to find before a campaign stops.
        batch: Materials per round.
        explore: Surrogate exploration weight.
        warm_start: Surrogate learns from prior calculations outside the candidate set.
        leakage_guard: Drop prior rows sharing a formula with any candidate.
        gap_min: Pre-filter (two-stage pipeline): minimum cheap OptB88vdW band gap, eV.
        gap_max: Pre-filter: maximum cheap OptB88vdW band gap, eV.
        any_of_elements: Pre-filter: keep only materials containing at least one of these elements.
        exclude_elements: Pre-filter: drop materials containing any of these elements.

    With any pre-filter set, all listed strategies screen the filtered pool and a
    random baseline on the full pool is added; the result reports the filter's
    hit recall (share of all hits it keeps). Fast but low-recall filters discard
    discoveries: always weigh speedup against recall.
    """
    seeds = max(1, min(int(seeds), 10))
    t0 = time.time()
    result = engine.benchmark(
        strategies, list(range(seeds)), min_slme=min_slme, target_hits=target_hits, batch=batch,
        prefilter=_prefilter(gap_min, gap_max, any_of_elements, exclude_elements),
        explore=explore, warm_start=warm_start, leakage_guard=leakage_guard,
    )
    result["runtime_s"] = round(time.time() - t0, 1)
    result["source"] = JARVIS
    _append_record({"kind": "benchmark", "agent": "tool", "summary": f"compare_strategies {strategies}: {result.get('speedup_vs_random')}", "evidence": result["conditions"]})
    return result


@mcp.tool()
def test_prefilter(
    gap_min: float | None = None,
    gap_max: float | None = None,
    any_of_elements: list[str] | None = None,
    exclude_elements: list[str] | None = None,
    min_slme: float = 30.0,
) -> dict:
    """Retrospective test of a cheap pre-filter rule: how many hits it keeps and non-hits it removes.

    Costs no budget. Uses only free information to filter, then scores the rule
    against the known values.

    Args:
        gap_min: Minimum cheap OptB88vdW band gap, eV.
        gap_max: Maximum cheap OptB88vdW band gap, eV.
        any_of_elements: Keep only materials containing at least one of these elements.
        exclude_elements: Drop materials containing any of these elements.
        min_slme: Efficiency target for a hit, in percent.
    """
    pf = _prefilter(gap_min, gap_max, any_of_elements, exclude_elements)
    if not pf:
        raise ValueError("set at least one filter rule")
    base = engine.candidates()
    kept = engine.candidates(pf)
    base_hit = engine.is_hit(base, min_slme)
    kept_hit = engine.is_hit(kept, min_slme)
    result = {
        "rule": pf,
        "pool_before": int(len(base)),
        "pool_after": int(len(kept)),
        "hits_before": int(base_hit.sum()),
        "hits_after": int(kept_hit.sum()),
        "hit_recall": round(float(kept_hit.sum() / base_hit.sum()), 3),
        "non_hits_removed": round(float(1 - (~kept_hit).sum() / (~base_hit).sum()), 3),
        "hit_rate_before": round(float(base_hit.mean()), 4),
        "hit_rate_after": round(float(kept_hit.mean()), 4) if len(kept) else 0.0,
        "not_testable": "Direct vs indirect band gap is not in the JARVIS fields used here; rules needing it cannot be tested.",
        "source": JARVIS,
    }
    _append_record({"kind": "benchmark", "agent": "tool", "summary": f"test_prefilter {pf}: recall {result['hit_recall']}, removed {result['non_hits_removed']} of non-hits", "evidence": pf})
    return result


@mcp.tool()
def surrogate_error_by_family(min_family_size: int = 30, leakage_guard: bool = True) -> dict:
    """Where does the surrogate model fail? Cross-validated SLME error grouped by anion family.

    Trains on prior calculations and predicts every candidate, then reports mean
    absolute error per family (halides, chalcogenides, pnictides, other).

    Args:
        min_family_size: Families with fewer candidates are merged into "other".
        leakage_guard: Drop prior rows sharing a formula with any candidate.
    """
    cand = engine.candidates()
    mean, std = engine.predict(engine.fit_surrogate(engine.prior_data(leakage_guard), 0), cand)
    err = np.abs(mean - cand["slme"].values)

    def family(elements: str) -> str:
        s = set(elements.split())
        if s & {"F", "Cl", "Br", "I"}:
            return "halide"
        if s & {"O"}:
            return "oxide"
        if s & {"S", "Se", "Te"}:
            return "chalcogenide"
        if s & {"N", "P", "Sb", "Bi"}:
            return "pnictide"
        return "other"

    fam = cand["elements"].map(family)
    out = {}
    for f in sorted(fam.unique()):
        m = (fam == f).values
        if m.sum() < min_family_size:
            continue
        out[f] = {"n": int(m.sum()), "mae_slme": round(float(err[m].mean()), 2), "mean_uncertainty": round(float(std[m].mean()), 2)}
    return {"overall_mae_slme": round(float(err.mean()), 2), "by_family": out, "note": "Trained only on prior data (no candidate labels).", "source": JARVIS}


# ── shared research record ──────────────────────────────────────────────


@mcp.tool()
def record_entry(kind: str, agent: str, summary: str, evidence: str = "", agent_generated_hypothesis: bool = False) -> dict:
    """Append an entry to the lab's append-only research record.

    Args:
        kind: One of "question", "evidence", "hypothesis", "plan", "decision", "result", "critique", "next_step".
        agent: The agent writing the entry, e.g. "planner".
        summary: One or two sentences.
        evidence: Citations, DOIs, JARVIS IDs or tool results that support the entry.
        agent_generated_hypothesis: True when the content is an untested idea produced by an agent.
    """
    allowed = {"question", "evidence", "hypothesis", "plan", "decision", "result", "critique", "next_step"}
    if kind not in allowed:
        raise ValueError(f"kind must be one of {sorted(allowed)}")
    return _append_record({"kind": kind, "agent": agent, "summary": summary, "evidence": evidence, "agent_generated_hypothesis": agent_generated_hypothesis})


@mcp.tool()
def read_record(last_n: int = 20) -> dict:
    """Read the most recent entries of the research record, oldest first.

    Args:
        last_n: Number of entries to return.
    """
    if not RECORD.exists():
        return {"entries": []}
    lines = RECORD.read_text().splitlines()[-max(1, int(last_n)):]
    return {"entries": [json.loads(line) for line in lines]}


def _forbid_unknown_arguments() -> None:
    """Make every tool reject parameters it does not declare.

    In the first live run a planner passed filter settings the tool did not
    support; they were silently ignored and the run tested something else.
    """
    for t in mcp._tool_manager.list_tools():
        model = t.fn_metadata.arg_model
        model.model_config["extra"] = "forbid"
        model.model_rebuild(force=True)


_forbid_unknown_arguments()


if __name__ == "__main__":
    mcp.run()
