# Falsify Labs

**An Omnigent AI lab that hunts for lead-free solar materials, and tries to prove itself wrong before it believes anything.**

Hack-Nation 7th Global AI Hackathon · Challenge 03: Agentic Scientific Discovery (Databricks)

**Live page:** [exact-view-wow.lovable.app](https://exact-view-wow.lovable.app) · **Demo video (2 min):** [docs/demo.mp4](docs/demo.mp4)

Six Claude agents, orchestrated by Omnigent on Databricks, run the discovery loop on 10,022 materials from JARVIS-DFT: they read the literature, form hypotheses, plan experiments under a calculation budget, run them, and send every result to a Skeptic before the lab acts on it.

> Status: hackathon prototype. Every number below comes from a logged run (`runs/`). All results are computational (DFT), not lab measurements.

---

## The question

The best new solar cells (halide perovskites) contain **lead**, which is toxic. Which **stable, non-toxic** materials could absorb sunlight as efficiently, and how can a lab find them with far fewer expensive calculations?

- **Hit:** SLME ≥ 30% (spectroscopic limited maximum efficiency), energy above hull ≤ 0.05 eV/atom (stable), no Pb, Cd, Hg, Tl or As.
- **Free information:** composition, a cheap DFT band gap (OptB88vdW), stability.
- **Expensive information:** SLME (needs MBJ + optics calculations). Revealing it costs one unit of budget.

## The bottleneck we attack

**Expensive calculations per discovery.** We measure it as the number of SLME calculations needed to find 10 hits, compared with random screening of the same pool under matched conditions (same pool, seeds, batch size and budget).

## Measured improvement

Bottleneck: expensive calculations per discovery. Pool: 1,362 stable non-toxic candidates, 129 hits (9.5% base rate). Each row is 10 repeated campaigns (seeds 0–9), leakage guard on; speedup = median calculations to 10 hits for random screening of the full pool divided by the strategy's median. Run by the agents; reproducible with `labmcp.engine.benchmark`.

| Strategy | Calculations to 10 hits | Speedup | Hits still reachable |
|---|---|---|---|
| Random screening, full pool | 106.8 ± 33.1 (mean ± sd) | 1.0× | 100% |
| Textbook heuristic (cheap gap near 1.1 eV) | 47.6 ± 2.8 | 2.1× | 100% |
| **Active learning, warm-started from 8,660 prior calculations** | **33.8 ± 1.8** | **3.0×** | **100%** |
| Cheap gap filter 0.5–1.5 eV + active learning (Round 3) | median 25 | 4.1× | 61% |
| Same, picking one material at a time (sensitivity) | — | 6.4× | 61% |

**Headline: 3.0× fewer expensive calculations per discovery, with nothing discarded.** Filtering buys up to 4.1×, and sequential picking 6.4×, at the cost of 39% of the hits. The Skeptic decomposed the 4.1×: the filter alone gives 2.7×, the learning model adds 1.5× on top.

The benchmark is conservative: every simulated campaign spends its first 20 calculations on random picks. The live campaign below starts from the model and did better.

## What the lab did, round by round

All entries are in the research record (`runs/record.jsonl`), with sources.

**Round 1: is the learning loop worth it?** Literature found published active-learning and SLME screening work (OpenAlex, with DOIs). Hypothesis proposed three ideas (agent-generated). The Planner compared options and chose a zero-budget retrospective benchmark. Result: active learning 3.0×, heuristic 2.1×. Skeptic: *trusted with caveats*; the speedup survives the leakage guard (3.1× → 3.0×) and 10 seeds, but the run had not tested the designed hypotheses: the tool had **silently dropped** unsupported filter settings. Fix: tools now reject arguments they do not declare.

**Round 2: the lab refutes its own hypothesis.** H1 (agent-generated): a cheap pre-filter (gap 0.7–1.2 eV and an ns² cation: Bi, Sb, Sn, Ge, Cu) would keep ≥75% of hits. The Planner set a gate before spending anything: abort if recall < 60%. `test_prefilter` returned **16% recall** (21 of 129 hits, 41 candidates), so the Runner aborted, with zero budget spent. The Skeptic then ran an ablation:

| Cheap rule | Hits kept | Non-hits removed |
|---|---|---|
| gap 0.7–1.2 eV + ns² cation (H1) | 16% | 98% |
| ns² cation only | 39% | 82% |
| gap 0.7–1.2 eV only | 36% | 94% |
| gap 0.5–1.5 eV only | **61%** | 84% |
| gap 0.3–2.0 eV only | 81% | 72% |

Verdict: H1 *not trusted*; both parts of the rule discard discoveries. The Director proposed a recalibrated gap-only filter.

**Round 3: the recalibrated pipeline.** Gap 0.5–1.5 eV + active learning, success criteria fixed in advance (median ≤ 30.5 calculations, recall ≥ 60%): both passed (median 25, 4.1×, recall 61%). Skeptic: *trusted with conditions*, and it **blocked a proposed live campaign of ≤ 20 calculations** because no simulated run had finished in fewer than 22. The Director raised the budget to 35.

**Round 4: live campaign.** Budget 35, gap 0.5–1.5 eV, active learning, batches of 10. **13 hits in 20 calculations** (65% hit rate vs 9.5% base rate); the 10th hit came at calculation 16. The lab stopped early with 15 calculations unspent.

| Material | SLME | MBJ gap (eV) | JARVIS ID |
|---|---|---|---|
| KMgBi | 33.9% | 1.36 | JVASP-35067 |
| CaMg₂Bi₂ | 33.8% | 1.14 | JVASP-4050 |
| Rb₃Sb | 33.7% | 1.40 | JVASP-8765 |
| MgTe | 33.6% | 1.41 | JVASP-36238 |
| NaCaSb | 33.5% | 1.23 | JVASP-104498 |
| Na₂AgSb | 33.4% | 1.43 | JVASP-2547 |
| YbMg₂Sb₂ | 33.3% | 1.26 | JVASP-3141 |
| K₂AgBi | 33.0% | 1.45 | JVASP-149919 |
| NaCuTe | 32.5% | 1.49 | JVASP-138749 |
| Ca₂Ge | 32.3% | 1.06 | JVASP-36395 |
| Ca₂Sn | 32.0% | 1.02 | JVASP-36382 |
| NaInTe₂ | 31.6% | 1.55 | JVASP-3504 |
| K₂NaInI₆ | 31.0% | 1.59 | JVASP-107598 |

These are **computational candidates**, not discoveries. Skeptic caveats: several are alkali compounds that are likely air- and moisture-sensitive (thermodynamic stability ≠ ambient stability), Yb is supply-critical, and the filter that found them excludes 39% of all hits. A single live run; the benchmark above is the measured result.

**Scientist's ruling (human approval, recorded as a decision).** The Skeptic flagged Yb as supply-critical and could not verify the campaign's pool size. The scientist ruled: all 13 hits stand, because the hit definition was fixed before the campaign and is not changed after seeing results; report *13 hits (12 excluding supply-critical elements)*; add supply risk as a second-stage screen. The scientist also authorized a confirmation campaign.

**Confirmation.** The agents' confirmation campaign could not start (the Databricks Free Edition session limit was reached). `scripts/reproduce_round4.py` replays the same selection offline instead: the pool is 275 candidates (filter applied), batch 1 is identical to the live run, batch 2 overlaps 9 of 10, and it finds 14 hits in 20 calculations (live: 13). This confirms reproducibility, not an independent sample: selection is deterministic.

## The discovery loop

```
Question → Literature → Hypothesis → Planner (≥2 tests, pick one) → Runner → Skeptic → Director decides next
    ▲                                                                                        │
    └──────────────────────── a surprising result reopens an earlier assumption ─────────────┘
```

| Agent | Decision it owns | Tools it may use |
|---|---|---|
| **Director** | Objective, when to stop, what to report | sub-agent delegation, `dataset_summary`, `campaign_status`, research record |
| **Literature** | What is already known | `literature_search` (OpenAlex), research record |
| **Hypothesis** | What is worth testing | `dataset_summary`, `surrogate_error_by_family`, `test_prefilter`, research record |
| **Planner** | Which test runs next (by expected learning, cost, feasibility) | `campaign_status`, research record |
| **Runner** | Spending calculation budget | campaigns, `run_calculations`, `compare_strategies`, `test_prefilter` |
| **Skeptic** | Whether a result is trusted | `compare_strategies`, `test_prefilter`, `surrogate_error_by_family`, research record |

Each agent sees only its own tools (per-agent MCP allow-lists in `lab/**/tools/mcp/lab.yaml`).

## Safety, controls and human approval

| Control | How it is enforced |
|---|---|
| Scientist approves big spends | Omnigent CEL policies on the Runner: **ASK** before a batch of more than 10 calculations or a campaign budget above 150 (`lab/agents/runner/config.yaml`) |
| Hard budget | `run_calculations` refuses requests beyond the campaign budget |
| Runaway agents | `max_tool_calls_per_session` on the Director |
| No silent misconfiguration | Tools reject arguments they do not declare (added after a live run where unsupported settings were silently ignored) |
| Toxic elements | Excluded from the candidate pool before any agent sees it |
| Traceable decisions | Append-only research record `runs/record.jsonl`: every question, hypothesis, plan, result and critique, with sources |
| Labelled hypotheses | Hypotheses are recorded with `agent_generated_hypothesis: true` |
| Citations | Literature claims carry DOIs from OpenAlex; data claims cite JARVIS IDs |

## Architecture

```
Databricks workspace ── Omnigent (managed): sessions, sub-agent hand-offs, policies, web UI
        │
        └── host: the scientist's machine (omnigent host)
                ├── agents: claude-sdk harness (Claude Code CLI, the scientist's Claude subscription)
                └── stdio MCP server  labmcp/server.py   (11 tools)
                        └── engine    labmcp/engine.py   (strategies, campaigns, benchmarks)
                                └── data/pool.csv        (JARVIS-DFT, built by scripts/build_features.py)
```

Omnigent's managed server on Databricks orchestrates the run; each agent's model calls go through the `claude-sdk` harness on the host, authenticated with the scientist's Claude subscription. Hosted Omnigent rejects uploaded Python tool files, so the lab's tools run as an MCP server on the host. Agents are YAML specs in `lab/`.

The Omnigent session transcripts (Director, Literature, Hypothesis, Planner, Runner, Skeptic) are exported to `runs/sessions/` with `omnigent session export`.

## Run it

Requirements: macOS or Linux, Python 3.12 via [uv](https://docs.astral.sh/uv/), [Omnigent](https://github.com/omnigent-ai/omnigent), a Databricks workspace with Omnigent enabled.

```sh
git clone https://github.com/jtfernandezi/falsify-labs && cd falsify-labs
./scripts/setup.sh                      # venv, JARVIS download (~48 MB), feature build
uv tool install --python 3.12 "omnigent[databricks]"
omnigent host https://<workspace>.cloud.databricks.com/omnigent      # leave running
omnigent run lab --server https://<workspace>.cloud.databricks.com/omnigent \
  -p "Run one full discovery round and tell me what experiment should come next."
```

Reproduce the benchmark without any agent:

```sh
.venv/bin/python -c "from labmcp.engine import benchmark; import json; \
print(json.dumps(benchmark(['random','heuristic','surrogate'], list(range(10)), leakage_guard=True)['results'], indent=1))"
```

## Data and evidence

- **JARVIS-DFT** `dft_3d` (93,902 materials; 10,022 with SLME). Choudhary et al., doi:[10.1016/j.commatsci.2025.114063](https://doi.org/10.1016/j.commatsci.2025.114063). Downloaded from figshare by `scripts/build_features.py`.
- **OpenAlex** for literature search ([api.openalex.org](https://api.openalex.org)).
- Run records: `runs/record.jsonl` (current) and `runs/archive/` (earlier runs, including an interrupted first run).

## Limits and what still needs validation

- SLME is a **theoretical upper bound** from DFT. Real devices lose efficiency to defects (antisites, deep traps), contacts and processing. Defect tolerance must be checked before any synthesis.
- The pool is the subset of JARVIS with SLME computed, not all possible materials.
- "Calculations" are simulated by revealing known JARVIS values; the speedup measures selection quality, not wall-clock DFT savings.
- Direct vs indirect band gap is not in the fields used, so rules that need it cannot be tested yet.

## Next experiment

1. **Novelty check** on the 13 candidates: which have never been studied as photovoltaic absorbers (literature and patent search).
2. **Practical-stability screen:** air and moisture sensitivity, and element supply risk, as a second stage after SLME.
3. **Defect tolerance** (point-defect calculations) for the survivors, then synthesis of the top 2–3 in a wet lab.
4. **Batch size study:** sequential picking reached 6.4× vs 4.1× in batches of 10; find the best batch size for a real compute cluster.

## Path to 10×

Measured: 3.0× with nothing discarded, 4.1× with a 61%-recall filter, 6.4× with sequential picking, on a pool of 1,362. Published active-learning screens report 5–100× on libraries of 10⁵–10⁶ compounds (found by the Literature agent): the advantage grows with pool size and with how rare hits are. To approach 10× at scale: screen the full JARVIS and Materials Project spaces, use multi-fidelity learning (cheap gap → MBJ → SLME) so cheap calculations steer expensive ones, and keep the Skeptic's recall check so speed never comes from discarding discoveries.

## License

MIT, see `LICENSE`.
