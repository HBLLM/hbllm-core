# HBLLM — Phase 2 Review #3: Real Generalization, Unreliable Measurement

**Date:** 2026-11-15  
**Verified SHA:** `199e6b17` (30 commits after `90e378e2`)  
**Predecessors:** `2026-11-10_phase2_navigation_diagnosis.md` (P2-1…P2-9), `2026-11-09_post_implementation_review.md`  
**Evidence:** `reports/audit/probes/2026-11-15/` (probe scripts + raw `results.txt`)  
**Audience:** the next implementing agent

---

## 0. Verdict

**Behaviour: right direction, and the result is real.** I wrote nine task variants that exist
in no suite — four surface changes to GridNav and four new mechanics — and ran the composed
system on them cold. It solved most of them, including a depth-2 key chain and a hidden
remote-switch mechanic it had never seen, and it learns from a single death. Nothing in the
core references the benchmark envs. This is the first evidence of generalization in this
project that I produced myself rather than read from a report.

**Process: wrong direction in four ways.** The held-out split was run after six consecutive
engine commits and is now a development set; the headline scores are computed from **one seed
per task (n = 4 episodes)**; the grounding stack has **exactly zero** effect on any action;
and the codebase grew 11k lines past the budget with no file deleted while the ablation report
says ~35 faculties do nothing.

**Production: not close.** The served API and the benchmarked cognitive system are still two
different programs. §6 lists what is missing.

---

## 1. What was verified

| Item | Status | Evidence |
|---|---|---|
| P2-7 undo discipline | Done | `999999` gone; all inhibitions bounded (`cortex_assimilator.py:461,615,…`) |
| P2-8 curiosity goal lifecycle | Done | `satisfied_when` (`cortex_planner.py:107,244-280`, `cortex_assimilator.py:354-401`) |
| P2-9 frontier exploration | Done, with a hole | `cortex_planner.py:1682+`; see §3.2 |
| Demonstration mode | Done | `runner.py:86,142` |
| Zero-shot headline separated from lifelong | Done | dev `1dc9213a`: zero-shot 0.7672, lifelong 0.8806 |
| Privileged metadata removed from ContinuousReach | Done (`fe167153`) | no core reader ever existed (`git log -S obstacle_center -- hbllm` empty) |
| Ablation on real harness | Done | `reports/ablation/20261114_171041.json` |
| Core free of benchmark names | Yes | `rg -i "minigrid\|frozenlake\|grid_nav" hbllm/hcir` → comments only |
| Full test suite | **7219 passed, 9 skipped** | 43 min wall-clock, run alone |
| ruff check / format | Clean | 788 files |
| LOC ≤ 250k | **Broken: 260,740** | +34,740 / −2,504 since `90e378e2`; 0 files deleted |
| HEAD measured | **No** | newest dev JSON is `1dc9213a`; `199e6b17` (35-file refactor, +2,416/−848) has no report |
| AGENTS.md banner | **Stale** (said 0.0000) | corrected with this review |

---

## 2. Independent generalization probe

Fresh system per seed, 6 seeds per variant, seeds never used elsewhere, 3–4 episodes with
shared memory. `zero-shot` = episode 1; `learned` = final episode.

| Variant | What is new | Zero-shot | Learned |
|---|---|---|---|
| base GridNav (control) | — | 4/6 | 6/6 |
| palette_swap | every role on a different colour; old key colour is now the goal | 4/6 | 6/6 |
| transposed | dividing wall horizontal | 4/6 | 6/6 |
| action_relabel | action ids mean different directions | 4/6 | 6/6 |
| **bg_nonzero** | floor = 6, walls = 0 | 4/6 | **3/6** |
| **two_keys** | 3 rooms, 2 doors, 2 keys (depth-2 prerequisite) | **6/6** | 6/6 |
| **switch_gate** | no door tile; a plate removes an ordinary wall cell | **6/6** | 6/6 |
| **decoy** | two unique tiles, one kills, one wins, colours random per seed | 2/6 | **6/6** |
| **maze** | 15×15 perfect maze, fully observable | 4/6 | 4/6 |

Reading this honestly:

- Palette, orientation and action-id invariance hold exactly (identical outcomes to control).
  Roles and action semantics are learned, not assumed.
- `two_keys` and `switch_gate` at 6/6 zero-shot is the strongest result: the conditional-
  affordance + curiosity machinery composes on mechanics nobody tuned for.
- All zero-shot failures in the GridNav family are **hazard deaths** (an unknown tile is
  explored by stepping on it). After one death the agent never repeats it. That is correct
  learning and unsafe exploration (§3.4).
- n = 6 per row. These are indications, not statistics. All worlds are small, fully
  observable, deterministic, single-agent grids — the regime the engine was built in.

---

## 3. Defects found (each reproducible from the probe directory)

### 3.1 Negative transfer when background ≠ 0 (`probe.py bg_nonzero`)
With floor = 6 and walls = 0, learning makes the agent **worse**: seed 910005 succeeds
episode 1 then times out twice; 910002 goes 22 → 42 steps; 910003 never solves. Somewhere the
value `0` is still privileged as empty/background. Candidates:
`cortex_assimilator.py:498,832` (`bg_feature if … else 0`), `cortex_perception.py:948`
(`np.nonzero(labels)`), and ~29 similar sites across `cortex_*.py`. Also investigate why
consolidated knowledge from a *successful* episode degrades the next one — that is a
consolidation bug independent of the palette.
*Acceptance:* `bg_nonzero` matches control on zero-shot and learned.

### 3.2 Agent freezes in corridors (`maze_trace.py`, seed 920002)
BFS shortest path is 64; the agent uses 250 steps to visit **3 cells**. Every step reports
`goal_target=None, plan_source=arbitration, phase=epistemic_exploration`, with a rolling set
of `inhibited_action_*_cooldown` vetoes. The frontier fallback (`cortex_planner.py:1684`,
guarded by `if not goals`) never produces a goal here, the visible unique goal tile is never
proposed as an epistemic target, and inhibition cooldowns are the only thing choosing actions.
Two of six mazes fail this way, all four episodes each — no learning rescues it.
*Acceptance:* maze 6/6; a unit test that from any start with ≥1 reachable unvisited cell,
`meta["goal_target"]` is non-None within 3 steps.

### 3.3 Grounding is behaviourally inert
`no_grounding` equals `system` to four decimals in **all eight** held-out reports and every
dev report (0.2067/0.2067 … 0.8744/0.8744). `ground.py` confirms it directly: with
`enable_grounding` on vs off, the action sequences are **identical** on 6/6 runs across three
env families. No planner, deliberation or motor code reads any `sge_*` signal
(`rg "sge_|grounding_bridge" cortex_planner.py cortex_deliberation.py cortex_motor.py` → 0
relevant hits). Meanwhile it costs ~40 % more workspace nodes (1,396 vs 980 after 166 steps)
and roughly doubles median step latency in my (contended) measurement.
Decide explicitly — see R4.

### 3.4 Exploration is not risk-aware
Unknown tiles are tested by stepping on them: 2/6 zero-shot deaths in every GridNav variant,
4/6 in `decoy`. Acceptable in a simulator, disqualifying for anything with real actuators.

### 3.5 Latency and growth
Harness reports p50 28–38 ms, p95 57–98 ms per step on 10×10 grids (uncontended). On the 15×15
maze I measured p50 173 ms / max 1.1 s (machine under load — treat as an upper bound, but the
scaling with grid area is real). AGENTS.md Invariant 3 says "sub-millisecond". State grows
linearly and without bound: ≈6–8 workspace nodes and ≈36 KB of event store per step, no
compaction or retention.

---

## 4. Measurement integrity

1. **Held-out is spent.** Eight held-out runs exist; six followed consecutive engine commits
   (`1ba1f412` 0.21 → `1c32d02f` 0.31 → `468d09fc` 0.52 → `aeca4f6e` 0.87 → `edbc9c41` 0.88 →
   `562e1c82` 0.87), with commit titles announcing the score moving. Whether or not anyone
   read per-task results, the split has guided development and can no longer certify anything.
   The env hash also changed twice underneath it.
2. **n = 4.** `run_closed_loop_task` uses `task_seed = seeds[0]` (`runner.py:51-52`) for the
   zero-shot, lifelong and imitation numbers. Each manifest lists 20 seeds per task; 19 are
   used only by the cross-seed transfer metric (first 5). "Zero-shot solve rate 0.75" is 3 of
   4 episodes; one flip moves the headline 25 points. No confidence interval is possible.
3. **Lifelong = replaying one layout.** Episodes 2–5 reuse the same seed, so lifelong ≈ oracle
   (0.8806 vs 0.8814) measures path memorisation. Cross-seed transfer (0.94) is the meaningful
   learning metric and should be the second headline.
4. **Same families in dev and held-out.** Still true. `independent.yaml` (MiniGrid/FrozenLake
   wrappers) is declared `split: dev`, and random scores 0.29 on it — weak discrimination.
5. **Runner swallows engine exceptions** (`runner.py`: `except Exception: pass` around
   `on_boundary` and the terminal `step`). A crash there is silently a non-event.

My §2 probe is itself now seen by me and by whoever reads this; it is a **dev-side robustness
suite**, not a held-out set.

---

## 5. Plan for the next agent (in order; one PR each; dev JSON before/after)

### R1 — Repair the measuring instrument (before any engine change)
- Runner iterates **all** seeds per task for zero-shot; report solve rate with a Wilson 95 %
  interval per task and family. Headlines: zero-shot (all seeds) and cross-seed transfer.
  Rename current "lifelong" to `same_layout_replay`.
- Retire held-out v1 into dev. Create **held-out v2**: new seed range, ≥ 2 families absent
  from dev (partial observability; stochastic transitions; a moving agent; a non-grid family),
  authored without looking at engine internals.
- Add `reports/heldout/LEDGER.md` and make `hbllm bench heldout` refuse to run unless the SHA
  is tagged `eval-*` and append to the ledger. Target cadence: once per milestone.
- Remove the `except Exception: pass` blocks in the runner; engine errors are task failures
  recorded in `errors`.
- Promote `reports/audit/probes/2026-11-15/{probe,mech}.py` into `benchmarks/envs/` +
  `benchmarks/suites/novelty.yaml` (dev split).
- *Acceptance:* dev report shows n ≥ 20 per task with intervals; test that the runner uses
  every manifest seed; a second held-out invocation on an untagged SHA exits non-zero.

### R2 — Measure HEAD
Run dev on `199e6b17`. A 35-file refactor with no measurement is exactly what the Phase 2
rule forbids. If any family drops, bisect before proceeding.

### R3 — Fix §3.1 and §3.2 generally
No palette constants; background is whatever the theory says it is. Frontier/novel-feature
goal must exist whenever reachable unvisited space exists; inhibition may veto, never choose.
Use decision provenance to find the first wrong stage — do not guess.

### R4 — Grounding: wire it or switch it off
Pick one, with evidence:
(a) define a task where grounding *should* matter (identity through occlusion; an instructed
goal "go to the <concept>"; transfer of a learned concept to a new palette without
re-exploration), route `sge_*` outputs into goal proposal, and show `system > no_grounding`
with non-overlapping intervals; or
(b) set `enable_grounding=False` by default and move it out of the step path until (a) exists.
Shipping 7k+ lines that cost latency and graph growth for a measured Δ of 0.0000 is not a
neutral choice.

### R5 — Subtract (overdue)
The ablation is good work; act on it.
- Delete the 33 faculties with `calls: 0`.
- Demote every `delta_score: 0.0` faculty out of the default pipeline. This includes several
  that AGENTS.md §2 lists as "kept": `surprise_engine`, `collision_cones`,
  `predictive_coding_cortex`, `snn_predictor`, `neuromodulatory_system`,
  `precision_controller`, `dorsal_kinetic_stream`, `deformation_tracker`,
  `metacognitive_agency_cortex`. Measured keepers are four: `subgoal_stack`, `habenular_ior`,
  `prediction_budget`, `acc_conflict_monitor`. Re-run the ablation on the R1 runner first —
  with n = 4 and sd floored at 0.05 the current rule cannot detect small real effects, so
  demote, re-measure, then delete.
- Move `hcir/world/experimental/morphology_arc.py` to `plugins/arc_agi_adapter/` (Invariant 7).
- Back under 250k LOC, enforced by an architecture test, not a text file.
- Update the AGENTS.md faculty roster to the measured one.

### R6 — Structural debt in the decision core
`autonomous_epistemic_engine.py` 6,214 lines, `cortex_planner.py` 5,188; **321 `hasattr` and
290 `getattr(view|engine|self, …, default)`** in `hcir/world`; 71 `except Exception: pass` in
`hbllm/`; 6 files still call `sqlite3.connect` directly. These are the Invariant 8 / protocol
rules AGENTS.md already states. Set a ratchet test (count may only go down).

### R7 — Risk-aware exploration
Before entering a never-tested feature, prefer (i) plans that avoid it, (ii) evidence from
analogous features/concepts, (iii) reversible probes. Report "deaths during exploration" as a
harness metric. This is where grounding (R4a) could earn its place.

### R8 — Bounded state and a real latency budget
Event-store compaction/snapshotting; retention policy per tier; node budget per world with
consolidation. Publish a per-stage latency profile; either meet a stated budget (say p95
≤ 20 ms at 32×32) with the Rust hot path, or change the "sub-millisecond" claim.

### R9 — Only then widen
Partial observability, stochastic dynamics, other agents, long horizons, continuous control
beyond one reach task, language-specified goals. Each as a dev family first; each new
capability admitted only with an R1-quality before/after.

**Do not:** add faculties or "G-item" hooks; run held-out v2 while iterating; report a
headline from fewer than 20 seeds.

---

## 6. What is missing for production

"Production" needs a definition first: *which* system, for *whom*, doing *what*. Today there
are two — an LLM chat/agent server and an embodied cognitive engine — and only the second is
measured.

**Product and architecture**
- `hbllm/serving` does not use `build_cognitive_system` (`rg build_cognitive_system
  hbllm/serving` → 0; `serving/boot.py` constructs its own `ExecutiveRuntime`). The thing the
  Docker image serves is not the thing the benchmarks evaluate. Unify, or ship and document
  them as separate products.
- No stated SLOs (latency, availability, durability), no capacity model. The engine is
  synchronous single-threaded Python, one instance per world; there is no story for N tenants
  × M worlds.
- Public claims: README SEO line advertises "Open Source AGI". The evidence supports "solves
  small fully-observable grid tasks zero-shot and adapts in one episode".

**Reproducibility and build**
- No Python lockfile; all dependencies are `>=`. Add `uv.lock`/pinned constraints.
- Dockerfile builds on Python 3.11 / Rust 1.80; development and all results are on 3.12.
- CI runs `tests/unit tests/modules` with `--timeout=60`; the full suite takes 43 minutes
  locally and at least one test exceeds 60 s under load. No CI job runs the dev benchmark,
  the LOC gate, or a score-regression check. Add a nightly dev-bench job with a floor.

**Security**
- Authentication **fails open**: without a bearer token and with `HBLLM_ENV` unset or
  anything other than `production`, requests are served under a local "sovereign identity"
  (`serving/auth.py:78-90`). Default must be deny; dev mode must be opt-in.
- JWT secret falls back to a random ephemeral value with a warning (`auth.py:~40`); HS256
  shared secret only; no key rotation.
- Several handlers fall back to `tenant_id="default"` or to a tenant id from the request
  body (`api.py:566-735`, `:688`) — a tenant-isolation hazard if the middleware is ever
  bypassed. Make tenant resolution a required dependency that raises.
- `pickle.loads` on sandbox IPC payloads (`hcir/kernel/capability_sandboxing.py:462`).
  Replace with a schema'd format.
- No threat model for the agent's own actions: constitutional policies need real content,
  an operator kill switch, and an audit trail export.

**Data and durability**
- Unbounded event-store growth (§3.5); no compaction, backup/restore procedure, or
  retention policy; no versioning/migration path for persisted HCIR node schemas (Alembic
  covers the SaaS DB, not the event store).
- SQLite single-writer per world; connection cleanup relies on `__del__`
  (`sqlite_event_store.py:422`). Needs explicit lifecycle, WAL settings, crash-recovery test
  (kill −9 mid-commit → replay → identical state).

**Reliability and operations**
- 71 silent `except Exception: pass`; failures must be counted and surfaced.
- Decision provenance exists but is not exported: wire `meta` into OpenTelemetry spans and
  expose per-stage latency, goal-source and veto counters at `/metrics`.
- No soak test (hours of continuous stepping), no load test, no chaos test, no fuzzing of
  observation adapters with malformed lattices/shapes.
- Graceful shutdown/resume of a running world (flush, snapshot, rehydrate) is untested end
  to end under the server.

**Safety for embodiment**
- §3.4: exploration by dying. Needs risk-aware exploration, a simulate-before-act gate with a
  real model, and human override before any physical or high-stakes actuator.

**Governance of the project itself**
- The scope rule has now been broken three times (17 commits on day one; G7–G15 hooks in
  `a332e92d`; +11k LOC). Encode it: LOC ratchet, hasattr ratchet, "new file under
  `hcir/world` requires an ablation delta" check in CI. Rules in a markdown banner have not
  held.
- License is GPL-3.0-only; confirm that matches the intended deployment/commercial model.

---

## 7. Bottom line

The engine now does something no earlier version did: it solves tasks it was not built for
and gets better with one exposure. That deserves to be measured properly — which it currently
is not. The next step is instrument first (R1–R2), two real bugs second (R3), then the
decisions that have been deferred three times: what grounding is for (R4) and what gets
deleted (R5). Production is a separate track that starts with deciding what the product is.
