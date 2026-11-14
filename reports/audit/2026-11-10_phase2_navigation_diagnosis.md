# HBLLM — Phase 2 Review #1: Knowledge Is Acquired, Behaviour Does Not Follow

**Date:** 2026-11-10  
**Verified SHA:** `796eb2db` (code: `bd076556`)  
**Predecessors:** `2026-11-08_project_audit.md`, `2026-11-08_agi_gap_analysis.md`,
`2026-11-09_post_implementation_review.md` (§9–§10 define Phase 2)  
**Audience:** the agent doing the next Phase 2 PR

---

## 0. One-paragraph summary

The §10 fixes landed and are verified: envs set `EmbodiedObservation.terminal`, `episode_id` is
stable per `reset()`, the runner feeds the terminal observation back, collision-based barrier
induction fires (`barrier_features = {5}`), and a test asserts that an oracle-driven episode
populates `goal_features`. Held-out was re-run honestly and committed
(`reports/heldout/bd076556…json`): **system still 0.0000 on every family**, oracle 0.8938.
The new, decisive finding (§2) is that the failure has moved: the engine can now *acquire*
correct world knowledge — after watching one oracle episode it holds the exact goal position,
the door position, the barrier feature and the walkable set — **and still fails the identical
layout 100/100 steps**. The bottleneck is no longer perception or reward; it is the
goal→plan→motor chain, which is also completely unobservable from the outside (§3). The next
PR must make decisions explainable before trying to make them better.

---

## 1. Verification of the §10 items

| §10 item | Status | Evidence |
|---|---|---|
| Envs set `terminal` SUCCESS/FAILURE/TIMEOUT | Done | `grid_nav.py:135,194,211`, `continuous_reach.py:130`, `symbolic_rules.py:113`, `grid_push.py` |
| `episode_id` stable per reset | Done | all four envs: `episode_id=str(self.current_seed)` |
| Runner feeds terminal obs back | Done | `runner.py:112-118` (`engine.step(obs)` inside `if done:`) |
| Test: oracle episode → `goal_features` | Done | `test_closed_loop_harness.py::test_successful_oracle_episode_populates_goal_features` |
| Barrier induction from collisions | Done | probe: `barrier_features={5}` after first episode (was `∅`) |
| Goal-position evidence discipline | Partial | `cortex_planner.py:418-438` now requires `feat ∈ goal_features`; but see §2.3 — other writers bypass it |
| Held-out re-run, committed | Done | `bd076556…_handcrafted_20261110_042821.json`, `env_hash 88b8bea0d5cc4bc9` |
| **Dev re-run on the new SHA** | **Not done** | newest `reports/dev/dev_handcrafted_20261110_040408.json` has `git_sha 6ec7de10` — the engine fixes were never measured on dev before held-out was spent |
| **Lifelong mode (N episodes, shared memory, learning curve)** | **Not done** | runner still builds a fresh system per seed; `steps_to_stable_policy: null` everywhere |

Process note: the held-out split was consumed on a SHA whose dev score was never recorded. §9
said dev is where you work; held-out is run once *after* dev moves. Do not spend another held-out
run until dev shows a nonzero number.

---

## 2. The decisive experiment (reproducible, Appendix A)

One composed system, `GridNavEnvironment`, seed `100010` (avatar (8,3), key (2,4), door (6,5),
goal (3,8), walls = feature 5, door = 2, key = 3, goal = 8). Five consecutive episodes on the
*same engine instance*:

| Ep | Driver | Result | Theory after episode |
|---|---|---|---|
| A | self (fresh) | fail, 100 steps | goal `∅`, barrier `{5}`, walk `{0,3}`, phase `mental_simulation` |
| B | self, same layout | fail, 100 | goal **`{2}`** (the *door*, no success observed), walk `{0,2,3}` |
| C | **oracle-driven**, engine observing | success, 18 | goal `{2,8}`, walk `{0,2,3,8}`, `learned_goal_positions={(3,8),(6,5)}` |
| D | self, same layout, after C | **fail, 100** | unchanged; phase `exploitation` |
| E | self, new seed 100011 | fail, 100 | unchanged |

Three conclusions, each independent of the others:

### 2.1 Knowledge acquisition works; policy does not
After C the engine knows the true goal cell, the door cell, the barrier feature and the
walkable set. D shows it cannot convert that into a 17-step path it has just been shown. The
step trace (Appendix B) is diagnostic: the engine walks up column 3 from (8,3) to (1,3) —
passing (2,3), one cell left of the key at (2,4), where the oracle says RIGHT — never turns,
hits the wall, reverses down column 3 to (8,2), presses INTERACT at nothing, and repeats the
same column sweep. It is not following an A* path to any of its own known targets; every step
reports `branch=default_arbitration`. Whatever the deliberation stage is optimising, it is not
"reach a known goal position".

### 2.2 Prerequisite structure is never induced
`walkable_features` contains `2` (door) because the oracle walked through it after collecting
the key. The engine recorded "door is walkable", not "door is walkable *given* key". The
key→door causal chain that `FrontopolarSubgoalStack.decompose_causal_chain`
(`autonomous_epistemic_engine.py:5195`) is supposed to produce never appears: no subgoal for
the key, no INTERACT at the door. The theory has no representation of a conditional affordance
at all — only flat feature sets (`goal / barrier / walkable / actuator`). This is a
representational gap in `HCIRSymbolicWorldTheory`, not a tuning problem.

### 2.3 Goal evidence is still leaky
Episode B marked feature `2` (door) as a **goal** without any success signal. `induce_goal()`
(`cortex_causal.py:337-357`) is the only disciplined entry — but there are at least seven
direct writers that bypass it:
`autonomous_epistemic_engine.py:1878`, `macro_actions.py:1393,1753`, `cortex_causal.py:805`
(hydration from `role == "goal"` concept nodes), plus `candidate_goal_features` writers in
`cortex_planner.py:916`, `cortex_deliberation.py:1809`, `autonomous_learning.py:145`. Worse,
`induce_goal` *removes the feature from barriers and adds it to walkable* — so a false goal
also deletes a true barrier. This is exactly the Invariant 8 failure mode (state mutated
through many ad-hoc paths rather than one governed one). The §10 fix closed one path and left
the rest.

---

## 3. The system cannot explain a single decision

`engine.step()` returns `(action, None)` on every normal step — `meta` is only populated on
pipeline rejection (`autonomous_epistemic_engine.py:5515-5517`). The stage signals that survive
to the caller are: `branch`, `avatar_candidate`, `entity_count`, `extracted_entities`, `step`,
and the `sge_*` grounding counters. **There is no `chosen_goal`, `plan`, `subgoal`, `path`,
`arbitration_scores`, `action_source`, or `rejected_candidates`.** `CognitiveJournal` /
`ReplayDebugger` can replay *state*, but not *why an action was chosen*, because the
deliberation stage never emits it.

Practically: the next agent cannot do §9 step 1 ("diagnose with ReplayDebugger") because there
is nothing to read. This is why the previous two fix rounds were guesses that happened to be
right about perception and wrong about behaviour. Decision provenance is the prerequisite for
every subsequent Phase 2 item.

---

## 4. Work items, in order (each one PR, dev JSON before/after attached)

### P2-1 — Decision provenance (do first; no behaviour change)
Make every `step()` return a non-None `meta` with, at minimum:
`goal_target (pos|None)`, `goal_source (reward|learned_position|epistemic|none)`,
`subgoal_stack (list)`, `plan (list[pos]|None)`, `plan_source (astar|macro|escape|arbitration)`,
`candidates ({action: score})`, `chosen_action`, `veto_reasons (list)`, `phase`.
Populate it from the stages' `CognitiveResult.signals` — do not add engine attributes
(Invariant 8). Surface it in `hbllm mind step --explain` and in the runner's per-step journal.
*Acceptance:* a test that runs scenario D for 5 steps and asserts `meta["goal_target"] is not
None` and `meta["plan"]` is a non-empty list — this test is **expected to fail** and is the
first real behavioural test of the planner; keep it red until P2-3 passes it.

### P2-2 — One governed path for ontology mutation
Make `goal_features / barrier_features / walkable_features / actuator_features` private on
`HCIRSymbolicWorldTheory`, exposed read-only, mutated only via `induce_goal / induce_barrier /
induce_walkable / retract_*` which each take an **evidence** argument
(`TerminalSignal`, collision, traversal, hydration) and record a `CausalPredicate` with that
provenance. Rewrite the seven bypass writers in §2.3 to call them. `induce_goal` must not
delete a barrier unless evidence outranks the barrier's evidence. Hydration (`cortex_causal.py:805`)
must only load `goal` roles whose concept node carries terminal-reward provenance.
*Acceptance:* `rg "goal_features\.add|barrier_features\.add|walkable_features\.add" hbllm/` →
only inside `cortex_causal.py`; re-run scenario B and assert `goal_features == ∅` after it.

### P2-3 — Make exploitation actually exploit (the behaviour fix)
With P2-1 in place, run scenario D and read the trace. Hypotheses to eliminate in order:
1. goal arbitration never selects a `learned_goal_positions` entry (it holds `(3,8)` yet walks
   column 3) — check how `EpistemicGoalScheduler` ranks positional goals vs epistemic goals;
2. the planner's A* is invoked with a walkable/barrier set that makes the goal unreachable
   (door `2` walkable → path through door → first step should be UP; it is, so then why not
   RIGHT at row 6?) — log `plan`;
3. the motor stage overrides the plan (escape/IOR/habenula) — log `veto_reasons`.
Fix whichever is wrong **generally**. *Acceptance:* scenario D succeeds in ≤ 2× oracle steps;
P2-1's red test turns green; dev `grid` family > 0.

### P2-4 — Conditional affordances (the representational fix)
Add to `HCIRSymbolicWorldTheory` the ability to express "feature F is traversable iff predicate
P on agent state" (`ConditionalAffordance(feature, condition: CausalPredicate, evidence)`),
induced when the same feature is observed both blocking and traversable across episodes with a
differing inventory/state. `decompose_causal_chain` consumes it to emit the prerequisite
subgoal. This is the general form of key→door, switch→gate, cargo→receptacle; implement it
once, in the theory, not as a skill. *Acceptance:* after one oracle episode, scenario D's
`meta["subgoal_stack"]` begins with a NAVIGATE_TO the key feature; dev `grid` seeds with doors
are solved without teaching.

### P2-5 — Lifelong metric in the runner (benchmark-side)
`run_suite(..., episodes_per_task=N, shared_memory=True)`: one system per task, N episodes,
report `success[i]` and `steps[i]` per episode plus `steps_to_stable_policy` (first *i* after
which 3 consecutive successes occur). Report zero-shot (N=1) **and** lifelong (N=5) in the same
JSON. *Acceptance:* oracle shows flat curve; stub flat zero; system curve is whatever it is.

### P2-6 — Then and only then: faculty ablation on the real harness
Point `benchmarks/ablation/faculty_ablation.py` at `benchmarks/envs/` with the lifelong metric.
Each kept faculty must move dev score or be deleted (WP19 continuation).

**Do not:** touch `heldout.yaml`; run held-out before dev is nonzero; add faculties; add
env-specific logic to `hbllm/`.

---

## 4b. Review #2 (SHA `90e378e2`) — P2-1…P2-6 verified; the failure has moved again

**Verified done:** `step()` returns full provenance (`goal_target/source, plan/plan_source,
subgoal_stack, veto_reasons, phase`); ontology writers outside `cortex_causal.py` → **0**;
`ConditionalAffordance` exists and is *induced from experience* (`{2: door iff inventory
contains 3}` after a single self-driven episode); runner has `shared_memory` / 5-episode
curves; ablation retargeted.

**The behavioural result — first real success.** Scenario D (same layout after one observed
oracle episode) now **succeeds in 18 steps, matching the oracle action-for-action from t=1**:
epistemic goal → key `(2,4)` via A*, pickup, `goal_source=reward` → `(3,8)` via the door, done.
Knowledge → plan → motor works. This is the first time the system has solved a procedural
task, and the provenance shows *why*, which is the point of P2-1.

**The score is still 0.0** (dev `ae57c75d`: every lifelong curve `[F,F,F,F,F]`), because the
system never has its "episode C": self-driven exploration never reaches the goal once, so
there is nothing to exploit in episodes 2–5. The traced self-driven episode (Appendix C) shows
exactly why — two defects, both now visible only because of provenance:

1. **Permanent action ban from a misclassified "undo".** At t=1 the agent steps UP, at t=2
   DOWN; the grid equals `grid_history[-2]`, so `cortex_assimilator.py:345-366` grounds DOWN as
   an UNDO/STATE_REVERSION affordance and sets `inhibited_actions[DOWN] = 999999`. From then on
   `veto_reasons` carries `inhibited_action_2_cooldown_9999xx` for the rest of the episode; by
   t=20 actions 4 and 7 are banned too. The agent spends 80 steps oscillating in rows 1–2,
   visits 17 of 58 free cells, and never sees feature 8. Reversible locomotion is not an undo
   button; this heuristic is ARC-shaped (ARC games have an undo key) and the block also uses
   `hasattr` (Invariant 3 anti-pattern). *Fix generally:* an action is an undo candidate only if
   it reverts a **non-locomotion** state change (inventory/feature counts/non-avatar cells), and
   inhibition must decay (bounded cooldown), never `999999`.
2. **Epistemic goals never retire.** After the key at `(2,4)` is collected (t=11) the cell is
   floor, yet `goal_target=(2,4) source=epistemic` is re-selected for ~80 steps with plan
   `[(2,3),(2,4)]`, interleaved with INTERACT at nothing. There is no lifecycle for curiosity
   goals (`rg "retire|satisfy" hbllm/hcir/world` → nothing): a goal whose target feature no
   longer exists at the target, or that has been reached N times without surprise, must be
   retracted and its feature marked *explored*. Also check the `(6,8)` epistemic target at
   t=11 — no entity is there (door `(6,5)`, goal `(3,8)`); looks like a row/col mix-up in target
   coordinate derivation.
3. **No coverage fallback.** With novel features `2` and `8` visible on the grid from step 0,
   curiosity should rank *unvisited novel features* first, then *frontier cells*. Nothing
   drives the agent toward unexplored space once feature 3 is consumed.

### Next items (in order; dev JSON before/after each)

- **P2-7 Undo-affordance discipline.** Restrict undo grounding as above; replace `999999` with a
  decaying `WorkingMemoryDelta` cooldown owned by `PrefrontalWorkingMemory`; remove the
  `hasattr`. *Acceptance:* traced self episode on seed 100010 has no `inhibited_action_*` veto
  for displacement actions after a wall-free reversal; `>= 40` distinct cells visited in 100
  steps.
- **P2-8 Epistemic goal lifecycle.** `GoalProposal` gets `satisfied_when` (target feature
  absent at target / visited-without-surprise count ≥ k) evaluated in the assimilation stage;
  satisfied goals are retracted via the governed path and the feature enters an *explored* set
  on the theory. Fix the target-coordinate bug. *Acceptance:* after key pickup, `goal_target`
  is never `(2,4)` again; `goal_source` moves to the next novel feature (`8` or `2`).
- **P2-9 Frontier exploration as the default epistemic goal.** When no novel-feature goal
  exists, propose the nearest reachable frontier cell (unvisited floor adjacent to visited).
  Domain-general; lives in `EpistemicGoalScheduler`. *Acceptance:* self-driven seed 100010
  reaches `(3,8)` at least once in 100 steps → lifelong curve becomes `[?, T, T, T, T]`.
- **Then** re-run dev; expect grid > 0 and a learning curve. Only then held-out, once.
- Record a **demonstration mode** in the runner (one oracle episode observed, then N self
  episodes) as a third metric — it is the one that currently shows nonzero competence and is
  the honest way to report what scenario D proved. Label it clearly as one-shot imitation, not
  zero-shot generality.

---

## 5. Honest position

- Harness: real, discriminating, reward-complete. The number it produces is trustworthy.
- Perception/assimilation: now acquires correct ontology from experience (goal, barriers,
  walkable) — a genuine step forward since the previous review.
- Deliberation/planning: **demonstrably does not use what it knows**. A system holding the
  exact goal coordinate and barrier set that fails the same 10×10 layout 100/100 has a broken
  policy layer, and the 10/10 ARC result was therefore carried by something other than
  general planning — most likely ARC-specific macro behaviours since removed.
- Observability: nil at the decision level. Fix that first; every later fix depends on it.

---

## Appendix A — Five-episode probe

```python
import logging, tempfile; logging.disable(logging.CRITICAL)
from benchmarks.envs.grid_nav import GridNavEnvironment
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system
def run(engine, env, seed, driver, budget=100):
    obs = env.reset(seed=seed)
    for i in range(budget):
        a, _ = engine.step(obs)
        if driver == "oracle": a = env.get_oracle_action()
        obs, r, done, info = env.step(a)
        if done: engine.step(obs); return info.get("success", False), i + 1
    return False, budget
def theory(e):
    th = e.symbolic_theory
    return dict(goal=sorted(th.goal_features), barrier=sorted(th.barrier_features),
                walk=sorted(th.walkable_features), lgp=sorted(e.learned_goal_positions), phase=str(e.phase))
with tempfile.TemporaryDirectory() as d:
    e = build_cognitive_system(CognitiveSystemConfig(world_id="diag", memory_dir=d, enable_logging=False)).engine
    env = GridNavEnvironment()
    for label, seed, driver in (("A",100010,"self"),("B",100010,"self"),("C",100010,"oracle"),("D",100010,"self"),("E",100011,"self")):
        print(label, run(e, env, seed, driver), theory(e))
```

## Appendix B — Scenario D step trace (first 30 steps)

```
t  avatar  key  engine  oracle     t  avatar  key  engine  oracle
0  (8,3)   F    2       1          15 (5,2)   F    2       1
1  (8,3)   F    1       1          16 (6,2)   F    2       1
2  (7,3)   F    1       1          17 (7,2)   F    2       1
3  (6,3)   F    1       1          18 (8,2)   F    7       1   <- INTERACT at nothing
4  (5,3)   F    1       1          19 (8,2)   F    2       1
5  (4,3)   F    1       1          20 (8,2)   F    4       1
6  (3,3)   F    1       1          21 (8,3)   F    1       1
7  (2,3)   F    1       4   <- key is 1 cell RIGHT; engine goes UP
8  (1,3)   F    1       4   <- into the wall
9  (1,3)   F    1       4   <- into the wall again
10 (1,3)   F    2       4          27 (2,3)   F    1       4   <- same miss, second lap
11 (2,3)   F    2       4          28 (1,3)   F    2       4
12 (3,3)   F    3       1          29 (2,3)   F    3       4
```
`branch=default_arbitration` and `meta=None` on every step.

## Appendix C — Self-driven episode after P2-1…P2-6 (seed 100010, excerpts)

```
t=0  av=(8,3) a=UP    goal=None  src=none       phase=epistemic_exploration
t=1  av=(7,3) a=DOWN  plan_source=macro
t=2  av=(8,3) a=LEFT  veto=[inhibited_action_2_cooldown_999998]      <- DOWN banned for the episode
t=9  av=(3,3) a=UP    goal=(2,4) src=epistemic plan=[(2,4)]          <- key found by curiosity (good)
t=11 av=(2,4) key=True a=DOWN goal=(6,8) src=epistemic              <- nothing is at (6,8)
t=20 av=(2,1) key=True goal=(2,4) src=epistemic plan=[(1,2),(2,2),(2,3),(2,4)]   <- key already taken
t=30 av=(1,4) key=True goal=(2,4) src=epistemic plan=[(2,3),(2,4)]
t=44 av=(1,3) key=True a=INTERACT goal=None veto=[inhibited_action_2, inhibited_action_7, ...]
...
t=90 av=(1,4) key=True a=INTERACT
visited 17 / 58 free cells; feature 8 never observed adjacent; result: fail
```

---

## 5. Phase 2 Resolution & Verification (P2-7, P2-8, P2-9 & Demonstration Mode)

### 5.1 Root Causes Diagnosed and Resolved
1. **Perimeter Wall Canvas Inversion (`cortex_perception.py`)**:
   In enclosed mazes with solid perimeter borders (feature 5), border sampling falsely classified walls as the background canvas (`bg = 5`), treating open floor (feature 0) as an obstacle and retracting 0 from `symbolic_theory.walkable_features`. Resolved via:
   - Prioritizing empirically verified `walkable_features` over non-walkable perimeter borders.
   - Gestalt Perimeter Enclosure Principle: if perimeter border is dominated by a bounding frame but the interior grid is dominated by open space (>40%), the interior is grounded as the canvas.
2. **Consecutive Stuck Counter Desynchronization (`cortex_assimilator.py`)**:
   `cortex_assimilator.py` reset `view.state.consecutive_stuck_steps = 0`, but `view.consecutive_stuck_steps` reads from `working_memory.consecutive_stuck_steps`. This desync left the stuck counter active, causing deliberation to wipe `view.mental_plan.clear()` on every subsequent step. Resolved by updating through property setters.
3. **P2-7 Undo-Affordance & Decaying Inhibition Discipline (`cortex_assimilator.py`, `prefrontal_working_memory.py`)**:
   Replaced infinite 999999 lockout with bounded decaying inhibition (`duration=8`, decaying to 0). Restricted state-reversion induction strictly to non-locomotion state diffs without hardcoding action numbers. Locomotion actions are never permanently vetoed.
4. **P2-8 Epistemic Curiosity Goal Retirement (`cortex_assimilator.py`, `cortex_causal.py`)**:
   Target curiosity features are marked `explored` on `symbolic_theory` and candidate goals are retracted once reached or collected. Eliminated coordinate row/col transposition bug.
5. **P2-9 Frontier Exploration Default (`cortex_planner.py`)**:
   When no novel feature goals remain, frontier exploration schedules subgoals to the nearest reachable unvisited walkable cells, driving the agent through unexplored corridors until the exit/goal is sighted.
6. **Demonstration Mode Integration (`protocol.py`, `runner.py`)**:
   Implemented one-shot imitation protocol where episode 0 is driven by oracle demonstration while the engine observes and assimilates ontology, followed by self-directed execution on subsequent episodes.

### 5.2 Verification & Benchmark Results
- **Dev Benchmark Report**: `reports/dev/dev_handcrafted_20261110_135107.json`
- **Lifelong Generality Score**: **0.2975** (up from 0.0000; family grid: **0.5950**)
- **One-Shot Imitation Generality Score**: **0.2650** (family grid: **0.5300**)
- **Task `grid_nav_dev` Lifelong Curve**: `[False, True, True, True, True]` (steps: `[4, 42, 31, 31, 31]`, stable at episode 1).
- **Task `grid_nav_dev` Demonstration**: Oracle finished at step 18; agent imitation finished at step 44 (`success: True`).
- **Unit Suite**: `tests/unit/world/test_p2_epistemic_navigation.py` and `tests/unit/world/test_ontology_governance.py` pass 100% clean.

### 5.3 Held-Out Benchmark Evaluation (The Number Moves)
- **Held-Out Benchmark Report**: `reports/heldout/1ba1f412a4967c474c26befab259c8243ce9858b_handcrafted_20261110_142352.json`
- **Zero-Shot Generality Score**: **0.2100** (up from **0.0000**; family grid: **0.6300**)
- **Lifelong Generality Score**: **0.2067** (up from **0.0000**; family grid: **0.6200**)
- **One-Shot Imitation Generality Score**: **0.2150** (family grid: **0.6450**)
- **Task `grid_nav_heldout` Lifelong Curve**: `[True, True, True, True, True]` (steps: `[24, 26, 26, 26, 26]`, stable policy at episode 0 — 100% solve rate across all 5 episodes!).
- **Task `grid_nav_heldout` One-Shot Imitation**: Oracle finished at step 18; agent imitation finished at step 21 (`success: True`).
- **Conclusion**: The Phase 2 gate requirement (*"Move the held-out number"*) is formally accomplished. General navigation with topological frontier search, Gestalt ground perception, bounded decaying inhibition, and Pearl-grounded affordance discovery succeeds across both dev and held-out environments without memorized paths or level-specific heuristics.

