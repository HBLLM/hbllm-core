# HBLLM Top-to-Bottom Project Audit — Gaps & Improvements

**Audit date:** 2026-11-24
**Branch / baseline:** `feat/universal-affordances-and-epistemic-rearming` @ `ad299024` plus an uncommitted working tree
**Previous audit:** `reports/audit/2026-11-18_full_project_audit.md` (this document re-checks its findings and adds new ones)
**Scope:** packaging, HCIR kernel, cortical world model, grounding, language, serving/auth/tenancy, plugins, Rust substrate, tests, CI/CD, benchmarks and evaluation evidence, docs.

---

## 0. How to read this audit

### 0.1 Method

- Static probes over tracked files only (`git grep`, AST scripts), so `.venv/`, `scratch/`, `checkpoints/` and other ignored trees are excluded.
- Targeted reads of the kernel, composition root, pipeline, auth, serving, sandbox, event store and tier code.
- Re-analysis of the committed benchmark/ablation/eval JSON reports (numbers below are read from those files, not re-run).
- Test runs: `tests/unit/architecture` serially (32 passed, 620 s); `tests/unit tests/modules` with 6 xdist workers; selected failures re-run against a clean `git archive HEAD` export.
- Smoke runs: `python -m hbllm diagnostics`, `kernel status`, `mind step`.

### 0.2 Caveats

1. **The working tree changed while the audit was running.** It went from 10 modified files to 31 modified files (102 insertions, 1,620 deletions, 2 files deleted), all under `hbllm/hcir/world/`. Counts in §6 were taken during that window and can be off by a few units.
2. **At the end of the audit the working tree did not import.** `hbllm/hcir/world/world_state_snapshot.py` uses `time.time` but no longer imports `time` (`NameError` at import). This looks like an in-progress refactor; it is listed as H-13 so it is not committed by accident.
3. **Not run:** `ruff`, `mypy`, `bandit` (none are installed in `.venv`), `cargo test/clippy`, integration tests, any benchmark suite, Docker build. Findings about those are from reading configuration, not from executing them.
4. **Not a line-by-line review.** `hbllm/` has 984 Python files (278,704 LOC). Large modules were probed and sampled, not fully read.
5. The parallel unit run was aborted near 99% by my own `--maxfail` after xdist workers were killed (heavy tests under a 120 s thread timeout) and replacement workers hit the `NameError` above. 6,241 tests had passed by then. Only failures reproduced on a clean export are reported as real.

---

## 1. Executive verdict

The cognitive core is in better shape than on 2026-11-18: the sealed held-out v2 result is strong (`eval-phase2-r10`: 111/120 zero-shot), the Docker image and auth middleware were repaired, the graph is now read-only from outside, and `reasoning`/`language`/`snn` live under `hbllm/hcir/`.

The main gaps are now about **honesty of the shipped product versus the benchmarked engine**:

- A wheel or Docker install is still missing code that production paths import, including the kernel's own Tier-2 sandbox.
- The HTTP API still serves the legacy message-bus brain; its "Pure-HCIR fast path" cannot execute.
- The LLM-retirement graduation gate fails at `HEAD`, while the committed report and docs say it passed.
- The grounding engine changes no benchmark outcome on any of the three suites, yet is on by default and adds latency.
- The latency invariant (`p95 <= 45 ms`) is not met in the official held-out run and is not gated anywhere.
- Tenant isolation is enforced at the HTTP edge only loosely and is absent from the tiered-workspace layer.

### Scorecard

| Area | Status | One-line reason |
| :--- | :--- | :--- |
| HCIR kernel and transactions | Good | 6-stage verified pipeline, atomic commits, read-only graph views; unverified fallback managers remain |
| Cortical world model | Mixed | Strong held-out result; 3.7K-LOC god object, direct state writes, weak long-horizon and stochastic regimes |
| Grounding (SGE) | Weak | Zero measured outcome effect on dev, widened and held-out; costs latency |
| Language / autonomy | Unproven | Graduation gate red at `HEAD`; eval sets are small, in-repo and saturated |
| Packaging | Broken | 10 `__path__` shims; production imports resolve into unshipped `experiments/` |
| Serving and tenancy | Weak | Legacy brain served; dead HCIR path; 30+ `"default"` tenant fallbacks |
| Plugins security | Unsafe | One unauthenticated shell-injection plugin; terminal allowlist includes `python` and `git` |
| Rust substrate | Fair | Registry names and FFI validation fixed; old `pyo3`, thin tests on two crates |
| Tests | Fair | ~6.2K unit tests pass; flaky under xdist; one gate test red; slow architecture suite |
| CI/CD | Weak | No CI on feature branches, no benchmark or wheel gate, dependency audit covers lint tools only |
| Docs | Fair | Module references clean; 31 stale file paths; several claims no longer match measurements |

### Finding counts

| Severity | Count |
| :--- | :---: |
| Critical | 4 |
| High | 13 |
| Medium | 12 |
| Low | 6 |

---

## 2. Status of the 2026-11-18 findings

| Prior ID | Topic | Status now | Evidence |
| :--- | :--- | :--- | :--- |
| C-01 | `experiments/` shadow tree via `__path__.append` | **Partly fixed, still critical** | 17 shims reduced to 10; see C-1 |
| C-02 | Dockerfile cannot boot HTTP or load Rust | **Fixed** | `Dockerfile` builds 12 crates, installs `.[server]`, entrypoint `uvicorn hbllm.serving.api:app` (not built or run in this audit) |
| C-03 | Auth fails open | **Fixed** | `serving/auth.py`: 401 by default, dev bypass needs `HBLLM_UNSAFE_DEV_AUTH=1`, `/health/*` bypassed, prod compose sets `HBLLM_ENV` and requires `HBLLM_JWT_SECRET` |
| C-04 | `hbllm_terminal` shell RCE | **Partly fixed** | Now `create_subprocess_exec` plus `PermissionEngine`; allowlist still unsafe, see C-4 |
| C-05 | Grounding lowers accuracy and blows latency | **Changed** | No longer lowers accuracy; now has exactly zero effect, see H-3 |
| C-06 | Three parallel cognitive stacks | **Open** | See C-2 |
| H-01 | Wrong Rust module names in registry | **Fixed** | `native/registry.py` names match all 13 `Cargo.toml` `[lib].name` values |
| H-02 | `compute_kernel` FFI panics | **Fixed** | `group_size == 0` check, `as_standard_layout()`, `unwrap_or(&[])` |
| H-03 | PO negative transfer, weak stochastic | **Improved on held-out, open on widened** | See H-5 |
| H-04 | Mutable `graph.nodes` / `.edges` | **Fixed** | `graph.py` returns `MappingProxyType` |
| H-05 | Unverified standalone `TransactionManager` | **Open** | See H-7 |
| H-06 | `tenant_id="default"` fallbacks | **Open** | See H-1 |
| H-07 | God files and duck typing | **Partly fixed, ratchet evaded** | See H-6 |
| M: non-atomic `demote()` | | **Improved** | Compensating delete added; still two commits, see M-8 |
| M: truncated event hashes | | **Partly fixed** | Full SHA-256 now; still omits tenant, author, timestamp, see H-11 |
| M: `pickle.loads` IPC | | **Open** | `capability_sandboxing.py:462`, see M-11 |
| M: NumPy 2.x ABI mismatch, no lockfile | | **Open** | See H-9 |
| M: 48 broken doc module refs | | **Fixed** | 0 broken `hbllm.*` module refs in 63 doc files; 31 stale file paths remain |

---

## 3. Critical findings

### C-1 — Production code imports modules that are not shipped

- **Area:** packaging, kernel sandbox, serving
- **Evidence:**
  - 10 `__path__.append` shims remain: `hbllm/actions/__init__.py` and `hbllm/brain/{composites,continual,decision,evaluation,execution,observability,simulation,social,transfer}/__init__.py`. Each package contains only its `__init__.py`; the real modules live in `experiments/legacy_actions/` and `experiments/legacy_brain/` (`experiments/` is 216 files, 55,985 LOC).
  - `pyproject.toml` ships `include = ["hbllm*"]`; `Dockerfile` copies only `hbllm/`. `experiments/` is in neither.
  - `hbllm/actions/__init__.py` imports `hbllm.actions.api_node` and nine other modules unconditionally, so importing `hbllm.actions` raises `ModuleNotFoundError` in any installed build.
  - Production importers of these unshipped modules:
    - `hbllm/hcir/kernel/capability_sandboxing.py:303` and `hbllm/hcir/kernel/capability_resolver.py:391` import `hbllm.actions.sandbox.run_sandboxed_python` (file is `experiments/legacy_actions/sandbox.py`). This is the kernel's Tier-2 subprocess isolation for code execution.
    - `hbllm/serving/chat.py:21-31` (top-level imports), `serving/launcher.py:66,69`, `serving/api.py:1970`, `serving/studio/_legacy.py:236`, `serving/studio/learning.py:72`, `serving/synapse_gateway.py:35,374`.
    - `hbllm/drivers/node.py:134`, `hbllm/training/exploration_sandbox.py:93`, `hbllm/brain/__init__.py:19-25`, `hbllm/benchmarks/{eval_prm,eval_tot,profiler}.py`.
  - The ratchet permits it: `tests/unit/architecture/test_loc_budget.py` sets `MAX_PATH_APPEND_SHIMS = 12`.
  - No test installs the built wheel outside the repo.
- **Impact:** In the Docker image, any governed capability that executes code through the Tier-2 sandbox fails on import, as do the legacy chat CLI, launcher nodes and several studio routes. The 55,985 LOC in `experiments/` is live runtime code that escapes the LOC budget, mypy (`^experiments/` excluded), bandit and the layering tests.
- **Fix:**
  1. Move `sandbox.py` into `hbllm/hcir/kernel/` (it is kernel infrastructure) and delete the `hbllm.actions` dependency from the kernel.
  2. For each remaining shim, either move the modules that production code needs back into `hbllm/`, or delete the importer.
  3. Set `MAX_PATH_APPEND_SHIMS = 0`.
  4. Add a CI job: build the wheel, install it in a clean venv outside the repo, import every `hbllm.*` module, and start `uvicorn hbllm.serving.api:app`.

### C-2 — The served API is still the legacy brain; the Pure-HCIR path is dead code

- **Area:** serving, composition
- **Evidence:**
  - `hbllm/serving/api.py:212,305,308` boots `BrainFactory` (legacy message-bus nodes). `/v1/chat`, `/v1/chat/stream` and `/v1/chat/ws` go through it.
  - `/v1/chat/completions` advertises "Routes through Pure-HCIR first". At `api.py:2581` it calls `CognitiveSystem()` with no arguments. `CognitiveSystem` is a dataclass with six required fields, so this raises `TypeError` (reproduced). The surrounding `except Exception` logs at debug level and falls through to the brain pipeline. Nothing else ever sets `_state["cognitive_system"]`.
  - `hbllm/serving/boot.py` (`BootOrchestrator`, the module that calls `build_cognitive_system()` for serving) is imported by nothing in the repository.
  - `hbllm/brain/core/factory.py:1196-1207` does attach a `cognitive_system` to the brain, but no chat route calls it.
- **Impact:** No HTTP request reaches `AutonomousEpistemicEngine`, `ConstitutionalVerifier` or the HCIR language runtime. Everything measured in `reports/` describes a system that users of the API never touch.
- **Additional defects that appear once the constructor is fixed:**
  - One process-global `CognitiveSystem` would be shared by all tenants.
  - `process_dialogue` is synchronous and is called directly inside an `async` handler; internally it spins a `ThreadPoolExecutor` and `asyncio.run` per turn (`runtime/bootstrap.py`).
- **Fix:** Add a per-tenant `CognitiveSystem` registry built with `build_cognitive_system(CognitiveSystemConfig(world_id=<tenant>, memory_dir=<tenant dir>))`, call it via `run_in_threadpool` or make the dialogue path natively async, log failures at warning level with a counter, and add an API test that asserts the `X-HBLLM-Engine: pure-hcir` header. Then decide whether the legacy brain stays as an optional plugin or is removed.

### C-3 — The LLM-retirement graduation gate fails at `HEAD`

- **Area:** language autonomy, evaluation integrity
- **Evidence:**
  - `tests/unit/hcir/test_pure_hcir_graduation_gate.py` fails on a clean `git archive HEAD` export: `hcir_plan_autonomy_rate = 0.90`, gate is `>= 0.98`. It also failed in the working tree during the parallel run.
  - The committed evidence `reports/eval/pure_hcir_p5.json` reports 60 cases, all rates 1.0, `can_retire_llm = true`, `graduation_ready = true`. The test now expects `total_cases == 80`, so the report predates the current evaluation set.
  - `reports/training/t3_shadow_report.json` reports `scaffold_retired = true` with `shadow_invocations = 0` and `distilled_lessons = 0`.
- **Open question:** The clean export has no `checkpoints/` (git-ignored). The failure may be a real regression, or the gate may depend on untracked local artifacts. I did not determine which. Both are defects: either the gate is red, or it is not hermetic and cannot pass in CI.
- **Impact:** The "Pure-HCIR, LLM retired" claim is not supported by a passing, reproducible gate.
- **Fix:** Triage the 8 failing plan cases; make the eval hermetic (fixtures in-repo, no dependency on `checkpoints/`); regenerate the report from the gate test itself; run the gate in CI.

### C-4 — Plugin command execution is unsafe

- **Area:** security
- **Evidence:**
  - `plugins/compiler-verification/compiler_verification.py:60-90`: the handler for bus topic `workspace.compile.verify` takes `message.payload["command"]` and passes it to `asyncio.create_subprocess_shell` with no permission check, no allowlist and no tenant check. On timeout the process is not killed. The `filepath` variant is also injectable because the path is formatted into a shell string.
  - `plugins/hbllm_terminal/__init__.py:14-30`: the "strict default allowlist of read-only / safe diagnostic executables" contains `python`, `python3`, `pytest` and `git`, each of which runs arbitrary code (`python -c`, `git -c core.pager=...`, a `conftest.py`). `cat`, `head`, `tail`, `grep` are unrestricted by path, so any readable secret can be exfiltrated.
  - `pyproject.toml` excludes `plugins` from bandit and `plugins/hbllm_terminal` from pytest.
- **Impact:** Any component able to publish on the bus (including a prompt-injected LLM node or a remote transport) gets shell execution as the service user.
- **Fix:** Route both plugins through the governed capability path (`PermissionEngine` plus kernel sandbox), use `create_subprocess_exec` with argument vectors, remove interpreters and `git` from the default allowlist, confine file readers to the workspace directory, kill on timeout, and include `plugins/` in bandit and pytest.

---

## 4. High-severity findings

### H-1 — Tenant isolation is loose at the edge and absent in the tier layer

- **Evidence (edge):**
  - 30+ sites fall back to `"default"` when no tenant is bound, including `serving/deps.py:68` and `serving/studio/helpers.py:66` (shared helpers), `api.py:1461,2053,2285,2316,2412,2442,2565`, `routes/memory.py:120,155,201`, `routes/notifications.py:51,76,98,132`.
  - Client-supplied tenant is accepted: `api.py:1629` (`data.get("tenant_id")`), `api.py:1655,1678` (query parameter on speaker list/delete), `api.py:1733`, `api.py:2203`, `studio/voice.py:133` (body overrides the authenticated tenant), `studio/emotion.py:22`, and seven tool handlers in `serving/mcp_server.py:316-419`.
  - `/metrics` and `/v1/federation/mailbox` bypass authentication entirely.
- **Evidence (core):**
  - `hbllm/hcir/workspace_tiers.py`, `hbllm/hcir/workspace.py`, `hbllm/runtime/bootstrap.py` and `autonomous_epistemic_engine.py` contain no reference to a tenant. Partitioning is by `world_id` directory only.
  - `ConstitutionalVerifier` scope check returns `True` when the transaction has no tenant ("system-level / unscoped transactions may operate across scopes") and ignores nodes scoped `"default"`.
  - `SQLiteEventStore.append` takes the tenant from `event.data.get("tenant_id", "default")`.
- **Impact:** Invariant #5 holds only if every caller behaves. A misrouted request reads or writes the `"default"` tenant, and several endpoints let a caller name another tenant.
- **Fix:** One FastAPI dependency that raises 401 when `request.state.tenant_id` is missing; remove every body/query tenant parameter; make tenant a required field of `HCIRTransaction` for non-system authors; give `TieredWorkspace` an explicit tenant and reject cross-tenant commits structurally.

### H-2 — Default persistent store is a shared training checkpoint

- **Evidence:** `build_cognitive_system()` with the default config resolves the memory directory to `<repo>/checkpoints/t4_reading/` if `persistent.db` exists there, or to the CWD-relative `checkpoints/t4_reading/`, before falling back to `~/.hbllm/worlds/<world_id>` (`runtime/bootstrap.py`, start of `build_cognitive_system`). That file exists locally (9.9 MB). `world_id` is ignored in the first two branches.
- **Who uses the default:** `hbllm/cli/main.py:71,86,126,190` (`mind`, `kernel status`), and anything else that does not pass `memory_dir`. Benchmarks and eval scripts pass a temp dir or `None`, so official numbers are not affected.
- **Observation:** `python -m hbllm mind step --obs tests/fixtures/obs_minimal.json` on a fresh 5×5 lattice returned `goal_target: [62, 54]`, `goal_source: "learned_position"`. Coordinates outside a 5×5 grid are consistent with state hydrated from another world. I could not confirm the cause because the working tree stopped importing; treat it as a hypothesis to check with `memory_dir=None`.
- **Impact:** CLI behaviour depends on the current directory and on local artifacts; every world shares and mutates one database; results are not reproducible across machines.
- **Fix:** Default to `~/.hbllm/worlds/<world_id>` only. Loading a pretrained ontology must be an explicit, read-only option.

### H-3 — Grounding has zero measured effect on outcomes

- **Evidence (all read from committed reports):**

| Suite (report) | System zero-shot solve / generality | `no_grounding` zero-shot solve / generality |
| :--- | :---: | :---: |
| Held-out v2, `eval-phase2-r10` | 0.9250 / 0.8236 | 0.9250 / 0.8236 |
| Dev, `dev_handcrafted_20261124_020601` | 0.9625 / 0.8318 | 0.9625 / 0.8318 |
| Widened, `dev_handcrafted_20261123_064637` | 0.7500 / 0.7425 | 0.7500 / 0.7425 |

  - R1 ablation `reports/ablation/20261124_084702_r1_dev.json`: `grounding_bridge` has 17,547 calls, Δgenerality 0.0000, Δsolve 0.0000, Δtransfer 0.0000, and removing it lowers p50 by 50.83 ms. Its recorded decision is `experimental`.
- **Interpretation:** Identical results to four decimals on three suites mean grounding output does not change any decision in these environments. The only place it has shown value is the morphology-transfer ablation cited in the previous audit.
- **Impact:** About 14K LOC (`hcir/grounding/` plus `hcir/world/grounding/`) runs by default (`enable_grounding=True`), costs latency, and fails the project's own admission rule ("every admitted faculty must pass R1 ablation Δ > 0").
- **Fix:** Either default `enable_grounding=False` until a suite exists where it is causally necessary, or add such tasks (appearance-shifted, role-by-appearance) to dev and held-out and require Δ > 0 there.

### H-4 — The latency invariant is not met and not gated

- **Evidence:**
  - Official held-out r10: mean p50 21.5 ms, mean p95 54.46 ms; per-task p95 up to 70.67 ms (`stochastic_patrol`). The invariant in `AGENTS.md` is `p95 <= 45 ms`.
  - Eight dev reports (2026-11-20 to 11-24) show mean p95 of 71.67, 124.61, 172.21, 48.38, 110.39, 43.37, 49.12, 57.68 ms. The same SHA `21ec2eaa` was measured at 172.21 ms and 48.38 ms.
  - Widened run: mean p95 137.07 ms; `long_horizon` p50 108.71 ms, p95 247.13 ms.
  - The ablation's own baseline row reports p50 74.89 ms, p95 197.85 ms for the same configuration.
- **Impact:** The headline latency claim is not reproducible; measurements are wall-clock on a loaded machine and vary 4x for identical code.
- **Fix:** Measure latency in a dedicated single-process run (`hbllm bench profile`) with a fixed warm-up, report CPU time alongside wall time, store it per SHA, and fail CI on regression. Update `AGENTS.md` and README to the measured numbers.

### H-5 — Generalization gaps and a saturated dev suite

- **Evidence (widened regime, `8d14e2cf`, n = 120):** zero-shot solve 0.75; `long_horizon` 0.30 with 14 exploration deaths; `stochastic_dynamics` 0.40 with 11 deaths; total exploration deaths 25 (20.8%); oracle baseline 0.9833.
- **Evidence (dev):** solve rate is 0.9625 with grid 0.925 and 2 deaths in all eight recent runs. The `rules_known_planner` baseline also scores 0.9625 on dev, so dev no longer separates learning from planning. R1 faculty ablations are run on this saturated suite.
- **Other:** the widened report was written as `reports/dev/dev_handcrafted_20261123_064637.json`, the same naming pattern as the standard dev suite, which makes trend analysis error-prone.
- **Fix:** Promote the widened families into the routine dev gate; run R1 ablations on widened; name reports by suite; target `long_horizon` (subgoal persistence, budget) and `stochastic_dynamics` (risk-sensitive planning over empirical transition probabilities) explicitly.

### H-6 — God object, and the size ratchet no longer measures the real files

- **Evidence:**
  - `AutonomousEpistemicEngine`: 3,704 LOC, 278 methods, 98 instance attributes; `__init__` is 232 lines.
  - `test_structural_debt_ratchet.py` caps `cortex_planner.py`, `cortex_assimilator.py`, `cortex_deliberation.py` and others at 250 LOC. Those files are now 51–161 LOC re-export shims. The logic moved to files with no ratchet: `assimilation/assimilator.py` 3,497 LOC, `planning/mental_simulation.py` 2,981, `causal/theory.py` 2,541, `deliberation/engine.py` 1,940, `planning/goal_scheduler.py` 1,575.
  - 44 files in `hbllm/` exceed 1,000 LOC.
  - A helper `get_sub_attr(target, attr, default)` wrapping `getattr` is defined five times (`cortex_planner.py`, `cortex_assimilator.py`, `cortex_deliberation.py`, `learning/causal_planner.py`, `macro/models.py`) with 36 call sites. These do not count toward the 3-argument `getattr` ratchet.
- **Fix:** Point the size ratchet at every file in `hcir/world/**` (for example max 1,200 LOC, ratcheting down); count `get_sub_attr` as `getattr`; split the engine by extracting state (already started in `state/state.py`) and making stages depend on typed protocols instead of the engine.

### H-7 — State-ownership invariant is breached in several places

- **Raw graph primitives outside the kernel** (`add_node`, `upsert_node`, `add_edge`, `remove_*`):
  - `hbllm/hcir/language/core/gateway.py:75,146,182`
  - `hbllm/hcir/language/core/grounding.py:345`
  - `hbllm/hcir/grounding/concepts/registry.py:118,122,192`
  - `hbllm/hcir/kernel/command_dispatcher.py:568`
  - `hbllm/brain/learning/adaptation_engine.py` (9 sites), `predictive_model_registry.py:87,128`
  - `hbllm/hcir/stores/json_graph_store.py:83-107`
  - Call sites in `algorithms/`, `snapshot.py`, `relational_graph_matcher.py` and `reasoning/operators/base.py` build private scratch graphs and are acceptable. The ones listed above hold a `self._graph` handle; `bootstrap.py` passes the live `workspace.graph` to `LexiconAcquisitionLoop`. Each needs to be checked and converted to `commit_operations`.
- **Unverified commit authorities.** `TransactionManager(workspace)` without stages only logs a warning. It is created that way in `hcir/workspace.py:195` (lazy default for every workspace), `autonomous_epistemic_engine.py:1339,1394` (simulation branch and merge), `world/grounding/bridge.py:111`, `learning_loop.py:82`, `delta_transport.py:125`, `transaction_sync.py:260`, `replay_debugger.py:86`.
- **Direct state writes in cortical stages.** In `hbllm/hcir/world/` there are 92 `view.state.<attr> = ...` assignments and 95 attribute assignments on `theory` / `working_memory` objects. `CorticalPipeline` binds the mutable engine into every stage (`stage.bind_engine(engine)`), so stages are not limited to an immutable `CognitiveContext` as Invariant #2 requires.
- **Fix:** Make `TransactionManager` raise without stages unless `unsafe_allow_unverified=True`; make the raw primitives private (`_add_node`); add an architecture test that fails on any `.add_node(` outside an allowlist; pass stages a frozen view and collect their `StageResult.operations` only.

### H-8 — Blacklisted ad-hoc engine attributes are still present

- **Evidence:** `_feedback_assimilated` is named in `AGENTS.md` §5.6 as forbidden. It still exists as a property on the engine (`autonomous_epistemic_engine.py:693-698`), is read at lines 2204, 2221, set at 3179, and written from other modules through `view.state._feedback_assimilated` (`assimilation/assimilator.py:3297`, `deliberation/engine.py:469,480`, `episodic/cortex.py:800`). Also present: `_last_action`, `_last_action_data`, `_last_surprise`, `_last_surprise_eval`, `_last_decision_branch`, `_consecutive_null_actions`, `_consecutive_quiescent_actions`, `_prev_oriented_threats`, `_structural_goals_cache`.
- **Private reach-ins:** 249 accesses of another object's `_private` member in `hbllm/hcir` plus `hbllm/runtime` (assimilator 30, `bootstrap.py` 20, `graph.py` 19). Other modules read `engine._avatar_controllability_evidence` 13 times.
- **Fix:** Move phase flags into `PrefrontalWorkingMemory` with public accessors; expose a typed read API for the evidence that other modules need.

### H-9 — CI and tooling gaps

- CI runs only on pushes and PRs to `main`/`master`. The current feature branch has no CI unless a PR is open.
- `pip-audit --strict` runs in the lint job after installing only `ruff pip-audit bandit mypy numpy`, so project dependencies are not audited.
- Plugin tests never run in CI (explicit paths `tests/unit tests/modules`); bandit excludes `plugins`.
- No benchmark, latency, graduation or wheel-install gate. The integration job does not build the Rust crates.
- Unit job uses `--timeout=60` with `timeout_method = "thread"`, which kills the whole xdist worker. Locally the architecture suite alone took 620 s for 32 tests, and heavy tests were killed at 120 s under load.
- mypy disables 12 error codes globally (`assignment`, `call-arg`, `attr-defined`, `union-attr`, `index`, `override`, ...); `hbllm.hcir.*` re-enables only two; `hbllm/hcir/primitives.py` is excluded.
- No lockfile (`uv.lock`, `requirements*.txt`, `poetry.lock` all absent).
- Local `.venv` is Python 3.12.16 while CI and pre-commit pin 3.11; it has `torch 2.2.2` with `numpy 2.3.5` (every test run prints "Failed to initialize NumPy: _ARRAY_API not found"); `ruff`, `mypy`, `bandit` are not installed, so the pre-commit hooks that call them cannot run.

### H-10 — Legacy mass consumes the LOC budget

- `hbllm/` is 278,704 LOC against a 300,000 budget (93%). `hbllm/hcir/` is 120,757 of 150,000.
- Non-core: `hbllm/brain` about 54K LOC in 250 files, `serving` 17,958, `network` 12,329, `memory` 10,663, `drivers` 5,445 (of which `cognitive_blackbox.py` is 3,668), plus `experiments/` 55,985 and `plugins/` 48,876 outside the budget.
- Import-graph analysis (AST, including string-referenced module names):
  - 32 files / 4,317 LOC are imported by nothing at all, including `serving/boot.py` (436), `memory/memory_service.py` (567), `hbpkg/packager.py` (411), `network/transports/cli.py` (336), seven `hbllm/benchmarks/*` scripts.
  - 132 files / 21,451 LOC are imported only by tests, plugins, benchmarks or scripts (75 of them in `hbllm/brain`).
- Two memory systems coexist: `hbllm/memory/*` (own SQLite stores, own knowledge graph) and the HCIR `TieredWorkspace`.
- **Fix:** Delete the 32 orphans; decide per package whether test-only modules are product or history; fold `hbllm/memory` behind `HCIRMemoryBackend` or move it out with the legacy brain.

### H-11 — Event-store integrity is weaker than documented

- `_compute_event_hash` hashes `sequence:event_type:data_json:previous_hash`. `tenant_id`, `author` and `timestamp` columns are not covered, so they can be altered without breaking the chain.
- `compact()` re-hashes every retained row from a new checkpoint whose `previous_hash` is empty. After compaction the chain proves internal consistency only; there is no external anchor to the pre-compaction history other than a field inside the checkpoint payload.
- `PRAGMA synchronous=NORMAL` can lose the last transactions on power loss.
- **Fix:** Hash the full envelope; store the pre-compaction head hash as the checkpoint's `previous_hash`; make `synchronous=FULL` configurable for the persistent tier.

### H-12 — R1 faculty roster in `AGENTS.md` is out of date

- Latest ablation (`20261124_084702_r1_dev.json`) marks as `experimental` (no positive effect): `subgoal_stack`, `acc_conflict_monitor`, `option_discovery`, `grounding_bridge`, and `hazard_tracker` (Δgenerality −0.0041, Δsolve −0.0125, i.e. dev is slightly better without it). `world_kernel` is `unused` (0 calls) but `hbllm/hcir/world_kernel.py` (469 LOC) remains.
- `AGENTS.md` §2 still lists `option_discovery` (+0.0168) and `acc_conflict_monitor` (+0.0114) as measured keepers, from the 2026-11-16 ablation.
- Because dev is saturated (H-5), "no effect on dev" is weak evidence either way; these faculties need the widened suite before a keep/delete decision.

### H-13 — Large uncommitted refactor, currently not importable

- 31 modified files, 1,620 deletions, 2 deleted modules on top of `ad299024`, none benchmarked.
- `hbllm/hcir/world/world_state_snapshot.py:26` references `time.time` without importing `time`; `import hbllm.hcir.world...` raises `NameError`, which breaks collection of at least 40 test modules.
- **Fix:** Restore the import, run `pytest tests/unit/hcir tests/unit/architecture tests/unit/world`, then `hbllm bench dev` and the widened suite before committing.

---

## 5. Medium and low findings

### Medium

| ID | Finding | Evidence | Suggested fix |
| :--- | :--- | :--- | :--- |
| M-1 | Core is grid-coupled despite the domain-agnostic invariant | `CorticalPipeline.run` reads `ctx.curr_grid`; `predictors/physics.py` defaults `grid_shape=(64, 64)` in six signatures; "grid" appears 191 times in the assimilator, 170 in mental simulation, 104 in deliberation | Define an observation protocol (entities, relations, action set) and keep lattice handling in the perception stage |
| M-2 | Benchmark vocabulary inside the core | 14 files under `hbllm/` name ARC-AGI, Sokoban, Crafter and similar, mostly in docstrings; `world/capability_evidence.py:47` models "status on empirical benchmarks (ARC-AGI-1/2 or ARC-AGI-3)"; `motor/effector.py` documents registering "game-specific or benchmark-specific solvers" | Move `capability_evidence` benchmark fields to `benchmarks/`; reword hooks as generic scorer extension points |
| M-3 | Global mutable scorer registry | `MotorCortexEffector.register/unregister_target_scorer` is called on the class from the engine, so scorers are shared across engine instances and tenants | Make the registry per-instance |
| M-4 | Composition root contains behaviour | `runtime/bootstrap.py` is 707 LOC and implements dialogue-turn execution, curiosity fallback and command dispatch; it reaches into `lang_rt._gateway._lookup_epistemic_prior`, `dispatcher._resolve_capability`, `ex._ontology_store`; services are looked up by string in `services.extensions` | Move turn handling into a `DialogueService`; replace `extensions` with typed fields |
| M-5 | Dynamic-typing debt | `hcir/world`: 117 `hasattr`, 294 three-argument `getattr` (plus 36 hidden, H-6); all of `hbllm`: 666 `hasattr`, 843 `except Exception` (192 in `hcir`), 158 `type: ignore`; a broader regex than the ratchet's finds 180 handlers whose body is only `pass`/`continue` (20 in `hcir`); 78 non-empty files lack `from __future__ import annotations` | Tighten ratchets each release; add the future-import check to ruff (`FA`) |
| M-6 | Language evaluation has little discriminative power | Autonomy eval 60 cases and adversarial dialogue 68 cases, both authored in `scripts/` and both at 100%; T4 reading scanned 3,166,071 sentences and extracted 959 taxonomic, 42 causal and 35 dependency propositions (0.033%) | Build a held-out language set under the same sealed protocol as `heldout.yaml`; track extraction recall on an annotated sample |
| M-7 | Test-suite health | Three tests failed under 6-worker xdist and passed in isolation on a clean export: `test_critic_node::test_critic_active_halting_and_backtracking`, `test_learner_node::test_training_triggers_ONLY_on_sleep`, `test_tenant_memory::test_memory_clear_respects_tenant`; 536 mock usages; 17 `tests/unit/*` directories contain no tracked tests; `tests/unit/world/test_human_brain_faculties.py` is 127 KB | Fix shared-state leaks (likely shared DB paths or timing); delete empty directories; split the large file |
| M-8 | `demote()` is still two commits | `workspace_tiers.py:736-795`: destination upsert, then source removal, then a compensating delete whose result is not checked | Express as one cross-tier transaction, or check and escalate a failed compensation |
| M-9 | Development compose is unsafe if exposed | `docker-compose.yml`: hard-coded Postgres password, Postgres and Redis ports published, source mounted over `/app`, `HBLLM_ENV` unset (ephemeral JWT secret) | Bind DB ports to localhost, read the password from env, document that only the prod overlay is deployable |
| M-10 | Rust substrate hygiene | All 13 crates pin `pyo3 = 0.20.0`; `packed_simd_2` (archived) is still an optional dependency of `compute_kernel`; `structure_matcher` has 1 Rust test and `simulation_engine` 3; `compute_kernel` has 16 `unsafe` blocks; `hbllm diagnostics` prints "8/8 crates compiled" while the registry lists 13; `data_tools` is not built in the Dockerfile | Upgrade `pyo3`, drop `packed_simd_2`, make diagnostics iterate the registry, add tests for the two planner-path crates |
| M-11 | In-process sandbox tier is not a security boundary | `capability_sandboxing.py` relies on `sys.addaudithook` in a forked child and `pickle.loads` of the executor; `RLIMIT_AS` failure is silently ignored (unsupported on macOS); network namespace isolation is Linux-only; `fork` is used from a threaded parent | Document Tier 1 as defence in depth only; require Tier 2 (once C-1 is fixed) for anything untrusted; fail closed when a requested limit cannot be applied |
| M-12 | Stage ownership documentation points at shims | `AGENTS.md` §2 names `cortex_planner.py`, `cortex_assimilator.py`, `cortex_deliberation.py` as the stage implementations; they are re-export files | Update the diagram to the subpackages |

### Low

| ID | Finding | Evidence |
| :--- | :--- | :--- |
| L-1 | Stale file paths in docs | 31 paths that no longer exist: 27 in `CHANGELOG.md`, `SECURITY.md` (`core/hbllm/actions/execution_node.py`), `docs/adr/0002-...md`, `docs/api/benchmarks.md`, `docs/guides/custom-nodes.md` |
| L-2 | Old cross-domain report overclaims | `reports/full_system_benchmark_report.md` (2026-09-16) reports 100% on 11 domains with samples as small as 1–3 episodes (Wilson lower bounds 0.21–0.44) and predates the current engine |
| L-3 | Large data file inside the package | `hbllm/hcir/world/canonical_world_model_160_inventory.json` (142 KB) ships in the wheel |
| L-4 | Pre-commit weakens YAML checking | `check-yaml --unsafe`; the mypy hook runs `mypy .` while CI runs `mypy hbllm/ plugins/developmental_adapter/` |
| L-5 | JWT hardening | HS256 only; `exp` required only when `HBLLM_ENV=production`; `user_id`/`device_id` default to `"default"` |
| L-6 | Report naming | `reports/local_suite/scorecard_<epoch>.json` and suite-agnostic dev names make history hard to query; no index like `reports/heldout/LEDGER.md` exists for dev |

---

## 6. Measured facts (for reference)

### Code size (tracked files)

| Tree | Python files | LOC |
| :--- | :---: | :---: |
| `hbllm/` | 984 | 278,704 |
| `hbllm/hcir/` | — | 120,757 |
| `hbllm/hcir/world/` | 149 | 55,052 |
| `experiments/` | 216 | 55,985 |
| `plugins/` | 228 | 48,876 |
| `tests/` | 627 | 138,249 |
| `rust/` (`.rs`) | — | 7,883 |

### Invariant probes

| Probe | Result |
| :--- | :--- |
| LLM/provider imports in `hbllm/hcir` or `brain/reasoning` | None found (one `hbllm.memory.interface` import in `adapters/memory_migration_proxy.py`) |
| Bare `except:` in `hbllm` and `plugins` | 0 |
| `sqlite3.connect` outside sanctioned stores | Within the existing ratchet (8 files) |
| `TODO/FIXME/HACK` markers in `hbllm` | 0 |
| `torch.load` without `weights_only=True` | None found in the sampled sites |
| `tests/unit/architecture` | 32 passed (620 s, serial) |
| `hbllm diagnostics`, `kernel status` | Verifier attached, 6 stages, native substrate available, 0 divergences |

### Benchmark evidence read from `reports/`

| Report | n | Zero-shot solve | Generality | Transfer solve | Exploration deaths | Mean p50 / p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Held-out v2 `eval-phase2-r10` (`a26f09bd`) | 120 | 0.9250 | 0.8236 | 0.9035 | 4 | 21.5 / 54.46 |
| Dev (`b612fb5c`, 2026-11-24) | 80 | 0.9625 | 0.8318 | 0.9868 | 2 | 15.64 / 57.68 |
| Widened (`8d14e2cf`, 2026-11-23) | 120 | 0.7500 | 0.7425 | 0.9035 | 25 | 49.64 / 137.07 |

Held-out per family (zero-shot solve): continuous 1.00, symbolic 1.00, grid 0.95, stochastic_dynamic 0.85, partial_observability 0.80.
Widened per family: language_goals 1.00, multi_agent 1.00, continuous_control 0.95, partial_observability 0.85, stochastic_dynamics 0.40, long_horizon 0.30.

No benchmark has been recorded for `ad299024` or for the uncommitted tree.

---

## 7. Architecture improvements

1. **One runtime, three front ends.** `build_cognitive_system()` should be the only way a cognitive runtime is created, and HTTP, CLI and drivers should be thin adapters over a per-tenant `CognitiveSystem`. Today `serving` creates the legacy brain, `CognitiveBlackbox` wraps its own per-source systems, and only CLI and benchmarks use the root directly.
2. **Make the invariants structural, not conventional.** Private graph primitives, a `TransactionManager` that refuses to run unverified, tenant as a required transaction field, and stages that receive a frozen context would turn Invariants #1, #2, #4 and #5 from review rules into things the code cannot violate.
3. **Shrink the engine around a typed state object.** `EngineState` already exists in `world/state/state.py`. Finish the move: the engine becomes a coordinator, stages read a frozen view and return operations, and faculties are injected through protocols rather than discovered with `hasattr`.
4. **Separate lattice perception from cognition.** A domain-neutral observation type (entities, relations, affordances) at the pipeline boundary would let continuous and symbolic tasks enter without being rendered as grids, and would remove `(64, 64)` defaults from predictors.
5. **Evidence pipeline.** Every claim in `AGENTS.md`/README should be generated from a report keyed by git SHA: suite results, latency from a dedicated profile run, R1 roster from the latest ablation, graduation from the gate test. Add a dev ledger like the held-out ledger.
6. **Admission rule for faculties, applied to grounding and the experimental set.** Use the widened suite for R1, and keep "experimental" faculties behind a flag that defaults off.
7. **Retire or isolate the legacy tree.** Decide whether `hbllm/brain`, `hbllm/memory`, `hbllm/network`, `hbllm/model`, `hbllm/training` and `experiments/` are product. If not, move them to an optional distribution so the core package, its LOC budget and its type checking describe the Cognitive OS only.
8. **Typed service container.** Replace `services.extensions[str]` with declared fields so that wiring errors are caught by mypy instead of at request time.

---

## 8. Recommended order of work

### P0 — correctness and safety (do first)

1. Restore the working tree to an importable state and benchmark the pending refactor before committing (H-13).
2. Move the Tier-2 sandbox into `hbllm/` and remove kernel/serving imports of shimmed packages; add the wheel-install CI job (C-1).
3. Fix or remove the two command-executing plugins; include `plugins/` in bandit (C-4).
4. Triage the graduation gate; make it hermetic; regenerate the report (C-3).
5. Remove client-supplied `tenant_id` parameters and `"default"` fallbacks behind one dependency (H-1).
6. Change the default memory directory so it never binds to `checkpoints/` (H-2).

### P1 — make the product match the evidence

1. Per-tenant `CognitiveSystem` behind `/v1/chat/completions` with a test for the engine header; decide the fate of the legacy brain routes (C-2).
2. Dedicated, reproducible latency measurement with a CI gate; correct the documented numbers (H-4).
3. Grounding: default off, or add tasks that need it and prove Δ > 0 (H-3).
4. Run CI on all branches; audit real dependencies; add a lockfile; align the local venv with Python 3.11 and a compatible `torch`/`numpy` pair (H-9).
5. `TransactionManager` fails closed without verification stages; convert the raw-primitive call sites (H-7).

### P2 — structure

1. Retarget the size and `getattr` ratchets at the real files; remove `get_sub_attr` (H-6).
2. Finish the engine-state extraction; move blacklisted attributes into working memory (H-6, H-8).
3. Delete the 32 orphan modules; classify the 132 test-only modules; set `MAX_PATH_APPEND_SHIMS = 0` (H-10, C-1).
4. Event-store envelope hashing and compaction anchoring (H-11).
5. Observation protocol and removal of grid defaults from predictors (M-1).

### P3 — capability

1. Promote widened families into dev; run R1 on widened; update the roster (H-5, H-12).
2. Risk-sensitive planning for `stochastic_dynamics`; subgoal persistence and budget for `long_horizon` (H-5).
3. Sealed held-out language evaluation and an extraction-recall metric for the reading curriculum (M-6).
4. Rust: `pyo3` upgrade, tests for `simulation_engine` and `structure_matcher` (M-10).

---

## 9. What was not verified

- Whether the Docker image builds and serves (configuration read only).
- `ruff`, `mypy`, `bandit`, `cargo clippy/test` results.
- Integration tests (Postgres/Redis) and all plugin tests.
- Any benchmark number for `ad299024` or the uncommitted tree.
- The root cause of the graduation-gate failure (C-3) and of the out-of-range `goal_target` in the CLI smoke run (H-2).
- Whether each raw-primitive call site in H-7 operates on the live workspace graph or on a private one.
- Line-by-line correctness of the planners, assimilator and theory modules.
