# Rule: General Intelligence Invariant for Core

The cognitive core of HBLLM (`core/hbllm/`) must ALWAYS embody domain-general intelligence, never specialized single-task machinery.

## 1. Absolute Prohibition of Single-Task Solvers in Core
- **No Single-Task Routines:** Never add functions, classes, heuristic branches, or algorithms designed only to solve a single game, level, or benchmark (e.g., dedicated ARC-AGI game/puzzle solvers, bespoke grid game tricks, or level-specific hacks) into `core/hbllm/`.
- **Universal Principles & Declared Priors Only:** All innate mechanisms in the core are strictly bounded by `docs/architecture/priors.md` (Spelke Core Knowledge: objectness, spatial topology, agency, discrete numerosity; plus temporal precedence, Pearl's causal *do(a)* calculus, and expected free energy / active inference). Any rule, concept, affordance, or linguistic meaning outside that catalog is empirical world knowledge and **must be learned in-environment or read via symbolic constructions**, never hardcoded in Python.

## 2. Benchmark & Adapter Layering
- **Adapters Outside Core:** Environment wrappers, competition-specific harnesses (e.g. `kaggle_submission/`, `benchmarks/`), and integration adapters (e.g. AI2-THOR, Safety Gym) belong outside the cognitive core.
- **Solving Novel Domains:** When a benchmark or environment exposes a failure mode, strengthen universal Layer-0 math/physics ALUs or Layer-2 homoiconic skill/construction acquisition — NEVER write a bespoke solver or hardcode domain rules in `hbllm/`.
