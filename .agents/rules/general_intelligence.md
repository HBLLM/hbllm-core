# Rule: General Intelligence Invariant for Core

The cognitive core of HBLLM (`core/hbllm/`) must ALWAYS embody domain-general intelligence, never specialized single-task machinery.

## 1. Absolute Prohibition of Single-Task Solvers in Core
- **No Single-Task Routines:** Never add functions, classes, heuristic branches, or algorithms designed only to solve a single game, level, or benchmark (e.g., dedicated ARC-AGI game/puzzle solvers, bespoke grid game tricks, or level-specific hacks) into `core/hbllm/`.
- **Universal Principles Only:** All algorithms in the core must be grounded in universal physical, mathematical, or neuro-symbolic principles (active inference, Pearl's causal *do(a)* calculus, topological manifold routing, dynamic affordance learning, A*/MCTS counterfactual simulation).

## 2. Benchmark & Adapter Layering
- **Adapters Outside Core:** Environment wrappers, competition-specific harnesses (e.g. `kaggle_submission/`, `benchmarks/`), and integration adapters (e.g. AI2-THOR, Safety Gym) belong outside the cognitive core.
- **Solving Novel Domains:** If a benchmark or environment highlights a failure mode, solve it by improving the domain-general reasoning faculties (e.g., richer causal hypothesis pruning, topological cut-set decomposition, general deadlock detection), NEVER by writing a bespoke solver for that specific game or task.
