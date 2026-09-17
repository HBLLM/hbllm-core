# Master Scientific Benchmark & Reproducibility Report
**Execution Timestamp**: 2026-09-17 21:36:45
**Evaluation Mode**: Quick Validation
**LLM Token Usage**: **0 tokens** ($0.00 total expenditure)

## 1. Summary Matrix
| Domain | Native Simulator | Literature Baseline | LLM Baseline | HBLLM HCIR | 95% Wilson CI | Token Cost |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **AI2-THOR** | Native Unity 3D / Simulated Engine | 35-45% (Embodied RL) | 0.0% (Coordinate divergence) | **100.0%** (4 eps) | [0.510, 1.000] | **0 tokens** ($0.00) |
| **Safety-Gymnasium** | safety-gymnasium / Dual Engine | 50-65% (PPO-Lagrangian) | 0.0% (Hazard violations) | **100.0%** (4 eps) | [0.510, 1.000] | **0 tokens** ($0.00) |
| **Sokoban** | gym_sokoban (5 Boxoban Tiers) | 82-85% (DRC(3,3), 1B steps) | < 10% (Corner traps) | **100.0%** (5 eps) | [0.566, 1.000] | **0 tokens** ($0.00) |
| **Overcooked-AI** | overcooked_ai_py (5 Layouts) | 60-70% (BC / PPO Self-Play) | ~15.0% (Counter clutter) | **100.0%** (5 eps) | [0.566, 1.000] | **0 tokens** ($0.00) |