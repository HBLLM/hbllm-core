# Official ARC-AGI-3 Interactive Reasoning Benchmark Report
**Evaluation Date**: 2026-09-16 10:13:45
**Overall Level Completion Rate**: **2/2 (100.0%)**
**Mean Fluid Action Efficiency**: **116.2%** (vs Human Baseline)
**Mean Epistemic Brier Uncertainty**: **0.0145**
**Total Actions Executed**: 75 across 2 environment(s)
**Total Evaluation Time**: 16.71s

## 1. Environment Performance Breakdown
| Environment | Levels Completed | Win Rate | Actions Taken | Human Baseline | Efficiency | Brier Error |
|---|---|---|---|---|---|---|
| `ls20` | 1/1 | 100.0% | 31 | 22 | **71.0%** | 0.0133 |
| `wa30` | 1/1 | 100.0% | 44 | 71 | **161.4%** | 0.0157 |

## 2. Level-by-Level Trace & Causal Dynamics
### Environment: `ls20`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **PASSED** | 31 | 22 | 71.0% | 0.0133 | `A1:(-5,+0), A2:(+5,+0), A3:(+0,-5), A4:(+0,+5)` |

### Environment: `wa30`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **PASSED** | 44 | 71 | 161.4% | 0.0157 | `A1:(-4,+0), A2:(+4,+0), A3:(+0,-3), A4:(+0,+4), A5:(+0,+0)` |
