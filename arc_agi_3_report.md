# Official ARC-AGI-3 Interactive Reasoning Benchmark Report
**Evaluation Date**: 2026-09-17 21:02:34
**Overall Level Completion Rate**: **3/8 (37.5%)**
**Mean Fluid Action Efficiency**: **58.7%** (vs Human Baseline)
**Mean Epistemic Brier Uncertainty**: **0.3532**
**Total Actions Executed**: 460 across 4 environment(s)
**Total Evaluation Time**: 24.40s

## 1. Environment Performance Breakdown
| Environment | Levels Completed | Win Rate | Actions Taken | Human Baseline | Efficiency | Brier Error |
|---|---|---|---|---|---|---|
| `ls20` | 1/2 | 50.0% | 99 | 145 | **35.5%** | 0.4297 |
| `wa30` | 2/2 | 100.0% | 95 | 190 | **199.5%** | 0.0105 |
| `cd82` | 0/2 | 0.0% | 200 | 63 | **0.0%** | 0.2500 |
| `su15` | 0/2 | 0.0% | 66 | 64 | **0.0%** | 0.7225 |

## 2. Level-by-Level Trace & Causal Dynamics
### Environment: `ls20`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **PASSED** | 31 | 22 | 71.0% | 0.0133 | `A1:(-5,+0), A2:(+5,+0), A3:(+0,-5), A4:(+0,+5)` |
| Level 2 | **ACTIVE** | 68 | 123 | 0.0% | 0.8461 | `A1:(-5,+0), A2:(+5,+0), A3:(+0,-5), A4:(+0,+5)` |

### Environment: `wa30`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **PASSED** | 36 | 71 | 197.2% | 0.0173 | `A1:(-4,+0), A2:(+4,+0), A3:(+0,-3), A4:(+0,+4), A5:(+0,+0)` |
| Level 2 | **PASSED** | 59 | 119 | 201.7% | 0.0037 | `A1:(-4,+0), A2:(+4,+0), A3:(+0,-4), A4:(+0,+4), A5:(+0,+0)` |

### Environment: `cd82`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 55 | 0.0% | 0.2500 | `A1:(+0,+0), A2:(+0,+0), A3:(+1,-5), A4:(-1,+5)` |
| Level 2 | **ACTIVE** | 100 | 8 | 0.0% | 0.2500 | `A1:(+0,+0), A2:(+0,+0), A3:(+1,-5), A4:(-1,+5)` |

### Environment: `su15`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 33 | 22 | 0.0% | 0.7225 | `A6:(+0,+0)` |
| Level 2 | **ACTIVE** | 33 | 42 | 0.0% | 0.7225 | `A6:(+0,+0)` |
