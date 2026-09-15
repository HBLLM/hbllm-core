# Official ARC-AGI-3 Interactive Reasoning Benchmark Report
**Evaluation Date**: 2026-09-15 19:41:47
**Overall Level Completion Rate**: **0/8 (0.0%)**
**Mean Fluid Action Efficiency**: **228.2%** (vs Human Baseline)
**Mean Epistemic Brier Uncertainty**: **0.5157**
**Total Actions Executed**: 604 across 4 environment(s)
**Total Evaluation Time**: 17.68s

## 1. Environment Performance Breakdown
| Environment | Levels Completed | Win Rate | Actions Taken | Human Baseline | Efficiency | Brier Error |
|---|---|---|---|---|---|---|
| `ls20` | 0/2 | 0.0% | 129 | 145 | **692.5%** | 0.3773 |
| `wa30` | 0/2 | 0.0% | 200 | 190 | **104.0%** | 0.4334 |
| `cd82` | 0/2 | 0.0% | 200 | 63 | **31.5%** | 0.5296 |
| `su15` | 0/2 | 0.0% | 75 | 64 | **85.0%** | 0.7225 |

## 2. Level-by-Level Trace & Causal Dynamics
### Environment: `ls20`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 22 | 18.3% | 0.4642 | `A1:(-5,+0), A2:(+5,+0), A3:(+0,-5), A4:(+0,+5)` |
| Level 2 | **ACTIVE** | 9 | 123 | 1366.7% | 0.2904 | `A1:(-5,+0), A2:(+5,+0), A3:(+0,-5), A4:(+0,+5)` |

### Environment: `wa30`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 71 | 59.2% | 0.7069 | `A1:(-4,+0), A2:(+3,+0), A3:(+0,-4), A4:(+0,+4)` |
| Level 2 | **ACTIVE** | 80 | 119 | 148.8% | 0.1600 | `A1:(-4,+0), A2:(+3,+0), A3:(+0,-4), A4:(+0,+4), A5:(+0,+0)` |

### Environment: `cd82`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 55 | 55.0% | 0.2500 | `A1:(-1,+0), A2:(-1,+0), A3:(-1,+0)` |
| Level 2 | **ACTIVE** | 100 | 8 | 8.0% | 0.8093 | `A1:(-11,+2), A2:(+11,-2), A3:(-1,+0), A4:(+0,+21)` |

### Environment: `su15`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 37 | 22 | 59.5% | 0.7225 | `A6:(+1,-1)` |
| Level 2 | **ACTIVE** | 38 | 42 | 110.5% | 0.7225 | `A6:(+1,-1)` |
