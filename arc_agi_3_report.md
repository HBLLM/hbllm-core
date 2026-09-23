# Official ARC-AGI-3 Interactive Reasoning Benchmark Report
**Evaluation Date**: 2026-09-23 09:42:12
**Overall Level Completion Rate**: **2/6 (33.3%)**
**Mean Fluid Action Efficiency**: **41.8%** (vs Human Baseline)
**Mean Epistemic Brier Uncertainty**: **0.4608**
**Total Actions Executed**: 405 across 3 environment(s)
**Total Evaluation Time**: 53.60s

## 1. Environment Performance Breakdown
| Environment | Levels Completed | Win Rate | Actions Taken | Human Baseline | Efficiency | Brier Error |
|---|---|---|---|---|---|---|
| `wa30` | 1/2 | 50.0% | 109 | 190 | **88.8%** | 0.4100 |
| `ls20` | 1/2 | 50.0% | 96 | 145 | **36.7%** | 0.4653 |
| `cd82` | 0/2 | 0.0% | 200 | 63 | **0.0%** | 0.5070 |

## 2. Level-by-Level Trace & Causal Dynamics
### Environment: `wa30`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **PASSED** | 40 | 71 | 177.5% | 0.0119 | `A1:(-4,+0), A2:(+4,+0), A3:(+0,-4), A4:(+0,+4), A5:(+0,+0)` |
| Level 2 | **ACTIVE** | 69 | 119 | 0.0% | 0.8082 | `A1:(-4,+0), A2:(+4,+0), A3:(+0,-4), A4:(+0,+4), A5:(+0,+0)` |

### Environment: `ls20`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **PASSED** | 30 | 22 | 73.3% | 0.0459 | `A1:(-5,+0), A2:(+5,+0), A3:(+0,-5), A4:(+0,+5)` |
| Level 2 | **ACTIVE** | 66 | 123 | 0.0% | 0.8847 | `A1:(-5,+0), A2:(+5,+0), A3:(+0,-5), A4:(+0,+5)` |

### Environment: `cd82`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 55 | 0.0% | 0.5210 | `A1:(-4,+1), A2:(+4,+0), A3:(+0,-4), A4:(+1,+4), A5:(+0,+0)` |
| Level 2 | **ACTIVE** | 100 | 8 | 0.0% | 0.4931 | `A1:(-5,-1), A2:(+6,+1), A3:(+1,-5), A4:(-1,+6), A5:(+0,+0)` |
