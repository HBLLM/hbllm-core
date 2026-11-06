# Official ARC-AGI-3 Interactive Reasoning Benchmark Report
**Evaluation Date**: 2026-09-29 10:20:39
**Overall Level Completion Rate**: **0/6 (0.0%)**
**Mean Fluid Action Efficiency**: **0.0%** (vs Human Baseline)
**Mean Epistemic Brier Uncertainty**: **0.5516**
**Total Actions Executed**: 456 across 3 environment(s)
**Total Evaluation Time**: 33.23s

## 1. Environment Performance Breakdown
| Environment | Levels Completed | Win Rate | Actions Taken | Human Baseline | Efficiency | Brier Error |
|---|---|---|---|---|---|---|
| `ar25` | 0/2 | 0.0% | 128 | 82 | **0.0%** | 0.7371 |
| `sb26` | 0/2 | 0.0% | 200 | 46 | **0.0%** | 0.5625 |
| `tr87` | 0/2 | 0.0% | 128 | 112 | **0.0%** | 0.3552 |

## 2. Level-by-Level Trace & Causal Dynamics
### Environment: `ar25`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 64 | 32 | 0.0% | 0.8131 | `A1:(-3,+0), A2:(+3,+0), A3:(+0,+3), A4:(+0,-3), A5:(+0,+0)` |
| Level 2 | **ACTIVE** | 64 | 50 | 0.0% | 0.6612 | `A1:(-3,+0), A2:(+3,+0), A3:(+0,+3), A4:(+0,-3), A5:(+0,+0)` |

### Environment: `sb26`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 18 | 0.0% | 0.5625 | `A6:(+0,+0)` |
| Level 2 | **ACTIVE** | 100 | 28 | 0.0% | 0.5625 | `A6:(+0,+0)` |

### Environment: `tr87`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 54 | 0.0% | 0.3505 | `A1:(+0,+0), A2:(+0,+0), A3:(+0,+0), A4:(+0,+0)` |
| Level 2 | **ACTIVE** | 28 | 58 | 0.0% | 0.3600 | `A1:(+0,+0), A2:(+0,+0), A3:(+0,+0), A4:(+0,+0)` |
