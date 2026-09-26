# Official ARC-AGI-3 Interactive Reasoning Benchmark Report
**Evaluation Date**: 2026-09-16 09:55:42
**Overall Level Completion Rate**: **0/10 (0.0%)**
**Mean Fluid Action Efficiency**: **0.0%** (vs Human Baseline)
**Mean Epistemic Brier Uncertainty**: **0.5209**
**Total Actions Executed**: 818 across 10 environment(s)
**Total Evaluation Time**: 36.46s

## 1. Environment Performance Breakdown
| Environment | Levels Completed | Win Rate | Actions Taken | Human Baseline | Efficiency | Brier Error |
|---|---|---|---|---|---|---|
| `ar25` | 0/1 | 0.0% | 66 | 32 | **0.0%** | 0.5836 |
| `bp35` | 0/1 | 0.0% | 22 | 21 | **0.0%** | 0.4368 |
| `cn04` | 0/1 | 0.0% | 75 | 29 | **0.0%** | 0.2343 |
| `dc22` | 0/1 | 0.0% | 120 | 59 | **0.0%** | 0.5501 |
| `g50t` | 0/1 | 0.0% | 120 | 78 | **0.0%** | 0.3568 |
| `ka59` | 0/1 | 0.0% | 100 | 28 | **0.0%** | 0.7966 |
| `re86` | 0/1 | 0.0% | 100 | 26 | **0.0%** | 0.8517 |
| `sk48` | 0/1 | 0.0% | 120 | 61 | **0.0%** | 0.7524 |
| `sp80` | 0/1 | 0.0% | 30 | 39 | **0.0%** | 0.3969 |
| `sc25` | 0/1 | 0.0% | 65 | 36 | **0.0%** | 0.2500 |

## 2. Level-by-Level Trace & Causal Dynamics
### Environment: `ar25`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 66 | 32 | 0.0% | 0.5836 | `A1:(-3,+0), A2:(+3,+0), A3:(+2,+0), A4:(-2,+0)` |

### Environment: `bp35`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 22 | 21 | 0.0% | 0.4368 | `A3:(-11,+14), A4:(+0,+1), A6:(+0,-1), A1:(+0,-1)` |

### Environment: `cn04`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 75 | 29 | 0.0% | 0.2343 | `A1:(-2,+0), A2:(+2,+0), A3:(+0,-2), A4:(+1,+2)` |

### Environment: `dc22`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 59 | 0.0% | 0.5501 | `A1:(-2,+0), A2:(+2,+0), A3:(+0,-2), A4:(+0,+2)` |

### Environment: `g50t`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 78 | 0.0% | 0.3568 | `A1:(+1,+0), A2:(+1,+0), A3:(+0,+0), A4:(+30,+26), A5:(+1,+0)` |

### Environment: `ka59`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 28 | 0.0% | 0.7966 | `A1:(+0,+8), A2:(+2,+0), A3:(+0,-2), A4:(+0,+2), A6:(+0,+0)` |

### Environment: `re86`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 26 | 0.0% | 0.8517 | `A1:(-3,+0), A2:(+3,+0), A3:(+0,-3), A4:(+0,+3), A5:(+0,+0)` |

### Environment: `sk48`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 61 | 0.0% | 0.7524 | `A1:(-2,+0), A2:(+0,+0), A3:(+6,+4), A4:(+0,+0), A6:(-2,+0), A7:(+3,+0)` |

### Environment: `sp80`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 30 | 39 | 0.0% | 0.3969 | `A1:(+0,-1), A2:(+0,-1), A3:(+0,-1), A4:(+0,-1)` |

### Environment: `sc25`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 65 | 36 | 0.0% | 0.2500 | `A1:(+0,+0), A2:(+1,+0), A3:(+0,-2), A4:(+0,+2), A6:(+0,+0)` |
