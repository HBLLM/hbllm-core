# Official ARC-AGI-3 Interactive Reasoning Benchmark Report
**Evaluation Date**: 2026-09-16 09:28:45
**Overall Level Completion Rate**: **0/10 (0.0%)**
**Mean Fluid Action Efficiency**: **0.0%** (vs Human Baseline)
**Mean Epistemic Brier Uncertainty**: **0.5184**
**Total Actions Executed**: 816 across 10 environment(s)
**Total Evaluation Time**: 104.35s

## 1. Environment Performance Breakdown
| Environment | Levels Completed | Win Rate | Actions Taken | Human Baseline | Efficiency | Brier Error |
|---|---|---|---|---|---|---|
| `ar25` | 0/1 | 0.0% | 64 | 32 | **0.0%** | 0.6243 |
| `bp35` | 0/1 | 0.0% | 22 | 21 | **0.0%** | 0.4368 |
| `cn04` | 0/1 | 0.0% | 75 | 29 | **0.0%** | 0.2343 |
| `dc22` | 0/1 | 0.0% | 120 | 59 | **0.0%** | 0.6958 |
| `g50t` | 0/1 | 0.0% | 120 | 78 | **0.0%** | 0.3539 |
| `ka59` | 0/1 | 0.0% | 100 | 28 | **0.0%** | 0.6044 |
| `re86` | 0/1 | 0.0% | 100 | 26 | **0.0%** | 0.8517 |
| `sk48` | 0/1 | 0.0% | 120 | 61 | **0.0%** | 0.7354 |
| `sp80` | 0/1 | 0.0% | 30 | 39 | **0.0%** | 0.3969 |
| `sc25` | 0/1 | 0.0% | 65 | 36 | **0.0%** | 0.2500 |

## 2. Level-by-Level Trace & Causal Dynamics
### Environment: `ar25`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 64 | 32 | 0.0% | 0.6243 | `A1:(-3,+0), A2:(+3,+0), A3:(+2,+0), A4:(-2,+0)` |

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
| Level 1 | **ACTIVE** | 120 | 59 | 0.0% | 0.6958 | `A1:(-2,+0), A2:(+2,+0), A3:(+0,-2), A4:(+0,+2)` |

### Environment: `g50t`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 78 | 0.0% | 0.3539 | `A1:(+1,+0), A2:(+1,+0), A3:(+0,+0), A4:(+30,+26), A5:(+1,+0)` |

### Environment: `ka59`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 28 | 0.0% | 0.6044 | `A1:(-2,+0), A2:(+2,+0), A3:(+0,-2), A4:(+0,+2)` |

### Environment: `re86`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 26 | 0.0% | 0.8517 | `A1:(-3,+0), A2:(+3,+0), A3:(+0,-3), A4:(+0,+3), A5:(+0,+0)` |

### Environment: `sk48`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 61 | 0.0% | 0.7354 | `A1:(-1,+0), A2:(+0,+0), A3:(+8,+4), A4:(+0,+0), A6:(-2,+0), A7:(+3,+0)` |

### Environment: `sp80`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 30 | 39 | 0.0% | 0.3969 | `A1:(+0,-1), A2:(+0,-1), A3:(+0,-1), A4:(+0,-1)` |

### Environment: `sc25`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 65 | 36 | 0.0% | 0.2500 | `A1:(+0,+0), A2:(+1,+0), A3:(+0,-2), A4:(+0,+2), A6:(+0,+0)` |
