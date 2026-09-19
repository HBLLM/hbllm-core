# Official ARC-AGI-3 Interactive Reasoning Benchmark Report
**Evaluation Date**: 2026-09-16 07:33:25
**Overall Level Completion Rate**: **3/25 (12.0%)**
**Mean Fluid Action Efficiency**: **51.2%** (vs Human Baseline)
**Mean Epistemic Brier Uncertainty**: **0.4375**
**Total Actions Executed**: 1930 across 25 environment(s)
**Total Evaluation Time**: 158.80s

## 1. Environment Performance Breakdown
| Environment | Levels Completed | Win Rate | Actions Taken | Human Baseline | Efficiency | Brier Error |
|---|---|---|---|---|---|---|
| `tu93` | 0/1 | 0.0% | 50 | 19 | **38.0%** | 0.3238 |
| `ar25` | 0/1 | 0.0% | 64 | 32 | **50.0%** | 0.6243 |
| `re86` | 0/1 | 0.0% | 100 | 26 | **26.0%** | 0.8517 |
| `su15` | 0/1 | 0.0% | 37 | 22 | **59.5%** | 0.7225 |
| `m0r0` | 0/1 | 0.0% | 120 | 30 | **25.0%** | 0.3117 |
| `cn04` | 0/1 | 0.0% | 75 | 29 | **38.7%** | 0.2343 |
| `ft09` | 0/1 | 0.0% | 120 | 43 | **35.8%** | 0.2500 |
| `tr87` | 0/1 | 0.0% | 120 | 54 | **45.0%** | 0.2500 |
| `sc25` | 0/1 | 0.0% | 52 | 36 | **69.2%** | 0.2500 |
| `lp85` | 1/1 | 100.0% | 82 | 17 | **20.7%** | 0.0225 |
| `dc22` | 0/1 | 0.0% | 120 | 59 | **49.2%** | 0.6958 |
| `sp80` | 0/1 | 0.0% | 30 | 39 | **130.0%** | 0.3969 |
| `ka59` | 0/1 | 0.0% | 100 | 28 | **28.0%** | 0.6044 |
| `g50t` | 0/1 | 0.0% | 120 | 78 | **65.0%** | 0.2500 |
| `sb26` | 0/1 | 0.0% | 120 | 18 | **15.0%** | 0.7225 |
| `lf52` | 0/1 | 0.0% | 64 | 32 | **50.0%** | 0.2500 |
| `bp35` | 0/1 | 0.0% | 40 | 21 | **52.5%** | 0.2730 |
| `s5i5` | 0/1 | 0.0% | 50 | 20 | **40.0%** | 0.7225 |
| `r11l` | 0/1 | 0.0% | 60 | 22 | **36.7%** | 0.7225 |
| `sk48` | 0/1 | 0.0% | 120 | 61 | **50.8%** | 0.7354 |
| `wa30` | 1/1 | 100.0% | 44 | 71 | **161.4%** | 0.0157 |
| `vc33` | 0/1 | 0.0% | 50 | 7 | **14.0%** | 0.7225 |
| `ls20` | 1/1 | 100.0% | 31 | 22 | **71.0%** | 0.0133 |
| `cd82` | 0/1 | 0.0% | 100 | 55 | **55.0%** | 0.2500 |
| `tn36` | 0/1 | 0.0% | 61 | 32 | **52.5%** | 0.7225 |

## 2. Level-by-Level Trace & Causal Dynamics
### Environment: `tu93`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 50 | 19 | 38.0% | 0.3238 | `A1:(+0,-1), A2:(+0,-1), A3:(+0,-1), A4:(+0,-1)` |

### Environment: `ar25`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 64 | 32 | 50.0% | 0.6243 | `A1:(-3,+0), A2:(+3,+0), A3:(+2,+0), A4:(-2,+0)` |

### Environment: `re86`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 26 | 26.0% | 0.8517 | `A1:(-3,+0), A2:(+3,+0), A3:(+0,-3), A4:(+0,+3), A5:(+0,+0)` |

### Environment: `su15`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 37 | 22 | 59.5% | 0.7225 | `A6:(+1,-1)` |

### Environment: `m0r0`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 30 | 25.0% | 0.3117 | `A1:(-5,+0), A2:(+5,+0), A3:(+0,+0), A4:(+0,+0), A5:(+0,+0), A6:(+0,+0)` |

### Environment: `cn04`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 75 | 29 | 38.7% | 0.2343 | `A1:(-2,+0), A2:(+2,+0), A3:(+0,-2), A4:(+1,+2)` |

### Environment: `ft09`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 43 | 35.8% | 0.2500 | `A6:(+0,+0)` |

### Environment: `tr87`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 54 | 45.0% | 0.2500 | `A1:(+0,+0)` |

### Environment: `sc25`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 52 | 36 | 69.2% | 0.2500 | `A1:(+0,+0)` |

### Environment: `lp85`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **PASSED** | 82 | 17 | 20.7% | 0.0225 | `A6:(+0,-3)` |

### Environment: `dc22`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 59 | 49.2% | 0.6958 | `A1:(-2,+0), A2:(+2,+0), A3:(+0,-2), A4:(+0,+2)` |

### Environment: `sp80`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 30 | 39 | 130.0% | 0.3969 | `A1:(+0,-1), A2:(+0,-1), A3:(+0,-1), A4:(+0,-1)` |

### Environment: `ka59`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 28 | 28.0% | 0.6044 | `A1:(-2,+0), A2:(+2,+0), A3:(+0,-2), A4:(+0,+2)` |

### Environment: `g50t`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 78 | 65.0% | 0.2500 | `A1:(+8,+4), A2:(+0,+0), A3:(+1,+0), A4:(+0,+0), A5:(+1,+0)` |

### Environment: `sb26`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 18 | 15.0% | 0.7225 | `A6:(-13,+0)` |

### Environment: `lf52`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 64 | 32 | 50.0% | 0.2500 | `A1:(+0,+0)` |

### Environment: `bp35`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 40 | 21 | 52.5% | 0.2730 | `A3:(+0,-5), A4:(-12,+0), A6:(+0,+0), A1:(+0,+4), A7:(+0,+0), A2:(+0,+4)` |

### Environment: `s5i5`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 50 | 20 | 40.0% | 0.7225 | `A6:(+1,+0)` |

### Environment: `r11l`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 60 | 22 | 36.7% | 0.7225 | `A6:(+1,+2)` |

### Environment: `sk48`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 120 | 61 | 50.8% | 0.7354 | `A1:(-1,+0), A2:(+0,+0), A3:(+8,+4), A4:(+0,+0), A6:(-2,+0), A7:(+3,+0)` |

### Environment: `wa30`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **PASSED** | 44 | 71 | 161.4% | 0.0157 | `A1:(-4,+0), A2:(+4,+0), A3:(+0,-3), A4:(+0,+4), A5:(+0,+0)` |

### Environment: `vc33`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 50 | 7 | 14.0% | 0.7225 | `A6:(+0,-2)` |

### Environment: `ls20`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **PASSED** | 31 | 22 | 71.0% | 0.0133 | `A1:(-5,+0), A2:(+5,+0), A3:(+0,-5), A4:(+0,+5)` |

### Environment: `cd82`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 100 | 55 | 55.0% | 0.2500 | `A1:(-1,+0), A2:(-1,+0), A3:(+0,+0), A4:(-1,+0)` |

### Environment: `tn36`
| Level | Completed | Actions | Baseline | Efficiency | Brier | Inferred Motor Dynamics |
|---|---|---|---|---|---|---|
| Level 1 | **ACTIVE** | 61 | 32 | 52.5% | 0.7225 | `A6:(+1,+0)` |
