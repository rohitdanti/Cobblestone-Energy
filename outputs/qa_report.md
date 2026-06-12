# Data Quality Report
**Run date:** 2026-06-11
**All checks passed:** True

---

## da_prices

### Coverage
- Expected hours: 59158
- Present hours: 59158
- Coverage: 100.0%
- Flag: False

### Gaps
- Total missing hours: 0
- Gaps ≤2h (forward-fillable): 0
- Gaps >2h (excluded from training): 0
- Note: Target (DA prices): missing rows DROPPED, never imputed.

### Outliers
- Count: 6
- Note: Negative DA prices are valid in Germany — flagged, not removed.

### Distribution
| Stat | Value |
|---|---|
| mean | 94.807 |
| std | 95.145 |
| min | -500.0 |
| p5 | 1.96 |
| p25 | 37.07 |
| p50 | 71.845 |
| p75 | 115.408 |
| p95 | 285.813 |
| max | 936.28 |

## load

### Coverage
- Expected hours: 59158
- Present hours: 59108
- Coverage: 99.915%
- Flag: False

### Gaps
- Total missing hours: 0
- Gaps ≤2h (forward-fillable): 0
- Gaps >2h (excluded from training): 0
- Note: Feature series: gaps ≤2h forward-filled; gaps >2h excluded from training.

### Outliers
- Count: 0
- Note: Negative DA prices are valid in Germany — flagged, not removed.

### Distribution
| Stat | Value |
|---|---|
| mean | 54606.336 |
| std | 9241.494 |
| min | 30893.055 |
| p5 | 40120.314 |
| p25 | 47057.386 |
| p50 | 54442.435 |
| p75 | 62231.088 |
| p95 | 69219.099 |
| max | 78154.37 |

## wind

### Coverage
- Expected hours: 59158
- Present hours: 59156
- Coverage: 99.997%
- Flag: False

### Gaps
- Total missing hours: 0
- Gaps ≤2h (forward-fillable): 0
- Gaps >2h (excluded from training): 0
- Note: Feature series: gaps ≤2h forward-filled; gaps >2h excluded from training.

### Outliers
- Count: 0
- Note: Negative DA prices are valid in Germany — flagged, not removed.

### Distribution
| Stat | Value |
|---|---|
| mean | 11691.489 |
| std | 9296.364 |
| min | 161.378 |
| p5 | 1583.49 |
| p25 | 4490.597 |
| p50 | 8918.938 |
| p75 | 16546.651 |
| p95 | 31423.348 |
| max | 46769.24 |

## solar

### Coverage
- Expected hours: 59158
- Present hours: 59156
- Coverage: 99.997%
- Flag: False

### Gaps
- Total missing hours: 0
- Gaps ≤2h (forward-fillable): 0
- Gaps >2h (excluded from training): 0
- Note: Feature series: gaps ≤2h forward-filled; gaps >2h excluded from training.

### Outliers
- Count: 0
- Note: Negative DA prices are valid in Germany — flagged, not removed.

### Distribution
| Stat | Value |
|---|---|
| mean | 6360.842 |
| std | 9778.41 |
| min | 0.0 |
| p5 | 0.0 |
| p25 | 0.0 |
| p50 | 278.779 |
| p75 | 10190.329 |
| p95 | 28152.476 |
| max | 50917.715 |

---

## Cross-Series Alignment
- Common timestamps: 59158
- Missing from any series: 0
- Aligned: True