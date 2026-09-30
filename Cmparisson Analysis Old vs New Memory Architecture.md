# OLD vs NEW — SFM Simulation Benchmark

## Executive Summary

This report compares two implementations, **OLD** and **NEW**, running the same SFM simulation under the same benchmarking conditions.

Each implementation was executed three times, measuring:

- Execution time
- CPU time
- Initial RAM usage
- Peak RAM usage
- Final RAM usage

The results show a substantial improvement in both computational performance and memory efficiency.

| Metric | OLD Average | NEW Average | Reduction | Improvement Factor |
|---|---:|---:|---:|---:|
| ⏱️ Execution Time | **141,619 ms** | **58,204 ms** | **58.90%** | **2.43× faster** |
| 🖥️ CPU Time | **144,620 ms** | **59,513 ms** | **58.85%** | **2.43× less CPU** |
| 🧠 Peak RAM | **978.60 MB** | **128.78 MB** | **86.84%** | **7.60× less RAM** |
| 🧠 Final RAM | **566.87 MB** | **121.63 MB** | **78.54%** | **4.66× less RAM** |

The NEW implementation reduces execution time and CPU usage by approximately **59%**, while reducing peak memory consumption by almost **87%**.

---

# 1. Raw Benchmark Results

## OLD

### OLD - SFM - ITER 1

```text
Time:          147400.99 ms
CPU:           150530.00 ms
Initial RAM:      110.91 MB
Peak RAM:         977.42 MB
Final RAM:        544.80 MB
```

### OLD - SFM - ITER 2

```text
Time:          141900.77 ms
CPU:           144970.00 ms
Initial RAM:      111.18 MB
Peak RAM:         979.20 MB
Final RAM:        529.16 MB
```

### OLD - SFM - ITER 3

```text
Time:          135554.99 ms
CPU:           138360.00 ms
Initial RAM:      110.94 MB
Peak RAM:         979.19 MB
Final RAM:        626.64 MB
```

## NEW

### NEW - SFM - ITER 1

```text
Time:           57894.20 ms
CPU:            59220.00 ms
Initial RAM:      110.96 MB
Peak RAM:         128.80 MB
Final RAM:        121.65 MB
```

### NEW - SFM - ITER 2

```text
Time:           57964.89 ms
CPU:            59250.00 ms
Initial RAM:      111.02 MB
Peak RAM:         128.84 MB
Final RAM:        121.69 MB
```

### NEW - SFM - ITER 3

```text
Time:           58752.69 ms
CPU:            60070.00 ms
Initial RAM:      110.88 MB
Peak RAM:         128.71 MB
Final RAM:        121.56 MB
```

---

# 2. Average Results

The following values are calculated from the three iterations for each implementation.

| Metric | OLD Average | NEW Average |
|---|---:|---:|
| Execution Time | **141,618.92 ms** | **58,203.93 ms** |
| CPU Time | **144,620.00 ms** | **59,513.33 ms** |
| Initial RAM | **111.01 MB** | **110.95 MB** |
| Peak RAM | **978.60 MB** | **128.78 MB** |
| Final RAM | **566.87 MB** | **121.63 MB** |

The initial RAM usage is practically identical between both implementations, which makes the comparison particularly useful.

---

# 3. Calculation Methodology

For reductions, the following formula is used:

```text
Reduction (%) = ((OLD - NEW) / OLD) × 100
```

For the improvement factor:

```text
Factor = OLD / NEW
```

For example, for execution time:

```text
OLD = 141,618.92 ms
NEW = 58,203.93 ms

Reduction:
((141,618.92 - 58,203.93) / 141,618.92) × 100
= 58.90%

Factor:
141,618.92 / 58,203.93
= 2.43×

Therefore:

NEW requires approximately 58.90% less execution time,
or equivalently, it is approximately 2.43× faster.
```

---

# 4. Execution Time Analysis

## Iteration-by-Iteration Comparison

| Iteration | OLD | NEW | Reduction | NEW is |
|---|---:|---:|---:|---:|
| ITER 1 | 147,400.99 ms | 57,894.20 ms | **60.72%** | **2.55× faster** |
| ITER 2 | 141,900.77 ms | 57,964.89 ms | **59.15%** | **2.45× faster** |
| ITER 3 | 135,554.99 ms | 58,752.69 ms | **56.66%** | **2.31× faster** |
| **Average** | **141,618.92 ms** | **58,203.93 ms** | **58.90%** | **2.43× faster** |

## Execution Time Graph

```text
Execution Time (seconds)

OLD
ITER 1  █████████████████████████████████████████████  147.40 s
ITER 2  ███████████████████████████████████████████   141.90 s
ITER 3  █████████████████████████████████████████     135.55 s

NEW
ITER 1  ██████████████████                              57.89 s
ITER 2  ██████████████████                              57.96 s
ITER 3  ██████████████████                              58.75 s
```

## Interpretation

The execution time decreases from approximately:

```text
OLD: 141.62 seconds
NEW:  58.20 seconds
```

This represents a reduction of:

```text
83.41 seconds per simulation
```

In other words, NEW requires only:

```text
41.10% of the execution time of OLD
```

or:

```text
2.43× faster
```

If 100 simulations were executed:

```text
OLD:
141.62 s × 100 = 14,161.89 s
≈ 3.93 hours

NEW:
58.20 s × 100 = 5,820.39 s
≈ 1.62 hours
```

Approximate time saved over 100 simulations:

```text
≈ 2.32 hours
```

---

# 5. CPU Analysis

## Iteration-by-Iteration Comparison

| Iteration | OLD | NEW | Reduction | Factor |
|---|---:|---:|---:|---:|
| ITER 1 | 150,530 ms | 59,220 ms | **60.65%** | **2.54×** |
| ITER 2 | 144,970 ms | 59,250 ms | **59.15%** | **2.45×** |
| ITER 3 | 138,360 ms | 60,070 ms | **56.58%** | **2.30×** |
| **Average** | **144,620 ms** | **59,513 ms** | **58.85%** | **2.43×** |

## CPU Graph

```text
CPU Time (seconds)

OLD
ITER 1  ███████████████████████████████████████████████  150.53 s
ITER 2  █████████████████████████████████████████████   144.97 s
ITER 3  ███████████████████████████████████████████     138.36 s

NEW
ITER 1  ██████████████████                                59.22 s
ITER 2  ██████████████████                                59.25 s
ITER 3  ██████████████████                                60.07 s
```

## Interpretation

Average CPU time:

```text
OLD: 144.62 seconds
NEW:  59.51 seconds
```

Reduction:

```text
58.85%
```

Factor:

```text
2.43×
```

NEW therefore consumes approximately **41.15% of the CPU time required by OLD**.

The fact that CPU reduction is almost identical to execution-time reduction is significant.

It suggests that the improvement is primarily associated with performing less computational work, rather than simply waiting less on external operations such as I/O.

---

# 6. Peak RAM Analysis

This is the largest improvement observed in the benchmark.

## Iteration-by-Iteration Comparison

| Iteration | OLD | NEW | Reduction | Factor |
|---|---:|---:|---:|---:|
| ITER 1 | 977.42 MB | 128.80 MB | **86.82%** | **7.59×** |
| ITER 2 | 979.20 MB | 128.84 MB | **86.84%** | **7.60×** |
| ITER 3 | 979.19 MB | 128.71 MB | **86.85%** | **7.61×** |
| **Average** | **978.60 MB** | **128.78 MB** | **86.84%** | **7.60×** |

## Peak RAM Graph

```text
Peak RAM

OLD   ██████████████████████████████████████████████████  978.6 MB
NEW   ███████                                               128.8 MB
```

## Interpretation

Average peak memory:

```text
OLD: 978.60 MB
NEW: 128.78 MB
```

Absolute reduction:

```text
978.60 - 128.78
= 849.82 MB
```

Percentage reduction:

```text
86.84%
```

Factor:

```text
978.60 / 128.78
= 7.60×
```

NEW uses only approximately:

```text
13.16%
```

of the peak RAM used by OLD.

Therefore:

> OLD requires approximately 7.6× more peak memory than NEW.

This is an extremely significant memory optimization.

---

# 7. Final RAM Analysis

Peak memory is not the only improvement.

The amount of memory retained at the end of the simulation is also substantially lower with NEW.

## Iteration-by-Iteration Comparison

| Iteration | OLD | NEW | Reduction | Factor |
|---|---:|---:|---:|---:|
| ITER 1 | 544.80 MB | 121.65 MB | **77.67%** | **4.48×** |
| ITER 2 | 529.16 MB | 121.69 MB | **77.00%** | **4.35×** |
| ITER 3 | 626.64 MB | 121.56 MB | **80.61%** | **5.16×** |
| **Average** | **566.87 MB** | **121.63 MB** | **78.54%** | **4.66×** |

## Final RAM Graph

```text
RAM at Completion

OLD   ██████████████████████████████████████████████████  566.9 MB
NEW   ███████████                                          121.6 MB
```

## Interpretation

Average final RAM:

```text
OLD: 566.87 MB
NEW: 121.63 MB
```

Absolute reduction:

```text
566.87 - 121.63
= 445.24 MB
```

Percentage reduction:

```text
78.54%
```

Factor:

```text
566.87 / 121.63
= 4.66×
```

NEW therefore leaves only approximately:

```text
21.46%
```

of the memory retained by OLD.

This suggests a substantial reduction in memory retention after the simulation completes.

---

# 8. Initial RAM vs Peak RAM

The initial RAM consumption is almost identical between both implementations.

Average initial RAM:

```text
OLD ≈ 111.01 MB
NEW ≈ 110.95 MB
```

This is important because both implementations start from practically the same memory baseline.

## OLD

```text
Initial RAM:  ~111 MB
Peak RAM:     ~979 MB

Approximate increase:
979 - 111 ≈ +868 MB
```

## NEW

```text
Initial RAM:  ~111 MB
Peak RAM:     ~129 MB

Approximate increase:
129 - 111 ≈ +18 MB
```

Therefore, the memory growth during execution is dramatically different:

```text
OLD: ≈ +868 MB
NEW: ≈  +18 MB
```

The additional memory growth of OLD is approximately:

```text
868 / 18 ≈ 48×
```

the growth observed with NEW.

This strongly suggests a major difference in how intermediate data or in-memory structures are handled by the two implementations.

The exact cause would require profiling the implementation itself, but the benchmark clearly demonstrates the effect.

---

# 9. Stability and Variability

Performance is not only about the average. Consistency between runs is also important.

## OLD

Execution times:

```text
147.401 s
141.901 s
135.555 s
```

Minimum:

```text
135.555 s
```

Maximum:

```text
147.401 s
```

Range:

```text
147.401 - 135.555
= 11.846 s
```

Approximate variation relative to the average:

```text
11.846 / 141.619 × 100
≈ 8.36%
```

## NEW

Execution times:

```text
57.894 s
57.965 s
58.753 s
```

Minimum:

```text
57.894 s
```

Maximum:

```text
58.753 s
```

Range:

```text
58.753 - 57.894
= 0.858 s
```

Approximate variation relative to the average:

```text
0.858 / 58.204 × 100
≈ 1.47%
```

## Stability Graph

```text
Variation between iterations

OLD   ████████████████████████████████████  ~8.36%
NEW   ███████                               ~1.47%
```

The range between the fastest and slowest iteration is reduced from approximately:

```text
11.85 seconds → 0.86 seconds
```

This represents a reduction of approximately:

```text
92.8%
```

in the execution-time range.

Therefore, NEW is not only significantly faster, but also considerably more consistent across repeated executions.

---

# 10. Global Comparison

Taking OLD as the 100% baseline:

| Metric | OLD Baseline | NEW Relative Usage |
|---|---:|---:|
| Execution Time | 100% | **41.10%** |
| CPU Time | 100% | **41.15%** |
| Peak RAM | 100% | **13.16%** |
| Final RAM | 100% | **21.46%** |

## Visual Comparison

```text
EXECUTION TIME

OLD  ██████████████████████████████████████████████████  100%
NEW  ████████████████████                                41%


CPU TIME

OLD  ██████████████████████████████████████████████████  100%
NEW  ████████████████████                                41%


PEAK RAM

OLD  ██████████████████████████████████████████████████  100%
NEW  ███████                                             13%


FINAL RAM

OLD  ██████████████████████████████████████████████████  100%
NEW  ███████████                                         21%
```

---

# 11. Improvement Factors

Another way to understand the results is by looking at the multiplication factors.

## Execution Time

```text
OLD = 141.62 s
NEW =  58.20 s

141.62 / 58.20 = 2.43×
```

NEW is approximately:

**2.43× faster**

---

## CPU

```text
OLD = 144.62 s
NEW =  59.51 s

144.62 / 59.51 = 2.43×
```

NEW requires approximately:

**2.43× less CPU time**

---

## Peak RAM

```text
OLD = 978.60 MB
NEW = 128.78 MB

978.60 / 128.78 = 7.60×
```

NEW uses approximately:

**7.60× less peak RAM**

---

## Final RAM

```text
OLD = 566.87 MB
NEW = 121.63 MB

566.87 / 121.63 = 4.66×
```

NEW retains approximately:

**4.66× less RAM at completion**

---

# 12. Iteration-by-Iteration Summary

## ITERATION 1

```text
                    OLD          NEW          Improvement

Time             147.401 s     57.894 s       60.72%
CPU              150.530 s     59.220 s       60.65%
Peak RAM          977.42 MB     128.80 MB      86.82%
Final RAM         544.80 MB     121.65 MB      77.67%
```

NEW is:

```text
2.55× faster
7.59× lower peak RAM
4.48× lower final RAM
```

---

## ITERATION 2

```text
                    OLD          NEW          Improvement

Time             141.901 s     57.965 s       59.15%
CPU              144.970 s     59.250 s       59.15%
Peak RAM          979.20 MB     128.84 MB      86.84%
Final RAM         529.16 MB     121.69 MB      77.00%
```

NEW is:

```text
2.45× faster
7.60× lower peak RAM
4.35× lower final RAM
```

---

## ITERATION 3

```text
                    OLD          NEW          Improvement

Time             135.555 s     58.753 s       56.66%
CPU              138.360 s     60.070 s       56.58%
Peak RAM          979.19 MB     128.71 MB      86.85%
Final RAM         626.64 MB     121.56 MB      80.61%
```

NEW is:

```text
2.31× faster
7.61× lower peak RAM
5.16× lower final RAM
```

---

# 13. Interesting Observation: OLD Improves Across Iterations

There is an interesting pattern in the OLD implementation:

```text
OLD Execution Time

147.40 s
    ↓
141.90 s
    ↓
135.55 s
```

From ITER 1 to ITER 3, OLD becomes approximately:

```text
8.0% faster
```

NEW, in contrast, remains almost completely flat:

```text
NEW Execution Time

57.89 s
    ↓
57.96 s
    ↓
58.75 s
```

This indicates that NEW has much lower variability between runs.

The variation could potentially be related to factors such as:

- Runtime warm-up
- JIT compilation
- Garbage collection
- CPU cache behavior
- Memory allocation patterns
- Data structure reuse
- System-level scheduling

However, the benchmark alone does not establish which of these mechanisms is responsible.

The important observation is that the effect is measurable:

```text
OLD: larger run-to-run variation
NEW: very stable execution time
```

---

# 14. NEW vs the Best OLD Execution

A useful additional comparison is to compare the **best OLD execution** against the **worst NEW execution**.

Best OLD execution:

```text
135.55 s
```

Worst NEW execution:

```text
58.75 s
```

Even under this favorable comparison for OLD:

```text
135.55 / 58.75 ≈ 2.31×
```

NEW remains approximately:

```text
2.31× faster
```

This demonstrates that the improvement is not simply an artifact of comparing an unusually slow OLD iteration against an unusually fast NEW iteration.

---

# 15. Absolute Resource Savings

Average resource savings per simulation:

| Resource | OLD | NEW | Absolute Savings |
|---|---:|---:|---:|
| Execution Time | 141.62 s | 58.20 s | **83.41 s** |
| CPU Time | 144.62 s | 59.51 s | **85.11 s** |
| Peak RAM | 978.60 MB | 128.78 MB | **849.82 MB** |
| Final RAM | 566.87 MB | 121.63 MB | **445.24 MB** |

Therefore, every simulation approximately saves:

```text
83.41 seconds of wall-clock execution time
85.11 seconds of CPU time
849.82 MB of peak memory
445.24 MB of final retained memory
```

---

# 16. Combined Efficiency Perspective

A simple combined indicator can be obtained by multiplying:

```text
Peak RAM × Execution Time
```

This is not a standard performance metric and should not replace proper profiling, but it can help visualize the magnitude of the combined improvement.

For OLD:

```text
978.60 MB × 141.62 s
≈ 138,580 MB·s
```

For NEW:

```text
128.78 MB × 58.20 s
≈ 7,494 MB·s
```

Approximate reduction:

```text
1 - (7,494 / 138,580)
≈ 94.6%
```

Therefore:

```text
OLD: ≈ 138,580 MB·s
NEW: ≈   7,494 MB·s

Reduction: ≈ 94.6%
```

Again, this should be treated as an illustrative combined indicator rather than a formal performance metric.

---

# 17. Overall Results

```text
┌─────────────────────────────────────────────┐
│              OLD → NEW IMPROVEMENT          │
├─────────────────────────────────────────────┤
│                                             │
│  ⏱ Execution Time     ↓ 58.90%   2.43×      │
│  🖥 CPU Time           ↓ 58.85%   2.43×      │
│  🧠 Peak RAM           ↓ 86.84%   7.60×      │
│  🧠 Final RAM          ↓ 78.54%   4.66×      │
│  📉 Time Variability   ↓ ~92.8%              │
│                                             │
└─────────────────────────────────────────────┘
```

---

# 18. Final Assessment

The benchmark demonstrates a substantial improvement in the NEW implementation across all major measured dimensions.

## Performance

Execution time decreases from:

```text
141.62 s → 58.20 s
```

This represents:

```text
58.90% reduction
2.43× improvement
```

## CPU

CPU time decreases from:

```text
144.62 s → 59.51 s
```

This represents:

```text
58.85% reduction
2.43× improvement
```

## Peak Memory

Peak RAM decreases from:

```text
978.60 MB → 128.78 MB
```

This represents:

```text
86.84% reduction
7.60× improvement
```

## Final Memory

Final RAM decreases from:

```text
566.87 MB → 121.63 MB
```

This represents:

```text
78.54% reduction
4.66× improvement
```

## Stability

Execution-time variability decreases from approximately:

```text
8.36% → 1.47%
```

with the execution-time range decreasing from:

```text
11.85 s → 0.86 s
```

This corresponds to an approximate:

```text
92.8% reduction in execution-time range
```

---

# 19. Key Takeaways

1. **NEW is approximately 2.43× faster than OLD.**

2. **NEW reduces CPU time by approximately 58.85%.**

3. **NEW reduces peak RAM consumption by approximately 86.84%.**

4. **NEW uses approximately 7.60× less peak memory.**

5. **NEW retains approximately 78.54% less memory after the simulation.**

6. **NEW is significantly more consistent between iterations.**

7. **The initial RAM consumption is practically identical**, meaning the observed memory improvement comes from the execution behavior rather than from a different initial process state.

8. **The improvement remains significant even when comparing the best OLD execution against the worst NEW execution.**

9. The largest improvement is in memory efficiency, where NEW reduces peak usage from approximately **979 MB to 129 MB**.

10. The performance improvement is also substantial, reducing a typical simulation from approximately **142 seconds to 58 seconds**.

---

# 20. Final Benchmark Snapshot

```text
                     OLD              NEW

Execution Time       141.62 s         58.20 s
                     ██████████████████████████████████████████████████
                     ████████████████████

CPU Time             144.62 s         59.51 s
                     ██████████████████████████████████████████████████
                     ████████████████████

Peak RAM             978.60 MB        128.78 MB
                     ██████████████████████████████████████████████████
                     ███████

Final RAM            566.87 MB        121.63 MB
                     ██████████████████████████████████████████████████
                     ███████████
```

## Bottom Line

```text
NEW delivers:

        ~2.43× faster execution
        ~58.9% less execution time
        ~58.9% less CPU time
        ~86.8% less peak RAM
        ~78.5% less final RAM
        ~92.8% lower execution-time range
```

The benchmark therefore indicates that the NEW implementation is substantially more efficient in both **compute utilization** and **memory consumption**, while also providing significantly more stable execution times across repeated runs.

