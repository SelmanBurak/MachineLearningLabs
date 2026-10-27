# Computational Environment Audit & Baseline Telemetry (Lab 01)
**Investigator:** Selman Burak Uygurtürk  
**Student ID:** 72922  
**Department:** Computer Engineering  

---

## 1. Investigative Target & Workstation Profile
Before deploying computationally intensive machine learning workloads, empirical profiling of workstation compute, memory bandwidth, and operating system scheduling is essential. This laboratory audits baseline execution times and memory utilization under heavy tensor operations.

## 2. Benchmarking Strategy & Software Stack
- **Software Stack:** Isolated Python 3.11 environment with verified NumPy BLAS binaries.
- **Hardware Profile:** 64-bit multi-threaded Windows workstation.
- **Workload Formulation:** Square matrix multiplications ($N \times N$) across dimension tiers $N \in \{256, 512, 1024, 2048\}$.
- **Measurement Instruments:** Monotonic high-resolution clock (`time.perf_counter()`) and memory tracker (`psutil`).

## 3. Systems Benchmark Metrics & Discussion

### Workstation Telemetry Profile
| Matrix Size ($N \times N$) | Floating-Point Operations | Latency (ms) | Measured GFLOPs | Memory Allocation (MB) |
| :---: | :---: | :---: | :---: | :---: |
| $256 \times 256$ | $3.36 \times 10^7$ | 1.80 | 18.6 | 0.51 |
| $512 \times 512$ | $2.68 \times 10^8$ | 12.05 | 22.2 | 2.06 |
| $1024 \times 1024$ | $2.15 \times 10^9$ | 84.80 | 25.3 | 8.30 |
| $2048 \times 2048$ | $1.72 \times 10^{10}$ | 636.80 | 27.0 | 33.00 |

```
Execution Latency Scaling Profile:
  256x256   :   1.80 ms [■]
  512x512   :  12.05 ms [■■■]
  1024x1024 :  84.80 ms [■■■■■■■■■■■]
  2048x2048 : 636.80 ms [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]
```

## 4. Three Systems Engineering Takeaways
1. **Vectorized Instruction Execution Delivers 27 GFLOPs:** Multi-threaded BLAS kernels achieve high arithmetic intensity on CPU vector units, outperforming interpreted Python loops by orders of magnitude.
2. **Cache Boundaries Induce Latency Non-Linearities:** As matrix buffers exceed CPU L3 cache capacity at $N=2048$, memory bus stalls slightly degrade scaling linearity.
3. **Deterministic Dependencies Prevent Environment Incompatibilities:** Pinned package dependencies prevent subtle performance regressions caused by unexpected library version shifts.
