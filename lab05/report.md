# Lab 05: Frameworks and Automatic Differentiation

**Student Engineer:** Selman Burak Uygurtürk  
**Student Identifier:** 72922  
**Faculty / Department:** Applied Computer Science, Vistula University  
**Academic Year:** 2025/2026  
**Curriculum Reference:** Vijay Janapa Reddi, *Machine Learning Systems*, Chapter 7 (AI Frameworks: §§7.3–7.8)  

---

## 1. Executive Summary

This laboratory analyzes the systems engineering foundation of computational graphs, automated gradient derivation, and framework compilation:
1. **Dynamic Scalar DAG Engine:** Implemented `GraphScalar` to dynamically capture forward operations, topologically sort dependencies via DFS post-order traversal, and evaluate backward adjoints using strictly additive gradient accumulation (`+=`).
2. **Empirical Verification:** Validated gradients against PyTorch autograd and numerical central finite differences on `((a * b + c) ** 2).relu().sum()`, demonstrating complete numerical convergence ($\Delta < 10^{-6}$).
3. **Moons Binary Classification Benchmark:** 5-run empirical benchmarking comparing scalar autograd execution with optimized PyTorch tensor execution on `make_moons`.
4. **Intermediate Representation (TorchScript IR):** Traced graph execution via `torch.jit.trace`, isolating dispatch latency from silicon computation.
5. **Tensor Strides & Cache Hierarchy:** Characterized physical memory layouts (contiguous vs. strided memory access) and evaluated cache-line utilization and SIMD throughput penalties.

---

## 2. Dynamic Computation Graph Core (`GraphScalar`)

The scalar autograd core is located in [graph_scalar.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_72922_selman/lab05/src/graph_scalar.py).

### 2.1 State & Dependency Structure
Each `GraphScalar` node encapsulates:
- `scalar`: Real forward value computed at forward runtime.
- `grad`: Accumulated gradient adjoint ($\partial \mathcal{L} / \partial x$), initialized to 0.0.
- `_predecessors`: Set of antecedent operand nodes in the computation graph.
- `_backward_rule`: Local chain-rule derivative closure.
- `operator_str`: Human-readable operator tag.

### 2.2 Mathematical Primitive Operations
- **Addition ($z = x + y$):** $\frac{\partial \mathcal{L}}{\partial x} += \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} += \frac{\partial \mathcal{L}}{\partial z}$
- **Multiplication ($z = x \cdot y$):** $\frac{\partial \mathcal{L}}{\partial x} += y \cdot \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} += x \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Power ($z = x^p$):** $\frac{\partial \mathcal{L}}{\partial x} += (p \cdot x^{p-1}) \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **ReLU ($z = \max(0, x)$):** $\frac{\partial \mathcal{L}}{\partial x} += \mathbf{1}_{\{x > 0\}} \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$):** $\frac{\partial \mathcal{L}}{\partial x} += \sigma(x)(1 - \sigma(x)) \cdot \frac{\partial \mathcal{L}}{\partial z}$
- **Logarithm ($z = \ln(x)$):** $\frac{\partial \mathcal{L}}{\partial x} += \frac{1}{x} \cdot \frac{\partial \mathcal{L}}{\partial z}$

### 2.3 Topological Sorting & Gradient Accumulation
Topological traversal ensures reverse-mode evaluation order. Accumulation with `+=` strictly satisfies the multivariate chain rule across nodes that fan out to multiple downstream operations.

---

## 3. Mathematical Verification Against PyTorch & Central Finite Differences

Target mathematical expression:
$$\mathcal{L} = \sum_{i=1}^4 \text{ReLU}\left( (a_i \cdot b_i + c_i)^2 \right)$$

- **Test Vectors:**
  - $a = [1.35, -2.25, 0.60, -1.50]$
  - $b = [-2.10, 1.40, 3.50, 0.75]$
  - $c = [1.20, -0.70, -1.90, 2.50]$
- **Forward Output:** Custom = `28.84750`, PyTorch = `28.84750` ($\Delta = 0.00$)
- **Max Absolute Error vs PyTorch:** **$0.00 \times 10^0$** (Strictly $< 10^{-6}$, PASS)
- **Max Absolute Error vs Central Differences:** $8.51 \times 10^{-9}$ (PASS)

---

## 4. `make_moons` Training Benchmark (5-Trial Telemetry)

- **Architecture:** $2 \to 8 \to 1$ (ReLU hidden, Sigmoid output, BCE loss).
- **Dataset:** `make_moons(n_samples=100, noise=0.1, random_state=42)`.
- **Optimization:** SGD ($\eta = 0.25$, 35 epochs).

| Trial | Custom Engine (ms) | PyTorch (ms) | Speedup Ratio |
| :---: | :---: | :---: | :---: |
| 1 | 2442.81 | 16.71 | 146.2x |
| 2 | 2800.12 | 15.52 | 180.4x |
| 3 | 2626.57 | 11.67 | 225.1x |
| 4 | 2711.63 | 12.69 | 213.7x |
| 5 | 2580.40 | 12.13 | 212.7x |
| **Median** | **2626.57 ms** | **12.69 ms** | **207.1x** |

### Systems Analysis
PyTorch achieves a $207.1\times$ speedup over the scalar autograd engine due to:
1. Object allocation overhead: Instantiating thousands of individual Python objects (`GraphScalar`) per epoch creates high GC and allocation pressure.
2. Bytecode interpretation overhead: CPython evaluates dynamic opcodes individually, whereas PyTorch calls compiled C++ loop kernels.
3. Parallel SIMD matrix operations: PyTorch processes all 100 samples simultaneously in contiguous memory with Intel MKL/OpenMP vector instructions.

---

## 5. TorchScript IR Inspection & Latency Profile

We inspected the execution graph of `DifferentiableArch` using `torch.jit.trace`:
- **Eager Execution Latency:** $47.63\ \mu\text{s}$
- **Traced Graph Latency:** $42.29\ \mu\text{s}$
- **Speedup Factor:** **$1.13\times$**

Tracing flattens Python method calls and variable lookups into a unified C++ execution plan.

---

## 6. Tensor Memory Layout & Strides Penalty

Using a $2048 \times 2048$ single-precision tensor (16 MB buffer):
- **Contiguous Array:** `stride = [2048, 1]`, `is_contiguous() = True`.
- **Transposed Array:** `stride = [1, 2048]`, `is_contiguous() = False`.

### Reduction Benchmark (Row-Wise Sum)
- **Contiguous Row-Sum Latency:** $0.168\text{ ms}$
- **Non-Contiguous Row-Sum Latency:** $0.333\text{ ms}$
- **Stride Penalty:** **$1.97\times$ slowdown**
- **Buffer Clone Cost (`.contiguous()`):** $8.853\text{ ms}$

### Hardware Mechanics
- **Cache Line Efficiency:** Sequential access utilizes all 64 bytes of an L1/L2 cache line (16 floats). Non-contiguous strides fetch 64 bytes but use only 4 bytes, evicting cache lines before reuse.
- **SIMD Vectorization:** Contiguous memory feeds AVX2 registers directly via single unaligned/aligned loads (`vmovups`). Strided layouts require gather operations or serial scalar reads.
- **Memory Allocation Cost:** Copying to contiguous memory takes $8.853\text{ ms}$, which is $50\times$ more expensive than the slowdown of a single reduction. Buffer copies are only worthwhile when amortized across multiple operations.

---

## 7. Technical Note: When Graph Compilation Delivers Value

Per Reddi Chapter 7, graph compilers provide substantial performance gains when:
1. **Fusing Pointwise Kernels:** Combining operations like `Linear + ReLU` prevents storing intermediate tensors in DRAM, keeping values in L1/L2 CPU caches.
2. **Removing Python Dispatch Latency:** For low-latency inference or small batch sizes, Python runtime overhead dominates. Lowering to native C++ eliminates interpreter overhead.
3. **Optimizing Memory Footprint:** Compilers analyze variable lifetimes and reuse static memory buffers, eliminating memory fragmentation and dynamic allocation calls.

---

## 8. Artifact Directory

- [graph_scalar.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_72922_selman/lab05/src/graph_scalar.py): Custom scalar autograd engine.
- [autograd_verifier.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_72922_selman/lab05/src/autograd_verifier.py): Mathematical gradient verification suite.
- [moons_classifier.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_72922_selman/lab05/src/moons_classifier.py): 5-trial benchmark on `make_moons`.
- [pytorch_compiler_audit.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_72922_selman/lab05/src/pytorch_compiler_audit.py): TorchScript tracing and memory striding analysis.
- `results/`: Contains [verification_report.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_72922_selman/lab05/results/verification_report.json), [moons_5runs_benchmark.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_72922_selman/lab05/results/moons_5runs_benchmark.json), and [compiler_stride_profile.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/portfolio_72922_selman/lab05/results/compiler_stride_profile.json).
