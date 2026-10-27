# Comparative Cost-Efficiency Investigation of Neural Topologies (Lab 03)
**Investigator:** Selman Burak Uygurtürk  
**Student ID:** 72922  
**Department:** Computer Engineering  
**Course:** Machine Learning Systems Engineering  

---

## 1. Project Goal & Investigative Scope
Modern machine learning applications demand a precise alignment between model algorithmic expressiveness and hardware inference efficiency. Deploying overly deep or computationally ill-conditioned networks results in wasted compute cycles, thermal throttling on edge devices, and unacceptable latency.

The central goal of this laboratory study is to conduct an empirical multi-attribute efficiency audit across three distinctive architectural paradigms on the Fashion-MNIST dataset:
1. **Fully Connected Multilayer Perceptron (`SelmanMLP`)**: Flat dense feature mapping with Gaussian Error Linear Units (GELU).
2. **Hierarchical Convolutional Model (`SelmanCNN`)**: Localized 2D spatial feature learning with Batch Normalization.
3. **8-Head Vision Transformer (`SelmanViT`)**: Non-local self-attention mechanism over visual patch tokens.

By auditing parameter volume, theoretical operation counts (FLOPs/MACs), wall-clock latency, and classification accuracy under standardized hardware constraints, this study establishes rigorous deployment guidelines.

---

## 2. Experimental Setup and Architecture Design

### Structural Architectures
```
[1] SelmanMLP:
    Input (28x28) ──> Flatten (784) ──> Linear(784, 320) ──> GELU 
                                    ──> Linear(320, 80) ──> GELU 
                                    ──> Linear(80, 10)

[2] SelmanCNN:
    Input (1x28x28) ──> Conv2d(1->32, k=3, p=1) ──> BatchNorm ──> ReLU ──> MaxPool2d(2x2)
                    ──> Conv2d(32->64, k=3, p=1) ──> BatchNorm ──> ReLU ──> AdaptiveAvgPool2d(3x3)
                    ──> Flatten (576) ──> Linear(576, 128) ──> ReLU ──> Linear(128, 10)

[3] SelmanViT:
    Input (1x28x28) ──> Conv2d(1->64, k=7, s=7) ──> 16 Tokens (dim=64) + Positional Encodings
                    ──> 2x TransformerEncoderLayers (d_model=64, nhead=8, d_ff=128)
                    ──> LayerNorm ──> Mean Token Pooling ──> Linear(64, 10)
```

### Measurement Methodology
To isolate true inference overhead, benchmarking scripts executed 15 initial burn-in passes followed by 80 timed iterations using high-precision timers. FLOPs and Multiply-Accumulate operations were quantified via `thop` profiling.

---

## 3. Empirical Benchmarks & Systems Diagnostics

### Comprehensive Performance Profile

| Model Architecture | Parameters | FLOPs / MACs | Inference Latency | Peak VRAM/RAM | Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GELU Perceptron (`SelmanMLP`)** | 277,690 | 0.55 MFLOPs (276k MACs) | 0.36 ms | 1.95 MB | 88.60% |
| **Batch-Normalized CNN (`SelmanCNN`)** | 94,858 | 5.24 MFLOPs (2.61M MACs) | 0.96 ms | 2.80 MB | 91.52% |
| **8-Head Vision Transformer (`SelmanViT`)** | 72,010 | 2.14 MFLOPs (1.06M MACs) | 1.45 ms | 4.95 MB | 85.35% |

### Bottleneck Diagnostic: Memory Access Cost (MAC) vs Raw Compute
A key finding in systems engineering is that **FLOP count alone is an insufficient proxy for wall-clock latency**:
- **Why CNN Outperforms ViT in Latency despite Higher FLOPs:** The Batch-Normalized CNN requires $5.24$ MFLOPs, whereas the 8-Head ViT requires only $2.14$ MFLOPs ($2.45\times$ fewer operations). However, the CNN runs in **0.96 ms** compared to the ViT's **1.45 ms** (a $51\%$ increase in wall-clock latency for the ViT).
- **Explanation:** Convolutions exhibit high arithmetic intensity and sequential stride memory access that map optimally onto CPU SIMD instructions and cache lines. In contrast, the Vision Transformer incurs substantial memory bandwidth penalties from token reshaping, sequence transpositions, and the generation of dynamic $Q K^T$ attention matrices.

### Profiler Accounting Convention (FLOPs vs. MACs)
In modern hardware architectures, floating-point arithmetic is evaluated via:
$$\text{FLOPs} = 2 \times \text{MACs}$$
A single fused Multiply-Accumulate operation counts as two separate floating-point operations (one multiply, one add). Profilers reporting MACs must be multiplied by 2 to reconcile with theoretical FLOP counts.

---

## 4. Three Primary Systems Conclusions

1. **Spatial Convolutions Deliver Optimal Pareto Efficiency for Vision:**  
   The Batch-Normalized CNN achieved the highest accuracy (**91.52%**) while maintaining a compact parameter footprint (**94,858** parameters). The spatial inductive bias of shared kernels and local receptive fields is indispensable for vision tasks without billions of pre-training tokens.

2. **Attention Mechanisms Incur Severe Cache and Memory Latency Penalties:**  
   At small visual resolutions ($28\times 28$), the Vision Transformer exhibits higher wall-clock latency (**1.45 ms**) than the CNN despite performing fewer than half the arithmetic operations. Developers must not assume lower theoretical FLOPs automatically translate into faster execution.

3. **Dense MLPs Excel in Raw Throughput but Struggle with Parameter Redundancy:**  
   While the MLP demonstrated the fastest inference time (**0.36 ms**) due to straightforward GEMM BLAS acceleration, it required nearly $3\times$ more parameters (**277,690**) than the CNN to achieve lower classification accuracy (**88.60%**).
