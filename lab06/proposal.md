# Individual Project Proposal: Distilled Edge Image Classifier

**Student Name:** Selman Burak Uygurtürk  
**Student ID:** 72922  
**Course:** Machine Learning [CII5SE039CI] — Group 01  

---

## 1. Problem Statement
Standard convolutional neural networks such as ResNet-18 achieve strong classification accuracy on natural RGB images but exceed tight edge-deployment budgets (~44 MB in FP32). Conversely, training a tiny CNN from scratch to fit strict storage and latency constraints typically results in a noticeable accuracy drop. This project applies **Knowledge Distillation** to transfer dark knowledge from a high-capacity teacher model into a lightweight student CNN capable of fast, accurate on-device CPU inference.

## 2. Dataset
- **Dataset:** CIFAR-10 (60,000 RGB images of size $32 \times 32$ across 10 object classes).
- **Split:** 45,000 training / 5,000 validation / 10,000 held-out test images (stratified, fixed seed = 42).
- **Preprocessing & Augmentation:** Channel-wise standardization, random horizontal flip, and random crop with 4-pixel padding during training.

## 3. Architectures & Success Metrics
- **Teacher Model:** ResNet-18 adapted for $32 \times 32$ inputs (~11.2M parameters, ~44.7 MB FP32).
- **Student Model:** Compact 3-block depthwise/standard CNN with Batch Normalization (~200k–300k parameters, ~0.8–1.2 MB FP32).
- **Primary Accuracy Metric:** Top-1 Test Accuracy $\ge 82.0\%$, demonstrating a measurable improvement over the same student trained from scratch without distillation.
- **Systems Metrics:** Single-sample CPU inference latency (p95 over 100 timed runs after 20 warm-up passes) and serialized `.pt` model size on disk.

## 4. Deployment Constraints
| Constraint | Target Budget | Measurement Protocol |
| :--- | :---: | :--- |
| **Inference Latency** | $\le 50\text{ ms}$ | Batch size = 1, CPU-only, p95 wall-clock time |
| **Model Size** | $\le 5\text{ MB}$ | Serialized `state_dict` / TorchScript file on disk |
| **Target Hardware** | Edge CPU (ARM64 / x86) | No GPU acceleration required at inference |

##