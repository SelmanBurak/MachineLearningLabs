# Production-Grade Data Pipelines & Quality Verification (Lab 04)
**Investigator:** Selman Burak Uygurtürk  
**Student ID:** 72922  
**Department:** Computer Engineering  

---

## 1. Project Goal & Data Quality Objectives
Machine learning performance in real-world deployments is fundamentally limited by data hygiene and engineering. This study focuses on constructing an end-to-end, leakage-free data processing pipeline that systematically handles missing entries, eliminates extreme outliers, and generates informative polynomial interaction features.

## 2. Robust Pipeline Architecture & Imputation Protocol
```
Raw Ingestion ──> Missingness Audit ──> Median Imputation ──> IQR Outlier Capping 
              ──> Polynomial Interactions ──> Robust Scaling ──> Evaluator
```
- **Leakage Prevention Guarantee:** All statistical transformations (medians, IQR fences, scaling standard deviations) are computed strictly on the training partition and frozen before transforming out-of-fold samples.
- **Feature Generation:** Engineered interaction products and non-linear ratios to create explicit decision boundaries for downstream classifiers.

## 3. Performance Uplift & Cross-Validation Results

### Pipeline Progress Benchmarks
| Processing Step | Raw Baseline | Cleaned + Imputed | Feature Engineered Pipeline | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Model Classification Accuracy** | 71.70% | 76.00% | 83.40% | +16.3% |
| **ROC-AUC Metric** | 0.766 | 0.817 | 0.888 | +15.9% |
| **Pipeline Latency (ms)** | 1.12 | 8.05 | 13.95 | — |
| **Total Features Produced** | 8 | 10 | 22 | +175% |

## 4. Three Primary Data Systems Conclusions
1. **Information Leakage Inflaters Validation Metrics:** Fitting scaling models on un-split data introduces subtle label leakage that leads to disastrous model drops during live deployment.
2. **Domain-Specific Interaction Features Enhance Accuracy:** Creating non-linear feature ratios increased the ROC-AUC score from 0.766 to 0.888, highlighting that feature engineering provides larger performance gains than complex model tuning.
3. **Missingness Indicator Masks Retain Signal:** Appending binary flags to denote imputed values preserved crucial collection signals that improved tree splitting efficiency.
