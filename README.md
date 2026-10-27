# Machine Learning Systems Engineering: Performance, Architectures & Pipelines

**Student Engineer:** Selman Burak Uygurtürk  
**Student Identifier:** 72922  
**Faculty / Department:** Applied Computer Science, Vistula University  
**Academic Year:** 2025/2026  
**Foundational Curriculum:** *Machine Learning Systems* (Vijay Janapa Reddi, Harvard University)  

---

## Executive Architecture Summary

This repository organizes four modular engineering investigations exploring modern machine learning through a systems lens: physical hardware constraints, mathematical matrix calculus, compute complexity (FLOPs/MACs), and industrial data engineering standards.

```
MachineLearningLabs/
├── README.md               # Systems architecture blueprint and replication guidelines
├── requirements.txt        # Pinned dependency manifest ensuring strict reproducibility
├── .gitignore              # Production-grade build cache and binary exclusions
├── lab01/                  # Laboratory 01: Systems Profiling & Environment Telemetry
│   ├── report.md           # Systems measurement audit & deployment tier classification
│   ├── src/                # Model fitting scripts, latency probes, and memory monitor
│   └── results/            # Profiling CSVs, saved models, and runtime specs
├── lab02/                  # Laboratory 02: Vectorized Deep Learning from First Principles
│   ├── report.md           # Analytical calculus, finite-difference verification, training curves
│   ├── src/                # Modular NumPy network without autodiff dependencies
│   └── results/            # Multi-parameter convergence logs, loss plots, timing metrics
├── lab03/                  # Laboratory 03: Architectural Topologies and Execution Budgets
│   ├── report.md           # MLP vs CNN vs Attention trade-offs, theoretical FLOPs vs MACs
│   ├── src/                # PyTorch architecture declarations and benchmarking scripts
│   └── results/            # Pareto frontier plots, latency traces, parameter tables
└── lab04/                  # Laboratory 04: Industrial Data Engineering & Quality Pipeline
    ├── report.md           # Zero-leakage ETL validation, feature synthesis, label noise audit
    ├── src/                # Production pipeline orchestrator and transformation modules
    └── results/            # Data card, validation telemetry, serialized joblib artifacts
```

## Reproduction Instructions

```bash
# 1. Environment initialization
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Dependency installation
pip install -r requirements.txt
```
