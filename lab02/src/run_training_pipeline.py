import os
import sys
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.feedforward import SelmanMultilayerPerceptron
from engine.gradient_validator import validate_gradients

def main():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=72922)

    mlp = SelmanMultilayerPerceptron(in_features=64, hidden_dims=(50, 25), num_classes=10, seed=72922)
    err = validate_gradients(mlp, X_tr[:32], y_tr[:32])
    print(f"[AUDIT] Gradient Validation Error: {err:.4e}")

    start = time.perf_counter()
    epochs = 40
    batch_size = 32
    for ep in range(epochs):
        perm = np.random.permutation(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            b_idx = perm[i:i+batch_size]
            mlp.forward(X_tr[b_idx], y_tr[b_idx])
            mlp.backward()
            mlp.step(lr=0.08)
    elapsed = time.perf_counter() - start

    preds = np.argmax(mlp.forward(X_val), axis=1)
    acc = np.mean(preds == y_val) * 100.0
    print(f"[PERF] Validation Accuracy: {acc:.2f}% | Wall Time: {elapsed:.3f}s")

if __name__ == "__main__":
    main()
