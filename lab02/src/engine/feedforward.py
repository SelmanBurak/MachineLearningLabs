import numpy as np
from .primitives import DenseTensorTransform
from .nonlinearities import LeakyReLUActivation, CrossEntropyLossModule

class SelmanMultilayerPerceptron:
    """Multi-layer feed-forward network with LeakyReLU non-linearities."""
    def __init__(self, in_features=64, hidden_dims=(50, 25), num_classes=10, seed=72922):
        self.fc1 = DenseTensorTransform(in_features, hidden_dims[0], seed=seed)
        self.act1 = LeakyReLUActivation(slope=0.05)
        self.fc2 = DenseTensorTransform(hidden_dims[0], hidden_dims[1], seed=seed + 1)
        self.act2 = LeakyReLUActivation(slope=0.05)
        self.fc3 = DenseTensorTransform(hidden_dims[1], num_classes, seed=seed + 2)
        self.last_probs = None
        self.last_labels = None

    def forward(self, x, labels=None):
        h1 = self.act1.forward(self.fc1.forward(x))
        h2 = self.act2.forward(self.fc2.forward(h1))
        logits = self.fc3.forward(h2)

        if labels is not None:
            loss, probs = CrossEntropyLossModule.forward(logits, labels)
            self.last_probs = probs
            self.last_labels = labels
            return loss, probs
        c = np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(logits - c)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def backward(self):
        d_logits = CrossEntropyLossModule.backward(self.last_probs, self.last_labels)
        d_h2 = self.fc3.backward(d_logits)
        d_z2 = self.act2.backward(d_h2)
        d_h1 = self.fc2.backward(d_z2)
        d_z1 = self.act1.backward(d_h1)
        self.fc1.backward(d_z1)

    def step(self, lr):
        self.fc1.step(lr)
        self.fc2.step(lr)
        self.fc3.step(lr)
