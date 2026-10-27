import numpy as np

class LeakyReLUActivation:
    """Leaky ReLU activation function with configurable negative slope."""
    def __init__(self, slope=0.05):
        self.slope = slope
        self.mask = None

    def forward(self, z):
        self.mask = (z > 0)
        return np.where(self.mask, z, self.slope * z)

    def backward(self, delta):
        return np.where(self.mask, delta, self.slope * delta)

class CrossEntropyLossModule:
    """Fused categorical cross-entropy loss with softmax calculation."""
    @staticmethod
    def forward(logits, labels):
        m = logits.shape[0]
        c = np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(logits - c)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        loss = -np.sum(np.log(np.clip(probs[np.arange(m), labels], 1e-15, 1.0))) / m
        return loss, probs

    @staticmethod
    def backward(probs, labels):
        m = probs.shape[0]
        d_out = probs.copy()
        d_out[np.arange(m), labels] -= 1.0
        return d_out
