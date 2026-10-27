import numpy as np

class DenseTensorTransform:
    """Vectorized linear transformation with Xavier normal initialization."""
    def __init__(self, in_features, out_features, seed=72922):
        rng = np.random.RandomState(seed)
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weights = rng.normal(0.0, std, (in_features, out_features))
        self.bias = np.zeros((1, out_features))
        self.grad_W = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.bias)
        self.cached_x = None

    def forward(self, x):
        self.cached_x = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, delta):
        m = self.cached_x.shape[0]
        self.grad_W = np.dot(self.cached_x.T, delta) / m
        self.grad_b = np.sum(delta, axis=0, keepdims=True) / m
        return np.dot(delta, self.weights.T)

    def step(self, alpha):
        self.weights -= alpha * self.grad_W
        self.bias -= alpha * self.grad_b
