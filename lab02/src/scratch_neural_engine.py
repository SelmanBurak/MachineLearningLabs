import numpy as np

class LinearTransform:
    def __init__(self, in_features, out_features, seed=42):
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((1, out_features))
        self.grad_W = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.bias)
        self.inputs = None

    def forward(self, X):
        self.inputs = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, dOut):
        N = self.inputs.shape[0]
        self.grad_W = np.dot(self.inputs.T, dOut) / N
        self.grad_b = np.sum(dOut, axis=0, keepdims=True) / N
        return np.dot(dOut, self.weights.T)

    def update(self, lr):
        self.weights -= lr * self.grad_W
        self.bias -= lr * self.grad_b

class LeakyReLUActivation:
    def __init__(self, slope=0.01):
        self.slope = slope
        self.mask = None

    def forward(self, Z):
        self.mask = (Z > 0)
        return np.where(self.mask, Z, self.slope * Z)

    def backward(self, dOut):
        return np.where(self.mask, dOut, self.slope * dOut)

class DeepMLPClassifier:
    def __init__(self, in_dim, hidden_dim, out_dim, seed=42):
        self.fc1 = LinearTransform(in_dim, hidden_dim, seed=seed)
        self.act1 = LeakyReLUActivation(slope=0.01)
        self.fc2 = LinearTransform(hidden_dim, hidden_dim, seed=seed+1)
        self.act2 = LeakyReLUActivation(slope=0.01)
        self.fc3 = LinearTransform(hidden_dim, out_dim, seed=seed+2)

    def forward(self, X, Y=None):
        out1 = self.fc1.forward(X)
        act1 = self.act1.forward(out1)
        out2 = self.fc2.forward(act1)
        act2 = self.act2.forward(out2)
        logits = self.fc3.forward(act2)

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        loss = None
        if Y is not None:
            eps = 1e-12
            loss = -np.mean(np.sum(Y * np.log(probs + eps), axis=1))

        self.probs = probs
        self.Y = Y
        return logits, probs, loss

    def backward(self):
        dZ3 = self.probs - self.Y
        dH2 = self.fc3.backward(dZ3)
        dZ2 = self.act2.backward(dH2)
        dH1 = self.fc2.backward(dZ2)
        dZ1 = self.act1.backward(dH1)
        dX = self.fc1.backward(dZ1)
        return dX

    def step(self, lr):
        self.fc1.update(lr)
        self.fc2.update(lr)
        self.fc3.update(lr)
