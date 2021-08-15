import numpy as np

class Optimizer:
    def update(self, params: dict, grads: dict):
        pass

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params: dict, grads: dict):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = 0.01
        self.momentum = momentum
        self.v = None

    def update(self, params: dict, grads: dict):
        if self.v is None:
            self.v = dict()
            for key, val in params.values():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]