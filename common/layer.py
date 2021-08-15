import numpy as np

from .math import Math
from .loss import Loss

class Layer:
    def forward(self, x):
        return x

    def backward(self, dout):
        return dout

class AddLayer(Layer):
    def forward(self, a, b):
        return a + b

    def backward(self, dout):
        return dout, dout

class MulLayer(Layer):
    def forward(self, a, b):
        self.a, self.b = a, b
        return a * b

    def backward(self, dout):
        return dout * self.b, dout * self.a

class ReLULayer(Layer):
    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0
        return y

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class SigmoidLayer(Layer):
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dout):
        return dout * self.y * (1.0 - self.y)

class AffineLayer(Layer):
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithCEELayer(Layer):
    def forward(self, x, t, axis=None):
        self.t = t
        self.y = Math.softmax(x, axis=axis)
        self.loss = Loss.cee(self.y, t) # Note that self.loss has only one value, not the batch size.
        return self.loss

    def backward(self, dout=None):
        # We have to divide the diff with batch size because loss function returns the average loss of all data in batch.
        return (self.y - self.t) / self.t.shape[0]