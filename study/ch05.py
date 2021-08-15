from typing import OrderedDict
import numpy as np

from .layer import AffineLayer, ReLULayer, SoftmaxWithCEELayer
from .ch04 import TwoLayerNet as TwoLayerNetNumerical

class TwoLayerNet(TwoLayerNetNumerical):
    ''' Two layer net of ch05
    '''

    def __init__(self, hidden_size, weight_init_std=0.01):
        ''' Initialze network
        '''
        super().__init__(hidden_size, weight_init_std)

        self.layers = OrderedDict()
        self.layers["Affine1"] = AffineLayer(self.params["W1"], self.params["b1"])
        self.layers["ReLU1"] = ReLULayer()
        self.layers["Affine2"] = AffineLayer(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithCEELayer()

    def predict(self, x):
        ''' Get prediction of x
        '''
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        ''' Get loss of x with answer t
        '''
        return self.lastLayer.forward(self.predict(x), t, axis=1)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        return np.sum(y == t) / x.shape[0]

    def grad(self, x, t):
        self.loss(x, t)

        layers = list(self.layers.values())
        layers.reverse()

        dout = self.lastLayer.backward()

        for layer in layers:
            dout = layer.backward(dout)

        grads = dict()
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads

    def grad_check(self):
        print("-- Check gradient --")

        x_batch = self.x_train[:3]
        t_batch = self.t_train[:3]

        grad_numertical = self.numerical_grad(x_batch, t_batch)
        grad_backprop = self.grad(x_batch, t_batch)

        for key in grad_numertical.keys():
            diff = np.average(np.abs(grad_numertical[key] - grad_backprop[key]))
            print(key + ": ", diff)

if __name__ == "__main__":
    n = TwoLayerNet(hidden_size=50)
    n.learn()
    n.grad_check()