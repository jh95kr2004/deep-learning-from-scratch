import pickle
from typing import OrderedDict
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

class Loss:
    delta = 1e-7

    @staticmethod
    def sse(y, t) -> float:
        ''' Returns a loss of result using sum of squares for error
        '''
        return 0.5 * np.sum((y-t) ** 2)

    @staticmethod
    def cee(y, t) -> float:
        ''' Returns a loss of result using cross entropy error
        '''
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)

        if y.size == t.size:
            # [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]] => [1, 2]
            t = t.argmax(axis=1)

        # from here, t should not be a one-hot encoded answer
        # so, t.shape should be (batch_size,)
        batch_size = y.shape[0]

        # comma(,) means extract specific column from 2-dim array

        # a = np.array(
        #     [[11, 12, 13, 14],
        #     [21, 22, 23, 24],
        #     [31, 32, 33, 34],
        #     [41, 42, 43, 44],
        #     [51, 52, 53, 54],
        #     [61, 62, 63, 64],
        #     [71, 72, 73, 74],
        #     [81, 82, 83, 84],
        #     [91, 92, 93, 94]]
        # )

        # a[:, 0] => array([11, 21, 31, 41, 51, 61, 71, 81, 91])

        return -np.sum(np.log(y[np.arange(batch_size), t] + Loss.delta)) / batch_size

class Grad:
    h = 1e-4

    @staticmethod
    def numerical_diff(f, x):
        return (f(x+Grad.h) - f(x-Grad.h)) / (2 * Grad.h)

    @staticmethod
    def numerical_grad(f, x):
        ''' Calculate numerical gradient of function f at x
        '''
        grad = np.zeros_like(x)

        with np.nditer(x, flags=["multi_index"], op_flags=["readwrite"]) as it:
            for _ in it:
                i = it.multi_index
                tmp = x[i]

                x[i] = tmp + Grad.h
                y1 = f(x)

                x[i] = tmp - Grad.h
                y2 = f(x)

                grad[i] = (y1 - y2) / (2 * Grad.h)
                x[i] = tmp

        return grad

    @staticmethod
    def grad_descent(f, x, lr=0.01, steps=100):
        ''' Calculate gradient descent
        - lr: learning rate
        '''
        for _ in range(steps):
            grad = Grad.numerical_grad(f, x)
            x -= lr * grad
        return x

class Math:
    @staticmethod
    def sigmoid(x):
        ''' Calculate sigmoid of x.
        '''
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x, axis=None):
        ''' Calculate softmax of input. Set axis if x is a batch form data.
        '''
        x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return x / np.sum(x, axis=axis, keepdims=True)

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

class PretrainedNet:
    ''' Pretrained network of ch03
    '''

    def __init__(self):
        ''' Initialze network
        '''
        # - x_train: training set (60000, 784)
        # - t_train: answer of training set (60000,)
        # - x_test: test set (10000, 784)
        # - t_test: answer of test set (10000,)
        (self.x_train, self.t_train), (self.x_test, self.t_test) = \
            load_mnist(normalize=True, flatten=True, one_hot_label=False)

        with open("ch03/sample_weight.pkl", "rb") as f:
            self.network = pickle.load(f)

    def forward(self, x, axis=None):
        a1 = np.dot(x, self.network["W1"]) + self.network["b1"]
        z1 = Math.sigmoid(a1)
        a2 = np.dot(z1, self.network["W2"]) + self.network["b2"]
        z2 = Math.sigmoid(a2)
        a3 = np.dot(z2, self.network["W3"]) + self.network["b3"]

        return Math.softmax(a3, axis=axis)

    def predict(self):
        ans_cnt = 0

        for i in range(len(self.x_test)):
            x = self.x_test[i]
            y = self.forward(x)
            p = np.argmax(y)

            if p == self.t_test[i]:
                ans_cnt += 1

        print("Accuracy: ", ans_cnt / len(self.x_test))

    def predict_batch(self, batch_size=100):
        ans_cnt = 0

        for i in range(0, len(self.x_test), batch_size):
            x_batch = self.x_test[i:i+batch_size]
            y_batch = self.forward(x_batch, axis=1)
            p_batch = np.argmax(y_batch, axis=1)
            ans_cnt += np.sum(p_batch == self.t_test[i:i+batch_size])

        print("Accuracy: ", ans_cnt / len(self.x_test))

class TwoLayerNetNumerical:
    ''' Two layer net of ch04
    '''

    def __init__(self, hidden_size, weight_init_std=0.01):
        ''' Initialze network
        '''
        # - x_train: training set (60000, 784)
        # - t_train: answer of training set (60000,)
        # - x_test: test set (10000, 784)
        # - t_test: answer of test set (10000,)
        (self.x_train, self.t_train), (self.x_test, self.t_test) = \
            load_mnist(normalize=True, flatten=True, one_hot_label=True)

        self.params = dict()
        self.params["W1"] = weight_init_std * np.random.randn(self.x_test.shape[1], hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, self.t_test.shape[1])
        self.params["b2"] = np.zeros(self.t_test.shape[1])

    def predict(self, x):
        ''' Get prediction of x
        '''
        a1 = np.dot(x, self.params["W1"]) + self.params["b1"]
        z1 = Math.sigmoid(a1)
        a2 = np.dot(z1, self.params["W2"]) + self.params["b2"]
        return Math.softmax(a2, axis=1)

    def loss(self, x, t):
        ''' Get loss of x with answer t
        '''
        return Loss.cee(self.predict(x), t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / x.shape[0]

    def grad(self, x, t):
        return self.numerical_grad(x, t)

    def numerical_grad(self, x, t):
        loss = lambda _: self.loss(x, t)

        # We will calculate gradient with loss function.
        # Gradient of each parameters will be used for learning.
        # By subtracting grads, the value of loss will be reduced.
        #
        # Grad.numerical_grad will calculate grads of params by modifying them internally.
        # (self.params will be modified temporarily in it.)
        # So, prediction should be calculated when the loss function is called.

        grads = dict()
        grads["W1"] = Grad.numerical_grad(loss, self.params["W1"])
        grads["b1"] = Grad.numerical_grad(loss, self.params["b1"])
        grads["W2"] = Grad.numerical_grad(loss, self.params["W2"])
        grads["b2"] = Grad.numerical_grad(loss, self.params["b2"])

        return grads

    def learn(self, batch_size=100, lr=0.1, iters=10000):
        train_size = self.x_train.shape[0]
        epoch = max(train_size // batch_size, 1)
        train_accs = []
        test_accs = []

        for i in range(iters):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]

            grad = self.grad(x_batch, t_batch)

            self.params["W1"] -= lr * grad["W1"]
            self.params["b1"] -= lr * grad["b1"]
            self.params["W2"] -= lr * grad["W2"]
            self.params["b2"] -= lr * grad["b2"]

            if i % epoch == 0:
                train_acc = self.accuracy(self.x_train, self.t_train)
                test_acc = self.accuracy(self.x_test, self.t_test)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

        x = np.arange(len(train_accs))

        plt.plot(x, train_accs, label='train acc')
        plt.plot(x, test_accs, label='test acc', linestyle='--')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

class TwoLayerNet(TwoLayerNetNumerical):
    ''' Two layer net of ch05
    '''

    def __init__(self, hidden_size, weight_init_std=0.01):
        ''' Initialze network
        '''
        super().__init__(hidden_size, weight_init_std)

        self.layers = dict()
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
        x_batch = self.x_train[:3]
        t_batch = self.t_train[:3]

        grad_numertical = self.numerical_grad(x_batch, t_batch)
        grad_backprop = self.grad(x_batch, t_batch)

        for key in grad_numertical.keys():
            diff = np.average(np.abs(grad_numertical[key] - grad_backprop[key]))
            print(key + ": ", diff)

# ch03
n = PretrainedNet()
n.predict()
n.predict_batch()

# ch04
# n = TwoLayerNetNumerical(hidden_size=50)
# n.learn()

# ch05
n = TwoLayerNet(hidden_size=50)
n.learn()
n.grad_check()