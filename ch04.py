import numpy as np
import matplotlib.pyplot as plt
from origin.dataset.mnist import load_mnist

from common.math import Math
from common.loss import Loss
from common.grad import Grad

class TwoLayerNet:
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
                print(i, "\ttrain acc, test acc | " + str(train_acc) + ", " + str(test_acc))

        x = np.arange(len(train_accs))

        plt.plot(x, train_accs, label='train acc')
        plt.plot(x, test_accs, label='test acc', linestyle='--')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

if __name__ == "__main__":
    n = TwoLayerNet(hidden_size=50)
    n.learn()