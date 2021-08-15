import pickle
import numpy as np
from origin.dataset.mnist import load_mnist

from .math import Math

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

        with open("origin/ch03/sample_weight.pkl", "rb") as f:
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

if __name__ == "__main__":
    n = PretrainedNet()
    n.predict()
    n.predict_batch()