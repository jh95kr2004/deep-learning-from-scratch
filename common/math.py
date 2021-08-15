import numpy as np

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