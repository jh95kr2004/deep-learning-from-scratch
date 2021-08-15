import numpy as np

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