import numpy as np

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