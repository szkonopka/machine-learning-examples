import numpy as np

class AdalineGDescent(object):
    def __init__(self, eta_factor = 0.01, iterations = 10):
        self.eta_factor = eta_factor
        self.iterations = iterations

    def learning(self, X, Y):
        self.weights_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []
        for i in range(self.iterations):
            output = self.net_input(X)
            errors = (Y - output)
            self.weights_[1:] += self.eta_factor * X.T.dot(errors)
            self.weights_[0] += self.eta_factor * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def activation(self, X):
        self.net_input(X)

    def predict_label(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

    def net_input(self, X):
        return np.dot(X, self.weights_[1:]) + self.weights_[0]
