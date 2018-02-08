import numpy as np

class Perceptron(object):

    def __init__(self, eta_factor, iterations):
        self.eta_factor = eta_factor
        self.iterations = iterations

    def learning(self, X, Y):
        self.weights_ = np.zeros(X.shape[1] + 1)
        self.errors_ = []
        for i in range(self.iterations):
            errors = 0
            for xj, expected_label in zip(X, Y):
                weights_delta = self.eta_factor * (expected_label - self.predict_label(xj))
                self.weights_[1:] += weights_delta * xj
                self.weights_[0] += weights_delta
                errors += int(weights_delta != 0.0)
            self.errors_.append(errors)

    def predict_label(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        return np.dot(X, self.weights_[1:]) + self.weights_[0]
