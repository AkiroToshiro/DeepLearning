import numpy as np


class Perceptron(object):

    def __init__(self, no_of_inputs, no_of_electrons=1, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = []
        for _ in range(no_of_electrons):
            self.weights.append(np.zeros(no_of_inputs + 1))

    def predict(self, inputs):
        predict = []
        for w in self.weights:
            summation = np.dot(inputs, w[1:]) + w[0]
            if summation > 0:
                predict.append(1)
            else:
                predict.append(0)
        return predict

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                for w, l in zip(self.weights, label):
                    prediction = self.predict(inputs)
                    w[1:] += self.learning_rate * (l - prediction[0]) * inputs
                    w[0] += self.learning_rate * (l - prediction[0])
