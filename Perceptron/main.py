import numpy as np

from perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels = []
labels.append(np.array([1, 0]))
labels.append(np.array([0, 0]))
labels.append(np.array([0, 0]))
labels.append(np.array([0, 1]))

perceptron = Perceptron(no_of_inputs=2, no_of_electrons=2)
perceptron.train(training_inputs, labels)

inputs = np.array([1, 1])

print(perceptron.predict(inputs))


