""" 
	Code from the video Neural Networks
	First type of NN: simple af Networks
	Can't represent nonlinear functions

"""
# import dependencies
import numpy as np

# For reproducibility
np.random.seed(0)

# input data - Linear correlation
input_data = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
output_labels = np.array([[0,1,1,0]]).T

print(input_data)
print(output_labels)

# Sigmoid function - converts to values between 0 and 1
def activate(x, deriv = False):
	if deriv == True:
		return x*(1-x)
	return 1/(1+np.exp(-x))

# weight matrix, 3*1 dimensions
synaptic_weights = 2 * np.random.random((3,1)) - 1

print(synaptic_weights)

# Dot product (matrix multiplication) of the two matrices
print(activate(np.dot(input_data, synaptic_weights)))

# Training
for i in range(10000):
	output = activate(np.dot(input_data, synaptic_weights))
	error = output_labels - output

	# print(error)
	synaptic_weights += np.dot(input_data.T, error * activate(output, True))

print(synaptic_weights)

# Testing
print(activate(np.dot(np.array([1,0,0]), synaptic_weights)))