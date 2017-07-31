"""
	Code from the video Neural Networks
	Second type of NN: feedforward w/ hidden layer
	Can represent nonlinearities
	Propagate data forward, propagate error backward

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

# weights matrices, 2 values. 3*4 the first, 4*1 the second
synaptic_weight_0 = 2 * np.random.random((3,4)) - 1
synaptic_weight_1 = 2 * np.random.random((4,1)) - 1

print(synaptic_weight_0)
print(synaptic_weight_1)

# Training
for i in range(60000):
	# 2 layers
	hidden_layer = activate(np.dot(input_data, synaptic_weight_0))
	output_layer = activate(np.dot(hidden_layer, synaptic_weight_1))

	# Calculate final error (output layer)
	output_error = output_labels - output_layer

	# Use error to compute the gradient
	output_layer_gradient = output_error *activate(output_layer,True)

	# Calculate error for hidden layer
	hidden_error = output_layer_gradient.dot(synaptic_weight_1.T)

	# Use it to compute the gradient
	hidden_layer_gradient = hidden_error * activate(hidden_layer,True)

	# Update the weights using the gradients
	synaptic_weight_0 += input_data.T.dot(hidden_layer_gradient)
	synaptic_weight_1 += hidden_layer.T.dot(output_layer_gradient)

	if i % 10000 == 0:
		print("Error:" + str(np.mean(np.abs(output_error))))

# Testing
print(activate(np.dot(np.dot(np.array([1,0,0]), synaptic_weight_0), synaptic_weight_1)))