""" 
	Code from the video Neural Networks
	Third type of NN: Recurrent Networks
	Can learn sequential mapping (time/order)

"""
import numpy as np
import copy

# Sigmoid function - converts to values between 0 and 1
def activate(x, deriv = False):
	if deriv == True:
		return x*(1-x)
	return 1/(1+np.exp(-x))

int_to_binary = {}
binary_dim = 8
max_val = (2**binary_dim) # 2^8 = 256
binary_val = np.unpackbits(np.array([range(max_val)], dtype = np.uint8).T, axis = 1) # Calc Binary
for i in range(max_val): # map Integer values to Binary Values 
	int_to_binary[i] = binary_val[i]

# Hyperparameters
inputLayerSize = 2
hiddenLayerSize = 16
outputLayerSize = 1

# 3 weight values
W1 = 2*np.random.random((inputLayerSize, hiddenLayerSize)) -1
W2 = 2*np.random.random((hiddenLayerSize, outputLayerSize)) -1
W_h = 2*np.random.random((hiddenLayerSize, hiddenLayerSize)) -1 # Current h to h in next time

# Initialize updated weight values
W1_update = np.zeros_like(W1)
W2_update = np.zeros_like(W2)
W_h_update = np.zeros_like(W_h)

# Compute the sum of two integers
for j in range(10000):

	# a+b = c (random values)
	a_int = np.random.randint(max_val/2)
	b_int = np.random.randint(max_val/2)
	c_int = a_int + b_int

	# Get binary values for a,b,c
	a = int_to_binary[a_int]
	b = int_to_binary[b_int]
	c = int_to_binary[b_int]

	# Save predicted binary outputs
	d = np.zeros_like(c)

	# Initialize Error
	overallError = 0

	# Store output gradients & hidden layer values
	output_layer_gradients = []
	hidden_layer_values = []
	hidden_layer_values.append(np.zeros(hiddenLayerSize)) # init as 0

	# Forward propagation to compute the sum of two 8 digit long binary integers
	for position in range(binary_dim):

		# input - binary values of a & b_int
		X = np.array([[a[binary_dim - position -1], b[binary_dim - position - 1]]])
		# output - the sum c
		y = np.array([[c[binary_dim - position -1]]]).T

		# Calculate the Error
		layer_1 = activate(np.dot(X, W1) + np.dot(hidden_layer_values[-1], W_h))
		layer_2 = activate(np.dot(layer_1, W2))
		output_error = y - layer_2

		# Save the error gradients at each step as it will be propagated back
		output_layer_gradients.append((output_error)*activate(layer_2, True))

		# Save the sum of the error at each binary position
		overallError += np.abs(output_error[0])

		# Round off the values to the nearest "0" or "1" and save it to a list
		d[binary_dim - position - 1] = np.round(layer_2[0][0])

		# Save the hidden layer to be used later
		hidden_layer_values.append(copy.deepcopy(layer_1))

	future_layer_1_gradient = np.zeros(hiddenLayerSize)

	# Backpropagate the error to previous timesteps
	for position in range(binary_dim):
		X = np.array([[a[position], b[position]]]) # a[0], b[0] => a[1], b[1]
		# Current layer
		layer_1 = hidden_layer_values [-position-1]
		# Layer before the current layer
		prev_hidden_layer = hidden_layer_values[-position-2]
		# Errors at output layer
		output_layer_gradient = output_layer_gradients[-1]
		layer_1_gradients = (future_layer_1_gradient.dot(W_h.T) + output_layer_gradient.dot(W2.T)) * activate(layer_1, )

		# Update all the weights and try again
		W2_update += np.atleast_2d(layer_1).T.dot(output_layer_gradient)
		W_h_update += np.atleast_2d(prev_hidden_layer).T.dot(layer_1_gradients)
		W1_update += X.T.dot(layer_1_gradients)

		future_layer_1_gradient = layer_1_gradients

	# Update the weights with the values
	W1  += W1_update
	W2  += W2_update
	W_h  += W_h_update

print(W1)
print(W2)
print(W_h)