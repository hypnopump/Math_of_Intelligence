# Code written entireley by Eric Alcaide
# https://github.com/EricAlcaide

# Import modules
import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

PATH = "diamonds.csv"
# Data extraction
def get_data(PATH):
	data = pd.read_csv(PATH)
	useless_features = ['color', 'clarity', 'cut', 'table', 'depth', 'y', 'z']
	for f in useless_features:
	    del data[f]
	cleaned = [list(data.loc[random.randint(1,53000)]) for i in range(500)]
	print(cleaned)

	return [[round(float(c[1]), 2), c[3], c[2]] for c in cleaned]

# Plot a graph representation of the data in 3D
def plot_graph(points,plane):
    f1 = [p[0] for p in points] # Mass
    f2 = [p[1] for p in points] # Width
    f3 = [p[2] for p in points] # Price
    fig = plt.figure()
    fig.suptitle("Correlation between Width, Weight and Diamond's prices")
    ax = fig.gca(projection='3d')
    ax.axis([0, 5, 3,11])
    ax.set_xlabel('Mass (carats)')
    ax.set_ylabel('Width (mm)')
    ax.set_zlabel('Price ($)')
    ax.plot(f1, f2, f3, 'r.', label='Diamonds') # Plot points
    ax.legend()

    if plane != False:
        # create x,y points to be part of the hyperplane
        xx, yy = np.meshgrid(np.linspace(0, 5), np.linspace(3.5, 11))
        # calculate corresponding z for each x,y pair
        z = plane[0] + plane[1]*xx + plane[2]*yy
        # plot the surface
        surf = ax.plot_surface(xx, yy, z, cmap=mpl.cm.coolwarm)

    plt.show()

# Z = w0 + w1*X + w2*Y
# w0 is z-intercept, w1 is x-slope, w2 is y-slope.
def compute_error_for_line_given_points(w0, w1, w2, points):
    totalError = 0
    for p in points:
        x, y, z = p[0], p[1], p[2]
        totalError += (z - (w0 + w1*x + w2*y)) ** 2
    return totalError / float(len(points))

# Calculate the gradients and take a step in that direction.
def step_gradient(w0_current, w1_current, w2_current, points, learningRate):
    w0_gradient = w1_gradient = w2_gradient = 0
    N = float(len(points))
    for p in points:
        x, y, z = p[0], p[1], p[2]
        # Make the prediction w/ the regression function
        f_x = w0_current + w1_current*x + w2_current*y
        # Build the gradients for each feature/dimension
        w0_gradient += -(2/N) * (z - f_x)
        w1_gradient += -(2/N) * x * (z - f_x)
        w2_gradient += -(2/N) * y * (z - f_x)
    # Update the weights with gradients*learningRate    
    new_w0 = w0_current - (learningRate * w0_gradient)
    new_w1 = w1_current - (learningRate * w1_gradient)
    new_w2 = w2_current - (learningRate * w2_gradient)

    return [new_w0, new_w1, new_w2]

# Run the Gradient Descent optimizer method
def gradient_descent_runner(points, starting_w0, starting_w1, starting_w2, learning_rate, num_iterations):
    w0 = starting_w0
    w1 = starting_w1
    w2 = starting_w2
    for i in range(num_iterations):
        w0, w1, w2 = step_gradient(w0, w1, w2, np.array(points), learning_rate)
    return [w0, w1, w2]

# Just Do It
def run(points):
    learning_rate = 0.005
    initial_w0 = initial_w1 = initial_w2 = 0 # initial weights set to 0
    num_iterations = 2500
    
    print("Starting gradient descent at w0 = {0}, w1 = {1}, w2 = {2} error = {3}".format(initial_w0, initial_w1, initial_w2, compute_error_for_line_given_points(initial_w0, initial_w1, initial_w2, points)))
    print("Running...")
    [w0, w1, w2] = gradient_descent_runner(points, initial_w0, initial_w1, initial_w2, learning_rate, num_iterations)
    print("After {0} iterations w0 = {1}, w2 = {2}, w3 = {3} error = {4}".format(num_iterations, w0, w1, w2, compute_error_for_line_given_points(w0, w1, w2, points)))
    plot_graph(points, [w0, w1, w2])

if __name__ == '__main__':
    run(get_data(PATH))