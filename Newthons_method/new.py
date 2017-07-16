import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sympy import *

# Get a quadratic approximation by using 2 terms of Taylor series
def taylor(function,x, x0,n):
    i = 0
    q = 0
    while i <= n:
        q = q + ((function.diff(x,i).subs(x,x0))*(x-x0)**i) / (np.math.factorial(i))
        i += 1
    return q

def slope(x1, x2, z1, z2):
	m = (x2-z2) / (x1-z1)
	return m

def bias(x, y, m):
	b = y - (x*m)
	return b

x = Symbol('x')
y = x**2+x-10
dx = y.diff()
print(dx)
# Get X values, Y values and dx values
lam_y = lambdify(x, y, modules=['numpy'])
lam_dx = lambdify(x, dx, modules=['numpy'])
x_vals = np.linspace(-100, 100, 1000)
y_vals = lam_y(x_vals)
dx_vals = lam_dx(x_vals)

plt.plot(x_vals, y_vals)
plt.title("Visualizing Function")
plt.grid()		
plt.show()

ini = 0
while abs(ini) < 80:
	ini = randint(-100, 100)
tol = 0.25
error = 100
steps = 0

while error > tol:
	# Find the linear equation y = mx + b
	m = lam_dx(ini)
	b = bias(ini, lam_y(ini), m)
	print("Ini:", ini, "Value at point:", lam_y(ini), "slope:", m, "bias:", b)
	plt.plot(ini, lam_y(ini), 'ro')
	# Find x-intercept
	ini = (-b)/m
	error = lam_y(ini)
	print("Error at step", steps, ":", error)
	steps +=1
	# Plot x-intercept
	plt.plot(ini, 0, 'ro')
	# Plot the Polynomial and Tangent Line
	plt.plot(x_vals, y_vals)
	plt.plot(x_vals, [m*x+b for x in x_vals], 'r')
	plt.xlim(-100+5*steps, 100-5*steps)
	plt.ylim(-500, 10000-750*steps)
	plt.grid()
	plt.show()
	
print("Stopping at: ", ini, error)

# quad = simplify(taylor(y, 7, 2))
# print(quad)
# print(quad.coeff(x))
# print(quad.coeff(x**2))
# print('vertex: ', float())

