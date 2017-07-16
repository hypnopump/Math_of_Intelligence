import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sympy import *

# Get an approximation by using n terms of Taylor series
def taylor(function,x0,n):
    q = 0
    for i in range(1, n+1):
        q = q + ((function.diff(x,i).subs(x,x0))*(x-x0)**i) / (np.math.factorial(i))
    return q
# Compute the avg y-variaton between 2 points
def slope(x1, x2, z1, z2):
	return (x2-z2) / (x1-z1)
# Return the bias of a line
def bias(x, y, m):
	b = y - (x*m)
	return b

x = Symbol('x')
y = x**4
# Get X values, Y values and dx values
lam_y = lambdify(x, y, modules=['numpy'])
x_vals = np.linspace(-50, 50, 1000)
y_vals = lam_y(x_vals)

ini = randint(-50, 50)
tol = 2
error = 100
steps = 0

plt.plot(x_vals, y_vals)
plt.plot([ini, ini],[0, lam_y(ini)], 'ro')
plt.title("Visualizing Function")
plt.grid()		
plt.show()

# while error > tol:
for i in range(3):
	# Find the quadratic approx and the vertex
	quad = simplify(taylor(y, ini, 2))
	print(quad)
	ini = -quad.coeff(x) / (2*quad.coeff(x**2))
	lam_quad = lambdify(x, quad, modules=['numpy'])
	quad_vals = lam_quad(x_vals)

	# Find Error and count steps
	error = round(float(lam_y(ini)), 2)
	print("Error at step", steps, ":", error)
	steps +=1
	# Plot
	plt.plot(x_vals, y_vals)
	plt.plot(x_vals, quad_vals)
	plt.plot(ini, lam_y(ini), 'o')
	plt.plot(ini, lam_quad(ini), 'o')
	plt.plot(ini, 0, 'o')
	
plt.grid()
plt.show()
	
print("Stopping at: ", ini, error)
print("After", steps, "steps")