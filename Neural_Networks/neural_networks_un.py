import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.misc import toimage

class SOM:
    def __init__(self, x_size, y_size, trait_num, t_iter, t_step):
        self.weights = np.random.randint(256, size=(x_size, y_size, trait_num)).astype('float64')
        self.t_iter = t_iter
        self.map_radius = max(self.weights.shape)/2
        self.t_const = self.t_iter/math.log(self.map_radius)
        self.t_step = t_step

    def show(self):
        plt.imshow(toimage(self.weights.astype('uint8'), mode='RGB'))
        plt.show()

    def distance_matrix(self, vector):
        return np.sum((self.weights - vector) ** 2, 2)

    def bmu(self, vector):
        distance = self.distance_matrix(vector)
        return np.unravel_index(distance.argmin(), distance.shape)

    def bmu_distance(self, vector):
        x, y, rgb = self.weights.shape
        xi = np.arange(x).reshape(x, 1).repeat(y, 1)
        yi = np.arange(y).reshape(1, y).repeat(x, 0)
        return np.sum((np.dstack((xi, yi)) - np.array(self.bmu(vector))) ** 2, 2)

    def hood_radius(self, iteration):
        return self.map_radius * math.exp(-iteration/self.t_const)

    def teach_row(self, vector, i, dis_cut, dist):
        hood_radius_2 = self.hood_radius(i) ** 2
        bmu_distance = self.bmu_distance(vector).astype('float64')
        if dist is None:
            temp = hood_radius_2 - bmu_distance
        else:
            temp = dist ** 2 - bmu_distance
        influence = np.exp(-bmu_distance / (2 * hood_radius_2))
        if dis_cut:
            influence *= ((np.sign(temp) + 1) / 2)
        return np.expand_dims(influence, 2) * (vector - self.weights)

    def teach(self, t_set, distance_cutoff=False, distance=None):
        for i in range(self.t_iter):
            for x in t_set:
                self.weights += self.teach_row(x, i, distance_cutoff, distance)
        self.show()

s = SOM(200, 200, 3, 100, 0.1)
# t_set = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200], [120, 0, 100]])
t_set = np.random.randint(256, size=(15, 3))
s.teach(t_set)