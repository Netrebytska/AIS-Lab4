import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

data = np.random.rand(100, 2)
test_point = np.array([[0.5, 0.5]])
n_neighbors = 5

nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
distances, indices = nbrs.kneighbors(test_point)

plt.scatter(data[:, 0], data[:, 1], c='black', marker='o', label='Data points')
plt.scatter(test_point[:, 0], test_point[:, 1], c='red', marker='x', label='Test point')
plt.scatter(data[indices][0][:, 0], data[indices][0][:, 1], c='blue', marker='o', label='Nearest neighbors')
plt.legend()
plt.show()
