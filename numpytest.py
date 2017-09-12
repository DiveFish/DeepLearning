import numpy as np

labels = np.zeros(
    shape=(512, 100, 100))

x2 = np.zeros(shape=(100, 50))
res = np.matmul(labels, x2)
print(res.shape)