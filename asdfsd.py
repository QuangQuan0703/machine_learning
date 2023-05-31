import math
from scipy.spatial import distance as dist
import numpy as np
a = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])
b = np.array([[ 1,  1,  2],[1,1,2]])
print(math.dist(a[0,:],b[0,:]))
a[0:] = (1,1,1)
print(a[0,:])