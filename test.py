import numpy as np
from scipy.spatial import distance as dist
ce = np.array([[2,5],[8,50],[9,6]])
c = [ce]
d = dist.cdist(c[-1], c[-1])
print (d)