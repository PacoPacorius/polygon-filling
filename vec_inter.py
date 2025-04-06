import numpy as np

def vector_inter(p1, p2, V1, V2, coord, dim):
    p = np.zeros(2)

    if dim == 1:
        lamda = - (p2[0] - coord) / (p1[0] - p2[0])
        p[1] = lamda * p1[1] + (1-lamda) * p2[1]
        p[0] = coord
    elif dim == 2:
        lamda = - (p2[1] - coord) / (p1[1] - p2[1])
        p[0] = lamda * p1[0] + (1-lamda) * p2[0]
        p[1] = coord

    V = lamda * V1 + (1-lamda) * V2
    return V



"""
demo
"""

p1 = np.array([5, 8])
p2 = np.array([7, 4])
V1 = np.array([1, -2])
V2 = np.array([-1, 3])
coord = 4
dim = 1
p = vector_inter(p1, p2, V1, V2, coord, dim)

print(p)
