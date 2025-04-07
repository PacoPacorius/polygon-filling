import polygon_fill as pf
import numpy as np

vertices = [[2, 3], [7, 1], [3, 11]]
img = np.zeros(1)

pf.fill_polygon(img, vertices)

print('vertices = ', vertices)
