import polygon_fill as pf
import numpy as np
import cv2 as cv

vertices = np.array([[1, 4], [6, 2], [8, 11]])
# colors in opencv's silly BGR format
vcolors = [[180, 0, 200], [220, 0, 0], [0, 0, 170]]
depth = np.zeros(20)
shading = "t"   # "f"lat, "t"exture, "d"ebug (monochrome)
vertices = np.multiply(vertices, [40])

print('vertices = ', vertices)
img2 = pf.render_img(vertices, vcolors, depth, shading)

cv.imshow('with vectorization', img2)
cv.moveWindow('window', 0, 0)
cv.waitKey(0)
