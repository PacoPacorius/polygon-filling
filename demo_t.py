import polygon_fill as pf
import numpy as np
import cv2 as cv

vertices = np.array([[1, 4], [6, 2], [8, 11],
                     [8, 8], [10, 10], [9, 10],
                     [4, 6], [2/40, 1/40], [8, 2],
                     [12, 12], [9, 7], [9, 11],
                     [10, 10], [9, 5], [8, 11]])
vertices = np.multiply(vertices, [40])
L = vertices.shape[0]
faces = np.array([vertices[0:L:3], vertices[1:L:3], vertices[2:L:3]])
# colors in opencv's silly BGR format
vcolors = np.array([[180, 0, 200], [220, 0, 0], [0, 0, 170],
                    [255, 255, 255], [255, 255, 0,], [255, 255, 255],
                    [0, 255, 255], [255, 255, 255], [255, 255, 255],
                    [120, 120, 120], [255, 100, 0], [255, 120, 120],
                    [0, 120, 255], [0, 120, 255], [120, 120, 120]])
depth = np.array([1, 2, 7, 4, 8])
shading = np.array(["t", "t", "f", "f", "t"])   # "f"lat, "t"exture 
img = pf.render_img(faces, vertices, vcolors, depth, shading)
cv.imshow('with vectorization', img)
cv.moveWindow('window', 0, 0)
cv.waitKey(0)
