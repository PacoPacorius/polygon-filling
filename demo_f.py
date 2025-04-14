import polygon_fill as pf
import numpy as np
import cv2 as cv

vertices = np.array([[1, 4], [6, 2], [8, 11],
                     [12, 11], [5, 7], [1, 12],
                     [5, 5], [5, 10], [10, 10]])

vertices = np.multiply(vertices, [40])
L = vertices.shape[0]
faces = np.array([vertices[0:L:3], vertices[1:L:3], vertices[2:L:3]])
# colors in opencv's silly BGR format
vcolors = np.array([[180, 0, 200], [220, 0, 0], [0, 0, 170],
                    [255, 255, 255], [255, 255, 0,], [255, 255, 255],
                    [0, 255, 255], [255, 255, 255], [255, 255, 255]])
depth = np.array([1, 10, 5])
shading = np.array(["f", "f", "f"])   # "f"lat, "t"exture 
img = pf.render_img(faces, vertices, vcolors, depth, shading)

print('vertices = ', vertices)
cv.imshow('depth demo', img)
cv.moveWindow('depth demo', 0, 0)
cv.waitKey(0)





vertices = np.array([[3, 4], [6, 4], [6, 11],
                     [3, 4], [6, 4], [8, 2],
                     [6, 4], [6, 11], [8, 6]])
vertices = np.multiply(vertices, [40])
L = vertices.shape[0]
faces = np.array([vertices[0:L:3], vertices[1:L:3], vertices[2:L:3]])
# colors in opencv's silly BGR format
vcolors = np.array([[180, 0, 200], [220, 0, 0], [0, 0, 170],
                    [255, 255, 255], [255, 255, 0,], [255, 255, 255],
                    [0, 255, 255], [255, 255, 255], [255, 255, 255]])
depth = np.array([1, 10, 5])
shading = np.array(["f", "f", "f"])   # "f"lat, "t"exture 
img = pf.render_img(faces, vertices, vcolors, depth, shading)

print('vertices = ', vertices)
cv.imshow('conventions demo', img)
cv.moveWindow('conventions demo', 0, 0)
cv.waitKey(0)
