import polygon_fill as pf
import numpy as np
import cv2 as cv

vertices = [[1, 4], [6, 2], [8, 11]]
vcolors = [[0, 0, 0], [120, 0, 100], [0, 80, 77]]
depth = np.zeros(20)
shading = "f"   # "f"lat, "t"exture, "d"ebug (monochrome)
#vertices = np.multiply(vertices, [20])
img = cv.imread('fresque-saint-georges-2452226686.jpg')
img2 = np.zeros((20, 20))
img2 = np.astype(img2, 'uint8')
img2.fill(255)

print('vertices = ', vertices)
pf.render_img(img2, vertices, vcolors, depth, shading)

cv.imshow('window', img2)
cv.moveWindow('window', 0, 0)
cv.waitKey(0)
print(img2.shape[0])
