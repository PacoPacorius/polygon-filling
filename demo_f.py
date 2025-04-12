import polygon_fill as pf
import numpy as np
import cv2 as cv

vertices = np.array([[1, 4], [6, 2], [8, 11]])
vcolors = [[180, 0, 200], [220, 0, 0], [0, 0, 170]]
depth = np.zeros(20)
shading = "f"   # "f"lat, "t"exture, "d"ebug (monochrome)
#vertices = np.multiply(vertices, [20])
img = cv.imread('fresque-saint-georges-2452226686.jpg')

print('vertices = ', vertices)
img2 = pf.render_img(vertices, vcolors, depth, shading)

cv.imshow('window', img2)
cv.moveWindow('window', 0, 0)
cv.waitKey(0)
