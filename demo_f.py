import polygon_fill as pf
import numpy as np
import cv2 as cv

vertices = [[1, 4], [6, 2], [6, 11]]
img = cv.imread('fresque-saint-georges-2452226686.jpg')
print(img.shape)
print(img.dtype)

print('vertices = ', vertices)
pf.fill_polygon(img, vertices)

