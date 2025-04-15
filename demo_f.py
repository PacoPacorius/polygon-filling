import polygon_fill as pf
import numpy as np
import cv2 as cv

data = np.load('hw1.npy', None, True).item()
faces = data['t_pos_idx']
vertices = data['v_pos2d']
uvs = data['v_uvs']
vcolors = data['v_clr']
vcolors = np.multiply(vcolors, [255])
depth = data['depth']
shading = "f"
print('data = ', data)
img = pf.render_img(faces, vertices, vcolors, uvs, depth, shading)

cv.imshow('depth demo', img)
cv.moveWindow('depth demo', 0, 0)
cv.waitKey(0)

