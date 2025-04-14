import vec_inter
import numpy as np
import cv2 as cv
import math

def render_img(faces, vertices, vcolors, depth, shading):
    # initialize blank image
    img = np.zeros((512, 512, 3))
    img = np.astype(img, 'uint8')
    img.fill(255)

    L = vertices.shape[0]
    K = faces.shape[0]
    #polygon_fill(img, np.array([vertices[0:L:3], vertices[1:L:3], vertices[2:L:3]]), 
                 #np.array([vcolors[0:L:3], vcolors[1:L:3], vcolors[2:L:3]]), shading[0:K])

    # get sorted indices of the depth array, flip them to descending order. 
    # this paints triangles in the correct depth order
    for l in np.flip(depth.argsort()):
        img = polygon_fill(img, np.array([vertices[3*l], vertices[3*l+1], vertices[3*l+2]]),
                                np.array([vcolors[3*l], vcolors[3*l+1], vcolors[3*l+2]]), shading[l])
    return img

def find_active_edges(active_edges, vertices, K, xmin, xmax, ymin, ymax, y):
    for k in range(0, K):
        if ymin[k] == y + 1 and ymax[k] != ymin[k]:
            active_edges[k][0][1] = ymax[k]
            active_edges[k][1][1] = ymin[k]
            # pws kseroume poia tetagmenh antistoixei se poia tetmhmenh; grammikh paremvolh
            # "metaksy" dyo shmeiwn, alla h paremvolh ginetai panw se ena apo ta dyo shmeia,
            # etsi vriskoume poies htan oi arxikes tetmhmenes
            V, p = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymax[k], 2)
            active_edges[k][0][0] = p[0] 
            V, p = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymin[k], 2)
            active_edges[k][1][0] = p[0] 
        if ymax[k] == y + 1:
            active_edges[k][0][1] = -1
            active_edges[k][1][1] = -1
            active_edges[k][0][0] = -1 
            active_edges[k][1][0] = -1 

def find_active_points(active_points, active_edges, vertices, m, K, xmin, xmax, ymin, ymax, y):
    y_total_min = np.astype(np.min(ymin), int)
    x_total_min = np.astype(np.min(xmin), int)
    for k in range(0, K):
        # protash 1
        if ymin[k] == y + 1:
            active_points[k][1] = ymin[k]
            V, p = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymin[k], 2)
            active_points[k][0] = p[0]
        # protash 3
        elif active_points[k][0] != -1 and m[k] != math.inf:
            active_points[k][0] = active_points[k][0] + 1/m[k]
    # exclude horizontal lines, protash 2
        if ymin[k] == ymax[k] or ymax[k] == y + 1:
            active_points[k][1] = -1
            active_points[k][0] = -1

def polygon_fill(img, vertices, vcolors, shading):
    M = img.shape[0]
    N = img.shape[1]
    K = 3 # a triangle has 3 vertices
    #norm_vertices = np.zeros((3, 2))
    uv = np.zeros((3, 2))
    ymax = np.zeros(3)
    ymin = np.zeros(3)
    xmax = np.zeros(3)
    xmin = np.zeros(3)
    # prwth diastash gia plevres. max 2 plevres mporoun na einai energes th fora logw
    # kyrtothtas. defterh diastash einai ta shmeia pou orizoun tis plevres. 
    # trith diastash einai oi syntagmenes tou kathe shmeiou. 0 gia tetmhmenh, 1 gia 
    # tetagmenh.
    active_edges = np.zeros((3,2,2))
    active_edges.fill(-1)
    active_points = np.zeros((3,2))
    active_points.fill(-1)
    m = np.zeros(3)

    for k in range(0, K): 
        #print(k)
        ymax[k] = max(vertices[k-1][1], vertices[k][1])
        ymin[k] = min(vertices[k-1][1], vertices[k][1])
        xmax[k] = max(vertices[k-1][0], vertices[k][0])
        xmin[k] = min(vertices[k-1][0], vertices[k][0])
        if vertices[k][0] != vertices[k-1][0]:
            m[k] = (vertices[k][1] - vertices[k-1][1]) / (vertices[k][0] - vertices[k-1][0])
        else:
            m[k] = math.inf

    
    y_total_min = np.astype(np.min(ymin), int)
    y_total_max = np.astype(np.max(ymax), int)
    y = y_total_min - 1
    x_total_min = np.astype(np.min(xmin), int)
    x_total_max = np.astype(np.max(xmax), int)

    # vriskoume lista energwn akmwn
    find_active_edges(active_edges, vertices, K, xmin, xmax, ymin, ymax, y)

    # vriskoume lista energwn oriakwn shmeiwn
    find_active_points(active_points, active_edges, vertices, m, K, xmin, xmax, ymin, ymax, y)

    print('mk = ', m)
    print('active edges = ', active_edges)
    print('active points = ', active_points)

    ### polygon fill loop ###
    for y in range(y_total_min, y_total_max):
        # sort lista oriakwn shmeiwn kata x  ;;__;;
        sorted_active_points = np.sort(active_points, 0)
        sorted_active_points = np.astype(sorted_active_points, 'int')
        print('sorted active points = ', sorted_active_points)
        
        # fast pixel drawing
        if shading == "t":
            # only two points will be active at a time, the inactive point 
            # with negative values will be at the beginning, so we only need
            # the range of the other two points' x coordinate
            img = t_shading(img, vertices, uv, y, range(sorted_active_points[1][0], sorted_active_points[2][0]+1), cv.imread('fresque-saint-georges-2452226686.jpg'))
        elif shading == "f":
            img = f_shading(img, vertices, vcolors, y, range(sorted_active_points[1][0], sorted_active_points[2][0]+1))

        # enhmerwnoume lista energwn akmwn
        find_active_edges(active_edges, vertices, K, xmin, xmax, ymin, ymax, y)
        
        # enhmerwnoume lista energwn oriakwn shmeiwn
        find_active_points(active_points, active_edges, vertices, m, K, xmin, xmax, ymin, ymax, y)
                
        #cv.imshow('win', img)
        #cv.moveWindow('window', 0, 0)
        #cv.waitKey(0)
        print('y = ', y+1)
        print('active edges = ', active_edges)
        print('active points = ', active_points)
    return img


def f_shading(img, vertices, vcolors, rows, cols):
    K = vertices.shape[0]
    mean_color = np.zeros(3)
    for i in range(0, 3):
        for k in range(0, K):
            mean_color[i] = mean_color[i] + vcolors[k][i]
    mean_color = np.multiply(mean_color, [1/K])
    mean_color = np.rint(mean_color)
    mean_color = np.astype(mean_color, 'uint8')
    cols = np.array(cols)
    #print('mean color = ', mean_color)
    #print('cols = ', cols)
    #print('cols type = ', cols.dtype)
    #print('cols size = ', cols.size)
    if cols.size > 1:
        img[img.shape[0] - rows][cols[0]:cols[-1]] = mean_color
    elif cols.size == 1:
        img[img.shape[0] - rows][cols] = mean_color
    #print('img[', rows, '][', cols,'] = ', img[img.shape[0] - rows][cols])
    return img

def t_shading(img, vertices, uv, rows, cols, textImg):
    K = textImg.shape[0]
    L = textImg.shape[1]
    M = img.shape[0]
    N = img.shape[1]
    # normalize trangle points to texture image coordinates
    text_cols = np.multiply(cols, [K/M])
    text_rows = L/N * rows
    text_cols = np.astype(np.rint(cols), 'int')
    text_rows = math.ceil(text_rows - 0.5)

    print('text_cols = ', text_cols)
    print('text_cols size = ', text_cols.size)
    print('text_cols type = ', text_cols.dtype)
    if text_cols.size > 1:
        img[img.shape[0] - rows][cols[0]:cols[-1]] = textImg[textImg.shape[0] - text_rows][text_cols[0]:text_cols[-1]]
    elif text_cols.size == 1:
        img[img.shape[0] - rows][cols] = textImg[textImg.shape[0] - text_rows][text_cols]
    return img


