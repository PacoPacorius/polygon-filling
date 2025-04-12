import vec_inter
import numpy as np
import cv2 as cv
import math



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
        if ymax[k] == y+1:
            active_edges[k][0][1] = -1
            active_edges[k][1][1] = -1
            active_edges[k][0][0] = -1 
            active_edges[k][1][0] = -1 

def find_active_points(active_points, active_edges, vertices, m, K, xmin, xmax, ymin, ymax, y):
    y_total_min = np.astype(np.min(ymin), int)
    x_total_min = np.astype(np.min(xmin), int)
    for k in range(0, K):
        #if y_total_min - 1 == y:
            #if ymin[k] == y + 1:
                #active_points[k][1] = ymin[k]
                #V, p = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymin[k], 2)
                #active_points[k][0] = p[0]
            ## protash 3
            #elif active_points[k][0] != -1 and m[k] != math.inf:
                #active_points[k][0] = active_points[k][0] + 1/m[k]
        #else:
            #if ymin[k] == y + 1:
                #active_points[k][1] = ymin[k]
                #V, p = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymin[k], 2)
                #active_points[k][0] = p[0]
            ## protash 3
            #elif active_points[k][0] != -1 and m[k] != math.inf:
                #active_points[k][0] = active_points[k][0] + 1/m[k]

        # protash 1
        if ymin[k] == y + 1:
            active_points[k][1] = ymin[k]
            V, p = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymin[k], 2)
            active_points[k][0] = p[0]
        # protash 3
        elif active_points[k][0] != -1 and m[k] != math.inf:
            active_points[k][0] = active_points[k][0] + 1/m[k]
    # exclude horizontal lines, protash 2
        if ymin[k] == ymax[k] or ymax[k] == y+1:
            active_points[k][1] = -1
            active_points[k][0] = -1

def render_img(vertices, vcolors, depth, shading):
    img = np.zeros((512, 512, 3))
    img = np.astype(img, 'uint8')
    img.fill(255)
    M = img.shape[0]
    N = img.shape[1]
    K = 3 # a triangle has 3 vertices
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
        # sort lista oriakwn shmeiwn kata x (;;;) ;;__;;
        sorted_active_points = np.sort(active_points, 0)
        print('sorted active points = ', sorted_active_points)
        cross_count = 0
        for x in range(x_total_min, x_total_max):
            for k in range(0, K):
                if x == math.ceil(sorted_active_points[k][0]):
                    cross_count = cross_count + 1
            #print('x = ', x, 'cross count = ', cross_count)
            if cross_count % 2 == 1:
                if shading == "f":
                    img = f_shading(img, vertices, vcolors, y, x)
                elif shading == "t":
                    t_shading(img, vertices, uv, textImg)
                elif shading == "d":
                    t_shading()
                    #drawpixel(img, y, x)
        

        # enhmerwnoume lista energwn akmwn
        find_active_edges(active_edges, vertices, K, xmin, xmax, ymin, ymax, y)
        
        # enhmerwnoume lista energwn oriakwn shmeiwn
        find_active_points(active_points, active_edges, vertices, m, K, xmin, xmax, ymin, ymax, y)
                
        cv.imshow('win', img)
        cv.moveWindow('window', 0, 0)
        cv.waitKey(0)
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
    for i in range(0, 3):
        mean_color[i] = math.ceil(mean_color[i])
    mean_color = np.astype(mean_color, 'uint8')
    #print('mean color = ', mean_color)
    img[img.shape[0] - rows][cols] = mean_color
    #print('img[', rows, '][', cols,'] = ', img[img.shape[0] - rows][cols])
    return img

def t_shading(img, vertices, uv, textImg):
    return img

def drawpixel(img, rows, cols):
    print('debug')
    img[img.shape[0] - rows][cols].fill(0)
    return 0

