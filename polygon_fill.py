import vec_inter
import numpy as np

def drawpixel(x, y):
    return 0

def fill_polygon(img, vertices):
    M = img # rows
    N = 100 # columns
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
    active_points = np.zeros((3,2))
    mk = np.zeros(3)

    for k in range(0, K): 
        print(k)
        ymax[k] = max(vertices[k-1][1], vertices[k][1])
        ymin[k] = min(vertices[k-1][1], vertices[k][1])
        xmax[k] = max(vertices[k-1][0], vertices[k][0])
        xmin[k] = min(vertices[k-1][0], vertices[k][0])
        m[k] = (vertices[k][1] - vertices[k-1][1]) / (vertices[k][0] - vertices[k-1][0]

    y_total_min = np.astype(min(ymin), int)
    y_total_max = np.astype(max(ymax), int)
    y = y_total_min - 1
    print('y total min = ', y_total_min)

    # vriskoume lista energwn akmwn
    for k in range(0, K):
        if ymin[k] == y + 1:
            active_edges[k][0][1] = ymax[k]
            active_edges[k][1][1] = ymin[k]
            # pws kseroume poia tetagmenh antistoixei se poia tetmhmenh; grammikh paremvolh
            # "metaksy" dyo shmeiwn, alla h paremvolh ginetai panw se ena apo ta dyo shmeia,
            # etsi vriskoume poies htan oi arxikes tetmhmenes
            V, p = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymax[k], 2)
            active_edges[k][0][0] = p[0] 
            V, p = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymin[k], 2)
            active_edges[k][1][0] = p[0] 
        if ymax[k] == y:
            active_edges[k][0][1] = -1
            active_edges[k][1][1] = -1
            active_edges[k][0][0] = -1 
            active_edges[k][1][0] = -1 

    print('active edges = ', active_edges)

    # vriskoume lista energwn oriakwn shmeiwn
    for k in range(0, K):
        # protash 1
        if ymin[k] == y + 1:
            active_points[k][1] = ymin[k]
            V, p = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymin[k], 2)
            active_points[k][0] = ymin[k]
        # protash 3
        elif active_points[k][0] <> -1:
            active_points[k][0] = active_points[k][0] + m[k]
        # exclude horizontal lines, protash 2
        if ymin[k] == ymax[k]:
            active_points[k][1] = -1
            active_points[k][0] = -1
            
    print('active points = ', active_points)

    for y in range(y_total_min, y_total_max):
        # sort lista oriakwn shmeiwn kata x (;;;) ;;__;;
        cross_count = 0
        for x in range(0, N):
            drawpixel(x,y)

        # enhmerwnoume lista energwn akmwn
        for k in range(0, K):
            if ymin[k] == y + 1:
                active_edges[k][0][1] = ymax[k]
                active_edges[k][1][1] = ymin[k]
                # pws kseroume poia tetagmenh antistoixei se poia tetmhmenh; grammikh paremvolh
                # "metaksy" dyo shmeiwn, alla h paremvolh ginetai panw se ena apo ta dyo shmeia,
                # etsi vriskoume poies htan oi arxikes tetmhmenes
                V, p = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymax[k], 2)
                active_edges[k][0][0] = p[0] 
                V, p = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymin[k], 2)
                active_edges[k][1][0] = p[0] 
            if ymax[k] == y:
                active_edges[k][0][1] = 0
                active_edges[k][1][1] = 0
                active_edges[k][0][0] = 0 
                active_edges[k][1][0] = 0 
        
        print('y = ', y+1)
        print('active edges = ', active_edges)


# algorithmos plhrwshs polygonwn
#
#for k=0:1:K-1
#Βρίσκουμε τα xkmin, xkmax, ykmin, ykmax της ακμής k
#end
#ymin = min_k {y0min, y1min, ...}
#ymax = max_k {y0max, y1max, ...}
#Βρίσκουμε τη <Λίστα Ενεργών Ακμών> για την Γραμμή Σάρωσης y == ymin;
#Βρίσκουμε τη <Λίστα Ενεργών Οριακών Σημείων> για την Γραμμή Σάρωσης y == ymin;
#for y=ymin:1:ymax
#Διατάσσουμε τη <Λίστα Ενεργών Οριακών Σημείων> ώς προς x
#cross_count=0;
#% Αρχικοποιούμε το πλήθος Ενεργών Οριακών Σημείων
#% που έχουν ήδη σαρωθεί
#% Αρχή σάρωσης γραμμής y
#for x=0:1:N
#if x == τετμημένη κάποιου Ενεργού Οριακού Σημείου
#cross_count = cross_count + 1;
#end
#if (cross_count είναι ΠΕΡΙΤΤΟ)
#drawpixel(x,y)
#end
#end
#% Τέλος σάρωσης γραμμής y
# Ενημερώνουμε αναδρομικά τη <Λίστα Ενεργών Ακμών>
# Ενημερώνουμε αναδρομικά τη <Λίστα Ενεργών Οριακών Σημείων>
