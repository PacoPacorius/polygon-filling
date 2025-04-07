import vec_inter
import numpy as np

def fill_polygon(img, vertices):
    M = img # rows
    N = img # columns
    K = 3 # a triangle has 3 vertices
    ymax = np.zeros(3)
    ymin = np.zeros(3)
    xmax = np.zeros(3)
    xmin = np.zeros(3)

    for k in range(0, K): 
        print(k)
        ymax[k] = max(vertices[k-1][1], vertices[k][1])
        ymin[k] = min(vertices[k-1][1], vertices[k][1])
        xmax[k] = max(vertices[k-1][0], vertices[k][0])
        xmin[k] = min(vertices[k-1][0], vertices[k][0])
    print('ymax = ', ymax)
    print('ymin = ', ymin)
    print('xmax = ', xmax)
    print('xmin = ', xmin)


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
