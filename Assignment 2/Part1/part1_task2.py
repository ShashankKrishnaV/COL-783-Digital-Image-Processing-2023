import os
import cv2
import json
import math
import numpy as np

path = input('Enter the path for images (I1, I2, I3): ')
path = path + '/'

I1 = cv2.imread(path + 'I1.jpg')
I2 = cv2.imread(path + 'I2.jpg')
I3 = cv2.imread(path + 'I3.jpg')

# Resize if shapes are not same
h2, w2, _ = I2.shape
I1 = cv2.resize(I1.copy(), (w2, h2))
I3 = cv2.resize(I3.copy(), (w2, h2))


# Task 2


anchor_points = None

with open(path + 'anchor_points_task1.txt', 'r') as f:
	line = ((f.readlines())[0])
	line = line.replace("(", "[").replace(")", "]").replace("'", "\"")
	anchor_points = json.loads(str(line))


#For image along anchor points

image_shape = I1.shape

I1_anchor = anchor_points['I1']
I2_anchor = anchor_points['I2']
I3_anchor = anchor_points['I3']


def compute_barycentric_coordinates(P, A, B, C):
    # Create the matrix [A, B, C]
    triangle_matrix = np.column_stack((A, B, C))

    barycentric_coords = np.linalg.solve(triangle_matrix, P)
    
    return barycentric_coords

def area(x1, y1, x2, y2, x3, y3):
 
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)
 
def isInside(x1, y1, x2, y2, x3, y3, x, y):
    A = area (x1, y1, x2, y2, x3, y3)

    A1 = area (x, y, x2, y2, x3, y3)
    A2 = area (x1, y1, x, y, x3, y3)
    A3 = area (x1, y1, x2, y2, x, y)

    if(A == A1 + A2 + A3):
        return True, A1, A2, A3
    else:
        return False, A1, A2, A3

def check_if_inside(points, triangle_points):
    indices = []

    for i in range(len(points)):
        if i>=4:
            x, y = points[i]
            x1, y1 = triangle_points[0]
            x2, y2 = triangle_points[1]
            x3, y3 = triangle_points[2]

            if isInside(x1, y1, x2, y2, x3, y3, x, y) == True:
                indices.append(i)
    return indices


def triangle_map_point(pt, anchor_image1, anchor_image2):
    # anchor_image1 : target_image
    # anchor_image2 : source_image
    # pt : point on target_image

    A, B, C = anchor_image1[0], anchor_image1[1], anchor_image1[2]
    A_dash, B_dash, C_dash = anchor_image2[0], anchor_image2[1], anchor_image2[2]

    Ax, Ay = A_dash[0], A_dash[1]
    Bx, By = B_dash[0], B_dash[1]
    Cx, Cy = C_dash[0], C_dash[1]

    x, y = pt

    u, v, w = compute_barycentric_coordinates(np.array([x, y, 1]), np.array([Ax, Ay, 1]), np.array([Bx, By, 1]), np.array([Cx, Cy, 1]))
    
    # Calculate corresponding point in the source image
    Px = u * A[0] + v * B[0] + w * C[0]
    Py = u * A[1] + v * B[1] + w * C[1]

    return (int(Px), int(Py))


def triangle_map(anchor_image1, anchor_image2):
    #u, v, w = barycentric_coords(anchor_image2)
    A, B, C = anchor_image1[0], anchor_image1[1], anchor_image1[2]
    A_dash, B_dash, C_dash = anchor_image2[0], anchor_image2[1], anchor_image2[2]

    pixel_map = {}
    pixel_map[(A_dash[1], A_dash[0])] = (A[1], A[0])
    pixel_map[(B_dash[1], B_dash[0])] = (B[1], B[0])
    pixel_map[(C_dash[1], C_dash[0])] = (C[1], C[0])

    Ax, Ay = A_dash[0], A_dash[1]
    Bx, By = B_dash[0], B_dash[1]
    Cx, Cy = C_dash[0], C_dash[1]

    minX = min(A_dash[0], B_dash[0], C_dash[0])
    maxX = max(A_dash[0], B_dash[0], C_dash[0])

    minY = min(A_dash[1], B_dash[1], C_dash[1])
    maxY = max(A_dash[1], B_dash[1], C_dash[1])

    for x in range(minX, maxX + 1):
        for y in range(minY, maxY + 1):
            check, A1, A2, A3 = isInside(Ax, Ay, Bx, By, Cx, Cy, x, y)
            if check and A1+A2+A3!=0:

                u, v, w = compute_barycentric_coordinates(np.array([x, y, 1]), np.array([Ax, Ay, 1]), np.array([Bx, By, 1]), np.array([Cx, Cy, 1]))
                # print('u, v, w: ', u, v, w)
                
                # Calculate corresponding point in the source image
                Px = u * A[0] + v * B[0] + w * C[0]
                Py = u * A[1] + v * B[1] + w * C[1]

                pixel_map[(y, x)] = (int(Py), int(Px))

    # return image1_mask
    return pixel_map

def transfer_color(source_image, pixel_map, target_image):
    # print(pixel_map)
    target_pix = pixel_map.keys()
    # masked_image = np.zeros(source_image.shape, dtype=np.uint8)

    for pix in target_pix:
        # print(pixel_map[pix])
        target_image[int(pix[0]), int(pix[1])] = source_image[int(pixel_map[pix][0]), int(pixel_map[pix][1])]

    return target_image

I1_prime = np.zeros_like(I1, dtype=np.uint8)
I3_prime = np.zeros_like(I3, dtype=np.uint8)

# I1 prime
map_pts = triangle_map(np.array(I1_anchor), np.array(I2_anchor))
I1_prime = transfer_color(I1.copy(), map_pts, I1_prime)

cv2.imwrite(path + 'I1_prime_tri_task2.jpg', I1_prime)


# I3 prime
map_pts = triangle_map(np.array(I3_anchor), np.array(I2_anchor))
I3_prime = transfer_color(I3.copy(), map_pts, I3_prime)

cv2.imwrite(path + 'I3_prime_tri_task2.jpg', I3_prime)


# I4
I1_prime_anchor = I2_anchor.copy()
I3_prime_anchor = I2_anchor.copy()

I4_tri = np.zeros_like(I2, dtype=np.uint8)

map_pts = triangle_map(np.array(I1_prime_anchor), np.array(I3_prime_anchor))
I4_tri = transfer_color(I2.copy(), map_pts, I4_tri)

cv2.imwrite(path + 'I4_tri_task2.jpg', I4_tri)

