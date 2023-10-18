import os
import cv2
import json
import math
import numpy as np
import dlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


path = input('Enter the path for images (I1, I2, I3): ')
path = path + '/'

I1 = cv2.imread(path + 'I1.jpg')
I2 = cv2.imread(path + 'I2.jpg')
I3 = cv2.imread(path + 'I3.jpg')

pts = int(input('Number of correspondences: '))

f_path = path + f'task3_1_technique1/{pts}'
os.makedirs(f_path, exist_ok=True)
f_path += '/'


h2, w2, _ = I2.shape
I1 = cv2.resize(I1.copy(), (w2, h2))
I3 = cv2.resize(I3.copy(), (w2, h2))


# Task 3 - 1

anchor_sets = {}

#select anchor points interactively using the mouse
def select_anchor_points(image, pts, image_name):
    anchor_points = []
    sample = image.copy()

    points = 0
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            anchor_points.append([x, y])
            # sample = image.copy()
            cv2.circle(sample, (x, y), 2, (0, 0, 255), -1)
            cv2.putText(sample, str(len(anchor_points)), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(f'Select {pts} Points', sample)
            if len(anchor_points) == pts:
                return


    cv2.imshow(f'Select {pts} Points', image)
    cv2.setMouseCallback(f'Select {pts} Points', on_mouse)

    while len(anchor_points) < pts:
        if cv2.waitKey(1) == 27 or len(anchor_points) == pts: 
        	cv2.imwrite(f_path + image_name + '_marked.jpg', sample)
        	cv2.destroyAllWindows()
        	break

    cv2.imwrite(f_path + image_name + '_marked.jpg', sample)
    cv2.destroyAllWindows()
    return anchor_points

def corners(image):
	l = []
	l.append([0, 0])
	l.append([0, image.shape[0]-1])
	l.append([image.shape[1]-1,image.shape[0]-1])
	l.append([image.shape[1]-1, 0])
	return l

#Ablation on points
# pts = [1, 5, 10, 20, 30, 35]

# Unpack for manual annotation
image = I1
anchor_points = select_anchor_points(image, pts, 'I1')
anchor_sets["I1"] = anchor_points

image = I3
anchor_points = select_anchor_points(image, pts, 'I3')
anchor_sets["I3"] = anchor_points

image = I2
anchor_points = select_anchor_points(image, pts, 'I2')
anchor_sets["I2"] = anchor_points

with open(f_path + f'markers_{pts}_points.txt', 'w+') as f:
    f.write(str(anchor_sets))


anchor_sets["I1"] = corners(I1) + anchor_sets["I1"]
anchor_sets["I2"] = corners(I2) + anchor_sets["I2"]
anchor_sets["I3"] = corners(I3) + anchor_sets["I3"]



# Delaunay
points_I1 = anchor_sets["I1"]
image = I1.copy()

points=[]
for i in points_I1:
    points.append(tuple(i))

points = np.array(points)

tri_I1 = Delaunay(points)

# Plot the Delaunay triangulation
plt.triplot(points[:, 0], points[:, 1], tri_I1.simplices.copy())
plt.plot(points[:, 0], points[:, 1], 'o')

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.savefig(f_path + "I1_delaunay.jpg", dpi=300, bbox_inches='tight')
plt.clf()


points_I2 = anchor_sets["I2"]
image = I2.copy()

points=[]
for i in points_I2:
    points.append(tuple(i))

points = np.array(points)

tri_I2 = Delaunay(points)

# Plot the Delaunay triangulation
plt.triplot(points[:, 0], points[:, 1], tri_I2.simplices.copy())
plt.plot(points[:, 0], points[:, 1], 'o')

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.savefig(f_path + "I2_delaunay.jpg", dpi=300, bbox_inches='tight')
plt.clf()


points_I3 = anchor_sets["I3"]
image = I3.copy()

points=[]
for i in points_I3:
    points.append(tuple(i))

points = np.array(points)

tri_I3 = Delaunay(points)

# Plot the Delaunay triangulation
plt.triplot(points[:, 0], points[:, 1], tri_I3.simplices.copy())
plt.plot(points[:, 0], points[:, 1], 'o') 

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.savefig(f_path + "I3_delaunay.jpg", dpi=300, bbox_inches='tight')


points_I1 = np.array(points_I1)
points_I2 = np.array(points_I2)


# Matching the vertices of triangles (I1 to match I2. Same triangles of I1 on I3 to match I2 And remap I3 points. Now compute traingle correspondence between I1prime and I3prime and use it to compute I4)
def warp_triangle(image, triangle, warp_matrix):
    h, w, _ = image.shape
    warped_image = np.zeros_like(image, dtype=np.uint8)

    #2x3 matrix to 3x3 matrix
    warp_matrix = np.vstack([warp_matrix, [0, 0, 1]])

    A, B, C = np.array(triangle, dtype=np.int32)
    A, B, C = list(A), list(B), list(C)

    min_x, max_x = min(A[0], B[0], C[0]), max(A[0], B[0], C[0])
    min_y, max_y = min(A[1], B[1], C[1]), max(A[1], B[1], C[1])

    for y in range(min_y, max_y+1):
        for x in range(min_x, max_x+1):

            if isInside(A[0], A[1], B[0], B[1], C[0], C[1], x, y):

                #apply the inverse transformation to find the corresponding pixel in the original image
                if np.abs(np.linalg.det(warp_matrix)) < 1e-6:
                    warp_mat_inv = np.linalg.pinv(warp_matrix)
                else:
                    warp_mat_inv = np.linalg.inv(warp_matrix)

                original_coords = np.dot(warp_mat_inv, np.array([x, y, 1]))
                original_x, original_y = original_coords[:2] / original_coords[2]
                
                if 0 <= original_x <= w-1 and 0 <= original_y <= h-1:
                    # Used bilinear interpolation to get the pixel value at the non-integer coordinates
                    x0, y0 = int(original_x), int(original_y)
                    x1, y1 = x0 + 1, y0 + 1

                    if x1 >= w:
                        x1 = w-1
                    if y1 >= h:
                        y1 = h-1

                    dx, dy = original_x - x0, original_y - y0
                    
                    #bilinear interpolation for each channel
                    for channel in range(3):
                        value = (1 - dx) * (1 - dy) * image[y0, x0, channel] + \
                                dx * (1 - dy) * image[y0, x1, channel] + \
                                (1 - dx) * dy * image[y1, x0, channel] + \
                                dx * dy * image[y1, x1, channel]
                        warped_image[y, x, channel] = value
    return warped_image

def affineTransform(source_pts, target_pts):
    source_pts_hom = []
    for x,y in source_pts:
        source_pts_hom.append((x,y,1))
    
    M = np.linalg.solve(source_pts_hom, target_pts).T
    return M

def apply_affine_matrix(affine_matrix, point):
    a, b, tx = affine_matrix[0]
    c, d, ty = affine_matrix[1]
    x, y = point
    x_new = a * x + b * y + tx
    y_new = c * x + d * y + ty
    return (int(x_new), int(y_new))

def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)
 
def isInside(x1, y1, x2, y2, x3, y3, x, y):
    A = area (x1, y1, x2, y2, x3, y3)
 
    A1 = area (x, y, x2, y2, x3, y3)
    A2 = area (x1, y1, x, y, x3, y3)
    A3 = area (x1, y1, x2, y2, x, y)

    if(A == A1 + A2 + A3):
        return True
    else:
        return False

def check_if_inside(points, triangle_points):
    indices = []

    for i in range(len(points)):
        x, y = points[i]
        x1, y1 = triangle_points[0]
        x2, y2 = triangle_points[1]
        x3, y3 = triangle_points[2]

        if isInside(x1, y1, x2, y2, x3, y3, x, y) == True:
            indices.append(i)
    return indices


# I1_prime and I3_prime preparation

warped_image_I1 = np.zeros_like(I1, dtype=np.uint8)
warped_image_I3 = np.zeros_like(I3, dtype=np.uint8)

points_I3_new = points_I3.copy()

for triangle1 in tri_I1.simplices:
    source_triangle = np.float32(points_I1[triangle1])
    target_triangle = np.float32(points_I2[triangle1])

    matrix = affineTransform(source_triangle, target_triangle)

    indices_to_change_I3 = check_if_inside(points_I3, points_I1[triangle1])

    for index in indices_to_change_I3:
        points_I3_new[index] = apply_affine_matrix(matrix, points_I3[index])

    warp_tri_I1 = warp_triangle(I1, points_I2[triangle1], matrix)
    warp_tri_I2 = warp_triangle(I3, points_I2[triangle1], matrix)

    warped_image_I1 = np.bitwise_or(warped_image_I1, warp_tri_I1)
    warped_image_I3 = np.bitwise_or(warped_image_I3, warp_tri_I2)
    

cv2.imwrite(f_path + "I1_prime.jpg", warped_image_I1)
cv2.imwrite(f_path + "I3_prime.jpg", warped_image_I3)


I1_prime = warped_image_I1.copy()
I3_prime = warped_image_I3.copy()

points_I3_new = np.array(points_I3_new)


# I4 computation

warped_image_I4 = np.zeros_like(I2, dtype=np.uint8)

for triangle1 in tri_I2.simplices:
    source_triangle = np.float32(points_I2[triangle1])
    target_triangle = np.float32(points_I3_new[triangle1])

    matrix = affineTransform(source_triangle, target_triangle)

    warp_tri_I4 = warp_triangle(I2, points_I3_new[triangle1], matrix)

    warped_image_I4 = np.bitwise_or(warped_image_I4, warp_tri_I4)

cv2.imwrite(f_path + "I4.jpg", warped_image_I4)



