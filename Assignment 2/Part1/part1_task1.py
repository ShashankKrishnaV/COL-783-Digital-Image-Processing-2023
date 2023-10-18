import os
import cv2
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

# Task 1

anchor_sets = {}

#select anchor points interactively using the mouse
def select_anchor_points(image, image_name):
    anchor_points = []
    sample = image.copy()

    points = 0
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            anchor_points.append((x, y))
            cv2.circle(sample, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(sample, str(len(anchor_points)), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Select Three Points', sample)
            if len(anchor_points) == 3:
                return


    cv2.imshow('Select Three Points', image)
    cv2.setMouseCallback('Select Three Points', on_mouse)

    while len(anchor_points) < 3:
        if cv2.waitKey(1) == 27 or len(anchor_points) == 3: 
            cv2.imwrite(path + image_name + '_3_marked.jpg', sample)
            cv2.destroyAllWindows()
            break

    cv2.imwrite(path + image_name + '_3_marked.jpg', sample)
    cv2.destroyAllWindows()
    return anchor_points


#Manual annotation
image = I1
anchor_points = select_anchor_points(image, 'I1')
anchor_sets['I1'] = anchor_points

image = I2
anchor_points = select_anchor_points(image, 'I2')
anchor_sets['I2'] = anchor_points

image = I3
anchor_points = select_anchor_points(image, 'I3')
anchor_sets['I3'] = anchor_points

#set 1 points
# anchor_sets = {'I1': [(76, 150), (257, 154), (163, 279)], 'I2': [(70, 151), (247, 142), (165, 266)], 'I3': [(73, 146), (265, 146), (165, 278)]}

print('Selected Points: ', anchor_sets)
print()

with open(path + 'anchor_points_task1.txt', 'w+') as f:
    f.write(str(anchor_sets))

# Release OpenCV windows and resources
cv2.destroyAllWindows()


#2D warp matrix : affine transform
def affineTransform(source_pts, target_pts):
    source_pts_hom = []
    for x,y in source_pts:
        source_pts_hom.append((x,y,1))
    
    M = np.linalg.solve(source_pts_hom, target_pts).T
    return M


#sub-task 1
def warp(image, warp_matrix):
    h, w, _ = image.shape
    warped_image = np.zeros((h, w, 3), dtype=np.uint8)

    #2x3 matrix to 3x3 matrix
    warp_matrix = np.vstack([warp_matrix, [0, 0, 1]])


    for y in range(h):
        for x in range(w):
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
                dx, dy = original_x - x0, original_y - y0
                
                #bilinear interpolation for each channel
                for channel in range(3):
                    value = (1 - dx) * (1 - dy) * image[y0, x0, channel] + \
                            dx * (1 - dy) * image[y0, x1, channel] + \
                            (1 - dx) * dy * image[y1, x0, channel] + \
                            dx * dy * image[y1, x1, channel]
                    warped_image[y, x, channel] = value
    return warped_image


def warp_points(image, anchor_points, warp_matrix):
    new_set = []

    h, w, _ = image.shape
    x_min, x_max = 0, w-1
    y_min, y_max = 0, h-1

    warp_matrix = np.vstack([warp_matrix, [0, 0, 1]])

    for anchor_point in anchor_points:
        # if np.abs(np.linalg.det(warp_matrix)) < 1e-6:
        #     warp_mat_inv = np.linalg.pinv(warp_matrix)
        # else:
        #     warp_mat_inv = np.linalg.inv(warp_matrix)

        coords = np.dot(warp_matrix, np.array([anchor_point[0], anchor_point[1], 1]))
        new_x, new_y = coords[:2] / coords[2]
        
        # if 0 <= new_x < w - 1 and 0 <= new_y < h - 1:
        new_x = max(x_min, min(x_max, new_x))
        new_y = max(y_min, min(y_max, new_y))
        new_set.append((new_x, new_y))
    return new_set

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


#sub-task 2
source_triangle = np.array(anchor_sets['I1'], dtype=np.float32)
target_triangle = np.array(anchor_sets['I2'], dtype=np.float32)

warp_mat_H1 = affineTransform(source_triangle, target_triangle)

I1_prime = warp(I1, warp_mat_H1)
cv2.imwrite(path + 'I1_prime_task1.jpg', I1_prime)

I3_prime = warp(I3, warp_mat_H1)
cv2.imwrite(path + 'I3_prime_task1.jpg', I3_prime)


anchor_points_I1_prime = np.array(warp_points(I1_prime, anchor_sets['I1'], warp_mat_H1), dtype=np.float32)
anchor_points_I3_prime = np.array(warp_points(I3_prime, anchor_sets['I3'], warp_mat_H1), dtype=np.float32)


### I1 prime and I3 prime only triangles plot

A, B, C = np.array(anchor_points_I1_prime, dtype=np.int32)
A, B, C = list(A), list(B), list(C)

min_x, max_x = min(A[0], B[0], C[0]), max(A[0], B[0], C[0])
min_y, max_y = min(A[1], B[1], C[1]), max(A[1], B[1], C[1])

I1_prime_tri = np.zeros_like(I1_prime, dtype=np.uint8)

for y in range(min_y, max_y + 1):
    for x in range(min_x, max_x + 1):
        if isInside(A[0], A[1], B[0], B[1], C[0], C[1], x, y):
            I1_prime_tri[y, x] = I1_prime[y,x]

cv2.imwrite(path + 'I1_prime_tri_task1.jpg', I1_prime_tri)

A, B, C = np.array(anchor_points_I3_prime, dtype=np.int32)
A, B, C = list(A), list(B), list(C)

min_x, max_x = min(A[0], B[0], C[0]), max(A[0], B[0], C[0])
min_y, max_y = min(A[1], B[1], C[1]), max(A[1], B[1], C[1])

I3_prime_tri = np.zeros_like(I3_prime, dtype=np.uint8)

for y in range(min_y, max_y + 1):
    for x in range(min_x, max_x + 1):
        if isInside(A[0], A[1], B[0], B[1], C[0], C[1], x, y):
            I3_prime_tri[y, x] = I3_prime[y,x]

cv2.imwrite(path + 'I3_prime_tri_task1.jpg', I3_prime_tri)

###

# print(anchor_points_I1_dash, anchor_points_I3_dash)

warp_mat_H2 = affineTransform(anchor_points_I1_prime, anchor_points_I3_prime)

# print(warp_mat_H2)

I4 = warp(I2, warp_mat_H2)
cv2.imwrite(path + 'I4_image_task1.jpg', I4)


#Extract triangle
points_I2 = anchor_sets["I2"]


A, B, C = points_I2
A, B, C = list(A), list(B), list(C)

min_x, max_x = min(A[0], B[0], C[0]), max(A[0], B[0], C[0])
min_y, max_y = min(A[1], B[1], C[1]), max(A[1], B[1], C[1])

I2_without_tri = I2.copy()
I2_tri = np.zeros_like(I2, dtype=np.uint8)

for y in range(min_y, max_y + 1):
    for x in range(min_x, max_x + 1):
        if isInside(A[0], A[1], B[0], B[1], C[0], C[1], x, y):
            I2_tri[y, x] = I2_without_tri[y,x]
            I2_without_tri[y,x] = np.array([0, 0, 0])


I4_only_tri = warp(I2_tri, warp_mat_H2)
cv2.imwrite(path + "I4_tri_task1.jpg", I4_only_tri)


black_mask = (I4_only_tri == [0, 0, 0]).all(axis=2)
I4_only_tri[black_mask] = I2_without_tri[black_mask]

# I4_tri_task1 = np.bitwise_or(I4_only_tri, I2_without_tri)
cv2.imwrite(path + "I4_task1.jpg", I4_only_tri)






