import numpy as np
import cv2

# Recursive use of task 2

def compute_affine_matrix(src_points, dst_points):

    A = np.zeros((8, 6))
    for i in range(4):
        A[i * 2, :2] = src_points[i]
        A[i * 2, 2] = 1
        A[i * 2 + 1, 3:5] = src_points[i]
        A[i * 2 + 1, 5] = 1

    B = dst_points.reshape(8)

    # Solve for the transformation matrix using the least squares solution
    X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    # Create the affine transformation matrix
    M = np.append(X, [0, 0, 1]).reshape(3, 3)

    return M

def affine_transform(image, affine_matrix, output_shape):
    transformed_image = np.zeros(output_shape, dtype=np.uint8)

    if np.abs(np.linalg.det(affine_matrix)) < 1e-6:
        inverse_affine_matrix = np.linalg.pinv(affine_matrix)
    else:
        inverse_affine_matrix = np.linalg.inv(affine_matrix)

    input_height, input_width = image.shape[:2]
    output_height, output_width = output_shape[0], output_shape[1]

    # print(input_height, input_width, output_height, output_width)

    for y_out in range(output_height):
        for x_out in range(output_width):
            # Map the output coordinates back to the input image using the inverse transformation
            coordinates = np.dot(inverse_affine_matrix, [x_out, y_out, 1])
            x_in, y_in = coordinates[:2] / coordinates[2]

            if 0 <= x_in < input_width and 0 <= y_in < input_height:
                x1, y1 = int(x_in), int(y_in)
                x2, y2 = x1 + 1, y1 + 1

                if x2 == input_width:
                    x2 = input_width - 1
                if y2 == input_height:
                    y2 = input_height - 1

                alpha = x_in - x1
                beta = y_in - y1
                top_left = image[y1, x1]
                top_right = image[y1, x2]
                bottom_left = image[y2, x1]
                bottom_right = image[y2, x2]
                interpolated_value = (1 - alpha) * (1 - beta) * top_left + alpha * (1 - beta) * top_right + (1 - alpha) * beta * bottom_left + alpha * beta * bottom_right

                transformed_image[y_out, x_out] = interpolated_value

    # cv2.imwrite('image.jpg', image)

    return transformed_image

def warp(image, warp_matrix, output_shape):
    h, w, _ = output_shape
    warped_image = np.zeros((h, w, 3), dtype=np.uint8)

    #2x3 matrix to 3x3 matrix
    if warp_matrix.shape[0] == 2:
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

                if x1 == w:
                    x1 = w - 1
                if y1 == h:
                    y1 = h - 1
                
                #bilinear interpolation for each channel
                for channel in range(3):
                    value = (1 - dx) * (1 - dy) * image[y0, x0, channel] + \
                            dx * (1 - dy) * image[y0, x1, channel] + \
                            (1 - dx) * dy * image[y1, x0, channel] + \
                            dx * dy * image[y1, x1, channel]
                    warped_image[y, x, channel] = value
    return warped_image


def compute_barycentric_coordinates(P, A, B, C):
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

    return [int(Px), int(Py)]


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
                
                Px = u * A[0] + v * B[0] + w * C[0]
                Py = u * A[1] + v * B[1] + w * C[1]

                pixel_map[(y, x)] = [int(Py), int(Px)]

    return pixel_map

def transfer_color(source_image, pixel_map, target_image):
    target_pix = pixel_map.keys()

    for pix in target_pix:
        target_image[int(pix[0]), int(pix[1])] = source_image[int(pixel_map[pix][0]), int(pixel_map[pix][1])]

    return target_image