import numpy as np
import cv2

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