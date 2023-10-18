import cv2
import numpy as np


def select_anchor_points(image, pts, image_name, f_path):
    anchor_points = []
    sample = image.copy()

    points = 0
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            anchor_points.append((x, y))
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
            cv2.imwrite(f_path + image_name + f'_{pts}_marked.jpg', sample)
            cv2.destroyAllWindows()
            break

    cv2.imwrite(f_path + image_name + f'_{pts}_marked.jpg', sample)
    cv2.destroyAllWindows()
    return anchor_points

def corners(image):
    l = []
    l.append((0, 0))
    l.append((0, image.shape[0]-1))
    l.append((image.shape[1]-1,image.shape[0]-1))
    l.append((image.shape[1]-1, 0))
    return l

def area(x1, y1, x2, y2, x3, y3):
 
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)
 
def isInside(x1, y1, x2, y2, x3, y3, x, y):
    A = area(x1, y1, x2, y2, x3, y3)

    A1 = area(x, y, x2, y2, x3, y3)
    A2 = area(x1, y1, x, y, x3, y3)
    A3 = area(x1, y1, x2, y2, x, y)

    if(A == A1 + A2 + A3 and A != 0):
        return True
    else:
        return False

def compute_barycentric_coordinates(P, A, B, C):
    triangle_matrix = np.column_stack((A, B, C))

    barycentric_coords = np.linalg.solve(triangle_matrix, P)
    return barycentric_coords

def triangle_map(anchor_image1, anchor_image2):
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
            if isInside(Ax, Ay, Bx, By, Cx, Cy, x, y):
                u, v, w = compute_barycentric_coordinates(np.array([x, y, 1]), np.array([Ax, Ay, 1]), np.array([Bx, By, 1]), np.array([Cx, Cy, 1]))

                Px = u * A[0] + v * B[0] + w * C[0]
                Py = u * A[1] + v * B[1] + w * C[1]
                
                pixel_map[(y, x)] = (int(Py), int(Px))

    return pixel_map

def transfer_color(source_image, pixel_map, target_image, target_image_inv, triangle, white_mask):
    target_pix = pixel_map.keys()

    for pix in target_pix:
        target_image[int(pix[0]), int(pix[1])] = source_image[int(pixel_map[pix][0]), int(pixel_map[pix][1])]

        if np.any((triangle == 0) | (triangle == 1) | (triangle == 2) | (triangle == 3)):
            continue
        else:
            target_image_inv[int(pix[0]), int(pix[1])] = np.array([0, 0, 0])
            white_mask[int(pix[0]), int(pix[1])] = np.uint8(255)

    return target_image, target_image_inv, white_mask

def blur_image(image, kernel_size):
    channels = 1

    pad_x = kernel_size[0] // 2
    pad_y = kernel_size[1] // 2
    
    output_image = np.zeros_like(image)
    
    for c in range(channels):
        channel = image.copy()
        channel_padded = np.pad(channel, ((pad_x, pad_x), (pad_y, pad_y)), mode='edge')
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output_image[i, j] = np.mean(channel_padded[i:i+kernel_size[0], j:j+kernel_size[1]])
    
    return output_image

def find_closest_lab_color(target_lab, color_list_lab):
    target_L = target_lab[0][0]
    color_L_values = np.array([color[0] for color in color_list_lab])

    threshold = 200
    l_diffs = np.abs(color_L_values - target_L)
    closest_index = np.argmin(l_diffs)


    if l_diffs[closest_index] <= threshold:
        closest_color = color_list_lab[closest_index]
    else:
        closest_color = target_lab[0]
    
    return closest_color

def compute_affine_matrix(src_points, dst_points):
    # source_pts_hom = []
    # for x,y in src_points:
    #     source_pts_hom.append((x,y,1))
    
    # M = np.linalg.solve(source_pts_hom, dst_points).T
    # return M

    # M = cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))

    # return M


    # A = np.zeros((8, 9))

    # for i in range(4):
    #     x, y = src_points[i]
    #     u, v = dst_points[i]
    #     A[i * 2] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
    #     A[i * 2 + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]

    # _, _, V = np.linalg.svd(A)
    # h = V[-1, :]

    # # Reshape the vector h into a 3x3 matrix
    # affine_matrix = h.reshape(3, 3)

    # affine_matrix = affine_matrix / affine_matrix[2][2]

    # affine_matrix[2] = np.array([0, 0, 1])

    # A = np.zeros((8, 6))
    # b = np.zeros((8, 1))

    # for i in range(4):
    #     x, y = src_points[i]
    #     u, v = dst_points[i]
    #     A[i * 2] = [x, y, 0, 0, 1, 0]
    #     A[i * 2 + 1] = [0, 0, x, y, 0, 1]
    #     b[i * 2] = u
    #     b[i * 2 + 1] = v

    # affine_matrix, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # affine_matrix = np.vstack((affine_matrix, [0, 0, 1]))
    # print(affine_matrix.shape)

    A = np.zeros((8, 6))
    for i in range(4):
        A[i * 2, :2] = src_points[i]
        A[i * 2, 2] = 1
        A[i * 2 + 1, 3:5] = src_points[i]
        A[i * 2 + 1, 5] = 1

    # Create a vector B
    B = dst_points.reshape(8)

    # Solve for the transformation matrix using the least squares solution
    X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    # Create the affine transformation matrix
    M = np.append(X, [0, 0, 1]).reshape(3, 3)

    X = X.reshape(2, 3)

    return M, X

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
            
            h_i, w_i, _ = image.shape

            if 0 <= original_x <= w_i-1 and 0 <= original_y <= h_i-1:
                # Used bilinear interpolation to get the pixel value at the non-integer coordinates
                x0, y0 = int(original_x), int(original_y)
                x1, y1 = x0 + 1, y0 + 1
                dx, dy = original_x - x0, original_y - y0

                if x1 == w_i:
                    x1 = w_i - 1
                if y1 == h_i:
                    y1 = h_i - 1
                
                #bilinear interpolation for each channel
                for channel in range(3):
                    value = (1 - dx) * (1 - dy) * image[y0, x0, channel] + \
                            dx * (1 - dy) * image[y0, x1, channel] + \
                            (1 - dx) * dy * image[y1, x0, channel] + \
                            dx * dy * image[y1, x1, channel]
                    warped_image[y, x, channel] = value
    return warped_image


def histogram_match(source_image, target_image):
    if len(source_image.shape) == 3:
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    if len(target_image.shape) == 3:
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # source_hist = cv2.calcHist([source_image], [0], None, [256], [0, 256])
    # target_hist = cv2.calcHist([target_image], [0], None, [256], [0, 256])

    source_hist, _ = np.histogram(source_image.flatten(), bins=256, range=(0, 256))
    target_hist, _ = np.histogram(target_image.flatten(), bins=256, range=(0, 256))

    source_hist = np.float64(source_hist)
    target_hist = np.float64(target_hist)

    # Hist Norm
    source_hist /= np.float64(source_image.size)
    target_hist /= np.float64(target_image.size)

    # CDF
    source_cdf = np.cumsum(source_hist)
    target_cdf = np.cumsum(target_hist)

    # Mapping from source to target pixel values
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = 0
        while j < 256 and source_cdf[i] > target_cdf[j]:
            j += 1
        mapping[i] = j

    matched_image = mapping[source_image]

    return matched_image

def histogram_match_color(source_image, target_image):

    source_image_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
    target_image_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB)

    source_l, source_a, source_b = cv2.split(source_image_lab)
    target_l, target_a, target_b = cv2.split(target_image_lab)

    matched_l = histogram_match(source_l, target_l)

    matched_image_lab = cv2.merge((matched_l, source_a, source_b))

    matched_image = cv2.cvtColor(matched_image_lab, cv2.COLOR_LAB2BGR)

    return matched_image
