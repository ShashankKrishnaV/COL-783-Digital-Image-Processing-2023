import dlib
import cv2
import os
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from functions_faceswap import select_anchor_points, transfer_color, triangle_map, compute_barycentric_coordinates, isInside, corners, blur_image, find_closest_lab_color, histogram_match_color, compute_affine_matrix, warp 

path = input('Enter the path for two images (I1 and I2): ')
path = path + '/'

image1 = cv2.imread(path + "I1.jpg")
image2 = cv2.imread(path + "I2.jpg")

path = path + 'face_swap'
os.makedirs(path, exist_ok=True)
path += '/' 


# Global pose alignment
print('Select 4 corresponding points on 4 sides of the face for pose alignment\n')

image = image1
orient_points_I1 = np.array(select_anchor_points(image, 4, 'I1', path), dtype=np.float32)

image = image2
orient_points_I2 = np.array(select_anchor_points(image, 4, 'I2', path), dtype=np.float32)


affine_mat_12, rot_12 = compute_affine_matrix(orient_points_I1, orient_points_I2)
affine_mat_21, rot_21 = compute_affine_matrix(orient_points_I1, orient_points_I1)


image1_aligned = warp(image1.copy(), affine_mat_12, (image1.shape[0], image1.shape[1], image1.shape[2]))
image2_aligned = warp(image2.copy(), affine_mat_21, (image2.shape[0], image2.shape[1], image2.shape[2]))

cv2.imwrite(path + 'I1_aligned_to_I2.jpg', image1_aligned)
cv2.imwrite(path + 'I2_aligned_to_I1.jpg', image2_aligned)



##### Landmarks selection #####
option = int(input('\nPoint selection: 1. Automatic 2. Select manually: '))

face1_points = None
face2_points = None

if int(option) == 1:
    # Load the dlib facial landmark predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Load the face detector
    detector = dlib.get_frontal_face_detector()

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    faces1 = detector(gray1)
    faces2 = detector(gray2)

    landmarked_image1 = np.copy(image1)
    landmarked_image2 = np.copy(image2)

    face1_points = []
    face2_points = []

    for face in faces1:
        landmarks = predictor(gray1, face)

        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(landmarked_image1, (x, y), 2, (0, 255, 0), -1)

            face1_points.append((x,y))

    for face in faces2:
        landmarks = predictor(gray2, face)

        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(landmarked_image2, (x, y), 2, (0, 255, 0), -1)

            face2_points.append((x,y))

    cv2.imwrite(path + "I1_landmarks.jpg", landmarked_image1)
    cv2.imwrite(path + "I2_landmarks.jpg", landmarked_image2)

    cv2.destroyAllWindows()

else:
    def select_anchor_points(image, pts, image_name):
        anchor_points = []
        sample = image.copy()

        points = 0
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                anchor_points.append((x, y))
                cv2.circle(sample, (x, y), 3, (0, 0, 255), -1)
                cv2.putText(sample, str(len(anchor_points)), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow('Select Three Points', sample)
                if len(anchor_points) == pts:
                    return


        cv2.imshow('Select Three Points', image)
        cv2.setMouseCallback('Select Three Points', on_mouse)

        while len(anchor_points) < pts:
            if cv2.waitKey(1) == 27 or len(anchor_points) == pts: 
                cv2.imwrite(path + image_name + '_marked.jpg', sample)
                cv2.destroyAllWindows()
                break

        cv2.imwrite(path + image_name + '_marked.jpg', sample)
        cv2.destroyAllWindows()
        return anchor_points

    pts = int(input('Enter the number of points you want to choose: '))
    image = image1_aligned.copy()
    face1_points = select_anchor_points(image, pts, 'I1')

    image = image2_aligned.copy()
    face2_points = select_anchor_points(image, pts, 'I2')

cv2.destroyAllWindows()


face1_points_pre = face1_points.copy()
face2_points_pre = face2_points.copy()

face1_points_corners = corners(image1) + face1_points
face2_points_corners = corners(image2) + face2_points

face1_points = np.array(face1_points_corners)
face2_points = np.array(face2_points_corners)

tri1 = Delaunay(face1_points)
tri2 = Delaunay(face2_points)

tri1_pre = Delaunay(face1_points_pre)
tri2_pre = Delaunay(face2_points_pre)


##### Plotting delaunay triangulation #####
points = np.array(face1_points_pre)

# Plot the Delaunay triangulation
plt.triplot(points[:, 0], points[:, 1], tri1_pre.simplices.copy())
plt.plot(points[:, 0], points[:, 1], 'o')  # Plot the points

plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

plt.savefig(path + "I1_delaunay.jpg", dpi=300, bbox_inches='tight')
plt.clf()

points = np.array(face2_points_pre)

plt.triplot(points[:, 0], points[:, 1], tri2_pre.simplices.copy())
plt.plot(points[:, 0], points[:, 1], 'o')  # Plot the points

plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

plt.savefig(path + "I2_delaunay.jpg", dpi=300, bbox_inches='tight')
plt.clf()

#####


#### Point selection done and now the procedure starts ####

face1_points_new = []
face2_points_new = []

for i in range(len(face1_points)):
    if i>3:
        x, y = face1_points[i]
        pt = np.array([x, y, 1], dtype=np.float32)
        tar_pt = np.dot(affine_mat_12, pt.T)
        # print(pt.T.shape, tar_pt.shape)
        face1_points_new.append(np.array([int(tar_pt[0]/tar_pt[2]), int(tar_pt[1]/tar_pt[2])]))
    else:
        face1_points_new.append(np.array(face1_points[i]))

for i in range(len(face2_points)):
    if i>3:
        x, y = face2_points[i]
        pt = np.array([x, y, 1], dtype=np.float32)
        tar_pt = np.dot(affine_mat_21, pt.T)
        face2_points_new.append(np.array([int(tar_pt[0]/tar_pt[2]), int(tar_pt[1]/tar_pt[2])]))
    else:
        face2_points_new.append(np.array(face2_points[i]))



face1_points_new = np.array(face1_points_new)
face2_points_new = np.array(face2_points_new)

tri1_new = Delaunay(face1_points_new)
tri2_new = Delaunay(face2_points_new)


# Prepare warped images
mask_image1 = np.zeros_like(image2, dtype=np.uint8)
mask_image2_inv = image2.copy()

white_mask_2 = np.zeros(image2.shape[:2], dtype=np.uint8)

for triangle in tri1_new.simplices:
    map_pts = triangle_map(face1_points_new[triangle], face2_points[triangle])
    mask_image1, mask_image2_inv, white_mask_2 = transfer_color(image1_aligned.copy(), map_pts, mask_image1, mask_image2_inv, triangle, white_mask_2)

cv2.imwrite(path + "I1_face_warped.jpg", mask_image1)
cv2.imwrite(path + "I2_without_face.jpg", mask_image2_inv)


mask_image2 = np.zeros_like(image1, dtype=np.uint8)
mask_image1_inv = image1.copy()

white_mask_1 = np.zeros(image1.shape[:2], dtype=np.uint8)

for triangle in tri2_new.simplices:
    map_pts = triangle_map(face2_points_new[triangle], face1_points[triangle])
    mask_image2, mask_image1_inv, white_mask_1 = transfer_color(image2_aligned.copy(), map_pts, mask_image2, mask_image1_inv, triangle, white_mask_1)

cv2.imwrite(path + "I2_face_warped.jpg", mask_image2)
cv2.imwrite(path + "I1_without_face.jpg", mask_image1_inv)

white_mask_1 = blur_image(white_mask_1, (7,7))
white_mask_2 = blur_image(white_mask_2, (7,7))

cv2.imwrite(path + 'white_mask_1.jpg', white_mask_1)
cv2.imwrite(path + 'white_mask_2.jpg', white_mask_2)



# Face Swap step
white_mask_1 = white_mask_1 / 255.0
white_mask_2 = white_mask_2 / 255.0

mask_1 = np.stack((white_mask_1, white_mask_1, white_mask_1), axis=-1)
mask_2 = np.stack((white_mask_2, white_mask_2, white_mask_2), axis=-1)

image1_swapped = (image1 * (1 - mask_1) + mask_image2 * mask_1).astype(np.uint8)
image2_swapped = (image2 * (1 - mask_2) + mask_image1 * mask_2).astype(np.uint8)

cv2.imwrite(path + "I1_swapped.jpg", image1_swapped)
cv2.imwrite(path + "I2_swapped.jpg", image2_swapped)



# Color transfer step
def select_points(image, pts):
    points = []
    sample = image.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(sample, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow('Select Points to Color', sample)
            if len(points) == pts:
                return

    cv2.imshow('Select Points to Color', image)
    cv2.setMouseCallback('Select Points to Color', on_mouse)

    while len(points) < pts:
        if cv2.waitKey(1) == 27 or len(points) == pts: 
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()
    return points


use_hist_match = int(input('\nUse histogram matching: 1. Yes, 2. No: '))


#Image 1 face
rectangular_points = select_points(image1, 2)
image = image1.copy()

image_lab = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2LAB)

top_left =  rectangular_points[0] # Replace with your coordinates
bottom_right = rectangular_points[1]  # Replace with your coordinates

roi = image_lab[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

height, width, _ = roi.shape
sampled_colors = []

num_colors = 50

for _ in range(num_colors):
    y = np.random.randint(0, height)
    x = np.random.randint(0, width)
    color = roi[y, x]
    sampled_colors.append(color)


if use_hist_match == 1:
    ###Extra histogram match for better colors###

    roi_bgr = cv2.cvtColor(roi.copy(), cv2.COLOR_LAB2BGR)

    mask_image2_new = histogram_match_color(mask_image2.copy(), image.copy())

    mask_image2 = mask_image2_new.copy()

    #####


# mask_image2_colored = cv2.cvtColor(mask_image2.copy(), cv2.COLOR_BGR2LAB)
mask_image2_colored = cv2.cvtColor(mask_image2.copy(), cv2.COLOR_BGR2LAB)


height, width, _ = image_lab.shape
for y in range(height):
    for x in range(width):
        lab_pixel = mask_image2_colored[y, x]
        
        if lab_pixel[0] > 0:
            closest_color = find_closest_lab_color(np.array([lab_pixel]), sampled_colors)
            
            mask_image2_colored[y, x, 1] = closest_color[1]  
            mask_image2_colored[y, x, 2] = closest_color[2] 

mask_image2_colored_bgr = cv2.cvtColor(mask_image2_colored, cv2.COLOR_LAB2BGR)

cv2.imwrite(path + "I2_face_colored.jpg", mask_image2_colored_bgr)


# image1_swapped_corrected = np.bitwise_or(mask_image2_inv, mask_image1_colored_bgr)

image1_swapped_corrected = (image1 * (1 - mask_1) + mask_image2_colored_bgr * mask_1).astype(np.uint8)
cv2.imwrite(path + "I1_swapped_colored.jpg", image1_swapped_corrected)



#Image 2 face
rectangular_points = select_points(image2, 2)
image = image2.copy()

image_lab = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2LAB)

top_left =  rectangular_points[0] # Replace with your coordinates
bottom_right = rectangular_points[1]  # Replace with your coordinates

roi = image_lab[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

height, width, _ = roi.shape
sampled_colors.clear()
sampled_colors = []

num_colors = 50

for _ in range(num_colors):
    y = np.random.randint(0, height)
    x = np.random.randint(0, width)
    color = roi[y, x]
    sampled_colors.append(color)



if use_hist_match == 1:
    ###Extra histogram match for better colors###

    roi_bgr = cv2.cvtColor(roi.copy(), cv2.COLOR_LAB2BGR)

    mask_image1_new = histogram_match_color(mask_image1.copy(), image.copy())

    mask_image1 = mask_image1_new.copy()

    #####


# mask_image1_colored = cv2.cvtColor(mask_image1.copy(), cv2.COLOR_BGR2LAB)
mask_image1_colored = cv2.cvtColor(mask_image1.copy(), cv2.COLOR_BGR2LAB)

height, width, _ = image_lab.shape
for y in range(height):
    for x in range(width):
        lab_pixel = mask_image1_colored[y, x]
        
        if lab_pixel[0] > 0:
            closest_color = find_closest_lab_color(np.array([lab_pixel]), sampled_colors)
            
            mask_image1_colored[y, x, 1] = closest_color[1]
            mask_image1_colored[y, x, 2] = closest_color[2]

# Convert the result image back to BGR format
mask_image1_colored_bgr = cv2.cvtColor(mask_image1_colored, cv2.COLOR_LAB2BGR)

cv2.imwrite(path + "I1_face_colored.jpg", mask_image1_colored_bgr)


# image1_swapped_corrected = np.bitwise_or(mask_image2_inv, mask_image1_colored_bgr)

image2_swapped_corrected = (image2 * (1 - mask_2) + mask_image1_colored_bgr * mask_2).astype(np.uint8)
cv2.imwrite(path + "I2_swapped_colored.jpg", image2_swapped_corrected)

