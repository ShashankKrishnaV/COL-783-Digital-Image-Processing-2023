import os
import cv2
import json
import math
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from functions_bary import compute_barycentric_coordinates, area, check_if_inside, isInside, triangle_map, triangle_map_point, transfer_color, compute_affine_matrix, affine_transform, warp

path = input('Enter the path for images (I1, I2, I3): ')
path = path + '/'

I1 = cv2.imread(path + 'I1.jpg')
I2 = cv2.imread(path + 'I2.jpg')
I3 = cv2.imread(path + 'I3.jpg')

pts = int(input('Number of correspondences: '))

f_path = path + f'task3_2/{pts}'
os.makedirs(f_path, exist_ok=True)
f_path += '/'

# Resize if shapes are not same
h2, w2, _ = I2.shape
I1 = cv2.resize(I1.copy(), (w2, h2))
I3 = cv2.resize(I3.copy(), (w2, h2))


# Task 3 - 2

anchor_sets = {}

#select anchor points
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
            cv2.imwrite(f_path + image_name + f'_{pts}_marked.jpg', sample)
            cv2.destroyAllWindows()
            break

    cv2.imwrite(f_path + image_name + f'_{pts}_marked.jpg', sample)
    cv2.destroyAllWindows()
    return anchor_points

def corners(image):
    l = []
    l.append([0, 0])
    l.append([0, image.shape[0]-1])
    l.append([image.shape[1]-1,image.shape[0]-1])
    l.append([image.shape[1]-1, 0])
    return l


# Pose and size approximation of I1 and I3 to I2
image = I1
orient_points_I1 = select_anchor_points(image, 4, 'I1')

image = I2
orient_points_I2 = select_anchor_points(image, 4, 'I2')

image = I3
orient_points_I3 = select_anchor_points(image, 4, 'I3')

affine_mat_12 = compute_affine_matrix(np.array(orient_points_I1, dtype=np.float32), np.array(orient_points_I2,  dtype=np.float32))
affine_mat_32 = compute_affine_matrix(np.array(orient_points_I3, dtype=np.float32), np.array(orient_points_I2,  dtype=np.float32))


I1_prime_trans = warp(I1.copy(), affine_mat_12, (I2.shape[0], I2.shape[1], I2.shape[2]))
I3_prime_trans = warp(I3.copy(), affine_mat_32, (I2.shape[0], I2.shape[1], I2.shape[2]))

cv2.imwrite(f_path + 'I1_prime_trans.jpg', I1_prime_trans)
cv2.imwrite(f_path + 'I3_prime_trans.jpg', I3_prime_trans)



#Ablation on points
# pts = [1, 5, 10, 20, 30, 35]

print(f'Select the correpondences now. Number of points to mark: {pts}')

anchor_sets = {}

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



# anchor_sets = None
# with open(f_path + f'markers_{pts}_points.txt', 'r') as f:
#     line = ((f.readlines())[0])
#     line = line.replace("(", "[").replace(")", "]").replace("'", "\"")
#     anchor_sets = json.loads(str(line))


points_I1_new = []
points_I3_new = []

face1_points = anchor_sets["I1"]
face3_points = anchor_sets["I3"]

for i in range(len(face1_points)):
    x, y = face1_points[i]
    pt = np.array([x, y, 1], dtype=np.float32)
    tar_pt = np.dot(affine_mat_12, pt.T)
    points_I1_new.append(np.array([int(tar_pt[0]/tar_pt[2]), int(tar_pt[1]/tar_pt[2])]))

for i in range(len(face3_points)):
    x, y = face3_points[i]
    pt = np.array([x, y, 1], dtype=np.float32)
    tar_pt = np.dot(affine_mat_32, pt.T)
    points_I3_new.append(np.array([int(tar_pt[0]/tar_pt[2]), int(tar_pt[1]/tar_pt[2])]))


# Delaunay plot
points_I1 = corners(I1) + anchor_sets["I1"]
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

points_I2 = corners(I2) + anchor_sets["I2"]
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

points_I3 = corners(I3) + anchor_sets["I3"]
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
plt.clf()




anchor_sets["I1"] = points_I1_new
anchor_sets["I3"] = points_I3_new


I1 = np.array(I1_prime_trans.copy(), dtype=np.uint8)
I3 = np.array(I3_prime_trans.copy(), dtype=np.uint8)



diff_vec = []

for pt1, pt2 in zip(anchor_sets["I1"], anchor_sets["I3"]):
	diff_vec.append([pt2[0]-pt1[0], pt2[1]-pt1[1]])

I4_points = []

anchor_points = anchor_sets["I2"]

for i in range(len(anchor_points)):
	I4_points.append([anchor_points[i][0] + diff_vec[i][0], anchor_points[i][1] + diff_vec[i][1]])


I2_points = np.array(corners(I2) + anchor_sets["I2"], dtype = np.int64)
I4_points = np.array(corners(I2) + I4_points, dtype = np.int64)


tri_I2 = Delaunay(I2_points)


I4 = np.zeros_like(I2, dtype=np.uint8)

for triangle in tri_I2.simplices:
    map_pts = triangle_map(I2_points[triangle], I4_points[triangle])
    I4 = transfer_color(I2.copy(), map_pts, I4)

cv2.imwrite(f_path + 'I4.jpg', I4)















