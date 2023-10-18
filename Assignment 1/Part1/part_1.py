import cv2
import numpy as np

image_path = input('Give folder name containing the source image: ')

path = '../' + str(image_path) + '/'
print('')

img_bgr = cv2.imread(path + 'source.jpg') #Image is being read in BGR color palette
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.imread(path + 'source.jpg', 0)


#1.1.1 Shadow-Map Generation

#RGB to HSI

img_np = np.asarray(img)

h, w, c = img_np.shape

I = np.zeros((h, w, 1))
V1 = np.zeros((h, w, 1))
V2 = np.zeros((h, w, 1))
S = np.zeros((h, w, 1))
H = np.zeros((h, w, 1))

hsi_mat = np.array([[1/3, 1/3, 1/3], [-(np.sqrt(6))/6, -(np.sqrt(6))/6, (np.sqrt(6))/3], [(1/np.sqrt(6)), -(2/np.sqrt(6)), 0]]) #Eqn 1

for y in range(h):
    for x in range(w):
        rgb = img_np[y][x]
        rgb = rgb.reshape((3, 1))
        out = np.matmul(hsi_mat, rgb)
        I[y][x] = out[0][0]
        V1[y][x] = out[1][0]
        V2[y][x] = out[2][0]
        S[y][x] = np.sqrt((np.square(V1[y][x])) + np.square(V2[y][x])) #Eqn 2
        if V1[y][x] != 1:
            if V1[y][x] == 0 and V2[y][x] == 0:
                H[y][x] = 0
                continue
            elif V1[y][x] == 0:
                H[y][x] = 0
                continue
            H[y][x] = np.arctan((V2[y][x])/V1[y][x]) #Eqn 3

H_norm = (H - np.min(H)) / (np.max(H) - np.min(H))
I_norm = (I - np.min(I)) / (np.max(I) - np.min(I))

#Computing r-map
r_map = (H_norm + 1) / (I_norm + 1) #Eqn 4

r_map_scaled = 255 * ((r_map - np.min(r_map)) / (np.max(r_map) - np.min(r_map)))


#Threshold for Shadow Map

P = {}

for y in range(r_map_scaled.shape[0]):
    for x in range(r_map_scaled.shape[1]):
        keys_list = P.keys()
        if int(r_map_scaled[y][x]) in keys_list:
            P[int(r_map_scaled[y][x])] += 1
        else:
            P[int(r_map_scaled[y][x])] = 1

for key in P.keys():
    P[key] = P[key] / (h * w)

T_dict = {}

for t in range(0, 256):
    W1, W2 = 0, 0
    for i in range(0, t+1):
        if i in P.keys():
            W1 += P[i]
    for i in range(t+1, 256):
        if i in P.keys():
            W2 += P[i]

    mu1, mu2 = 0, 0
    for i in range(0, t+1):
        if i in P.keys():
            mu1 += (i * P[i]) / W1
    for i in range(t+1, 256):
        if i in P.keys():
            mu2 += (i * P[i]) / W2

    t1, t2 = 0, 0
    for i in range(0, t+1):
        if i in P.keys():
            t1 += P[i] * ((i - mu1) ** 2)
    for i in range(t+1, 256):
        if i in P.keys():
            t2 += P[i] * ((i - mu2) ** 2)

    T_dict[t] = t1+t2


lt = []
for key in T_dict.keys():
    lt.append(T_dict[key])

T = lt.index(min(lt))


#Shadow map

s = np.where(r_map_scaled > T, 1, 0)

s_inv = np.where(r_map_scaled > T, 0, 1)
cv2.imwrite(path + 'shadow_map.jpg', s_inv * 255)
cv2.imwrite('shadow_map.jpg', s_inv * 255)

print('Shadow map is generated\n')


#Shadow Image

lmd_1 = 0.8
SI_1 = np.where(s_inv == 0, lmd_1 * img_bgr + (1 - lmd_1) * s_inv, img_bgr)

cv2.imwrite(path + f'shadow_image_lmb_{lmd_1}.jpg', SI_1)
cv2.imwrite(f'shadow_image_lmb_{lmd_1}.jpg', SI_1)

# lmd_2 = 1.3
# SI_2 = np.where(s_inv == 0, lmd_2 * img_bgr + (1 - lmd_2) * s_inv, img_bgr)

# cv2.imwrite(path + 'shadow_image_lmd_1.3.jpg', SI_2)

# print(f'Shadow Image generated with lambda {lmd_1} and {lmd_2} is generated\n')



#1.1.2 Line Draft Generation


#Bilateral Filtering

def bilateral_filter(image, diameter, sigma_color, sigma_space):
    filtered_image = np.zeros_like(image, dtype=np.float64)
    height, width = image.shape

    range_radius = diameter // 2

    for y in range(height):
        for x in range(width):
            weighted_sum = 0
            normalization = 0

            for i in range(-range_radius, range_radius + 1):
                for j in range(-range_radius, range_radius + 1):
                    neighbor_y = y + i
                    neighbor_x = x + j

                    if 0 <= neighbor_y < height and 0 <= neighbor_x < width:
                        color_difference = float(image[neighbor_y, neighbor_x]) - float(image[y, x])

                        intensity_weight = np.exp(-((color_difference)**2) / (2 * sigma_color * sigma_color))
                        spatial_weight = np.exp(-(np.square(i) + np.square(j)) / (2 * sigma_space * sigma_space))

                        weight = intensity_weight * spatial_weight

                        weighted_sum += weight * image[neighbor_y, neighbor_x]
                        normalization += weight

            filtered_image[y, x] = weighted_sum / normalization

    return filtered_image


diameter = 5
sigma_colors = [50]
sigma_spaces = [50]

bil_img = None
for sigma_color in sigma_colors:
    for sigma_space in sigma_spaces:
        filtered_image = bilateral_filter(img_gray, diameter, sigma_color, sigma_space)
        filtered_image = np.array(filtered_image, dtype=np.uint8)
        bil_img = filtered_image
        cv2.imwrite(path + f'bilateral_filter_output_{diameter}_{sigma_color}_{sigma_space}.jpg', filtered_image)
        cv2.imwrite(f'bilateral_filter_output_{diameter}_{sigma_color}_{sigma_space}.jpg', filtered_image)
        print('Bilateral Filter output is generated\n')

# bil_opencv = cv2.bilateralFilter(img_gray.copy(), 7, 50.0, 50.0)
# cv2.imwrite(path + 'bilateral_filter_output_opencv.jpg', bil_opencv)


#Edge Detection

#Best sigma_color, Best sigma_space 

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

padded_image = np.pad(bil_img, ((1, 1), (1, 1)), mode='edge')

gradient_x = np.zeros_like(bil_img, dtype=np.float32)
gradient_y = np.zeros_like(bil_img, dtype=np.float32)

for y in range(bil_img.shape[0]):
    for x in range(bil_img.shape[1]):
        gradient_x[y, x] = np.sum(padded_image[y:y+3, x:x+3] * sobel_x)
        gradient_y[y, x] = np.sum(padded_image[y:y+3, x:x+3] * sobel_y)

edge_map = np.sqrt(gradient_x**2 + gradient_y**2)

cv2.imwrite(path + 'edge_map.jpg', edge_map)
cv2.imwrite('edge_map.jpg', edge_map)
print('Edge map output is generated\n')


#Line Draft using Threshold

ld_thresh = 120

line_draft = np.where(edge_map >= ld_thresh, 1, 0)

line_draft_inv = 255 * np.where(edge_map >= ld_thresh, 0, 1)

cv2.imwrite(path + 'line_draft_inv.jpg', line_draft_inv)
cv2.imwrite('line_draft_inv.jpg', line_draft_inv)
print('Line draft image is generated\n')


#1.2 Color Adjustment Step

lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

l, a, b = lab_img[:,:,0], lab_img[:,:,1], lab_img[:,:,2]
l_mid = np.int8(60)

neutral_l = np.ones((h, w), dtype=np.uint8) * l_mid

chromatic_map = np.stack((neutral_l, a, b), axis = -1)

chromatic_map_rgb = cv2.cvtColor(chromatic_map, cv2.COLOR_LAB2RGB)

cv2.imwrite(path + 'chromatic_map.jpg', cv2.cvtColor(chromatic_map, cv2.COLOR_LAB2BGR))
cv2.imwrite('chromatic_map.jpg', cv2.cvtColor(chromatic_map, cv2.COLOR_LAB2BGR))
print('Chromatic map is generated\n')


#SI in BGR

SI = SI_1.copy()
rho = 0.01 #[0.005, 0.2]

SI = np.array(SI, dtype=np.uint8)

SI = cv2.cvtColor(SI, cv2.COLOR_BGR2RGB)

SI_prime = SI * ((1 + np.tanh(rho * (chromatic_map_rgb - 128))) / 2)
SI_prime = np.array(SI_prime, dtype=np.uint8)

cv2.imwrite(path + 'shadow_image_color_enhanced.jpg', cv2.cvtColor(SI_prime, cv2.COLOR_RGB2BGR))
cv2.imwrite('shadow_image_color_enhanced.jpg', cv2.cvtColor(SI_prime, cv2.COLOR_RGB2BGR))
print('Shadow image is generated\n')


#Saturation Correction

saturation_scale = 1.3

SI_prime_HSV = cv2.cvtColor(SI_prime, cv2.COLOR_RGB2HSV)

H = SI_prime_HSV[:, :, 0]
S = SI_prime_HSV[:, :, 1]
V = SI_prime_HSV[:, :, 2]

S_corrected = np.array(np.round(255 * ((S - np.min(S)) / (np.max(S) - np.min(S))))).astype(int)

S_corrected = S_corrected * saturation_scale
S_corrected = np.clip(S_corrected, 0, 255)

S_corrected = np.array(S_corrected, dtype=np.uint8)

SI_corrected = np.stack((H, S_corrected, V), axis = -1) #Why not S

SI_corrected_RGB = cv2.cvtColor(SI_corrected, cv2.COLOR_HSV2RGB)

cv2.imwrite(path + 'shadow_image_corrected.jpg', cv2.cvtColor(SI_corrected, cv2.COLOR_HSV2BGR))
cv2.imwrite('shadow_image_corrected.jpg', cv2.cvtColor(SI_corrected, cv2.COLOR_HSV2BGR))
print('Shadow image with corrected saturation is generated\n')


#Artistic Enhanced Image
beta = 0.7 #[0, 1]

line_draft_inv_3d = np.stack([line_draft_inv, line_draft_inv, line_draft_inv], axis = -1) 

artistic_image = np.where(line_draft_inv_3d == 0, beta * SI_corrected_RGB, SI_corrected_RGB)

artistic_image = np.array(artistic_image, dtype=np.uint8)

cv2.imwrite(path + 'artistic_image.jpg', cv2.cvtColor(artistic_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('artistic_image.jpg', cv2.cvtColor(artistic_image, cv2.COLOR_RGB2BGR))
print('Final artistic image is generated\n')























