import cv2
import numpy as np
import random

image_path = input('Give folder name containing the source image and target image: ')

path = '../' + str(image_path) + '/'
print('')

#3 Color Transfer

#Global transfer

def linshift_match_images(source_image, target_image):
    source_image_lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2LAB)
    source_image_l_norm = source_image_lab[:, :, 0]
    a, b = source_image_lab[:, :, 1], source_image_lab[:, :, 2]
    source_image_l_norm = np.array(source_image_l_norm, dtype=np.float64) / 255.0

    target_image_wo_norm = target_image.copy()

    target_image_norm = np.array(target_image_wo_norm, dtype=np.float64) / 255.0

    mu_target = np.mean(target_image_norm)
    sigma_target = np.std(target_image_norm)

    mu_source = np.mean(source_image_l_norm)
    sigma_source = np.std(source_image_l_norm)

    source_image_l_linshift_pre = ((sigma_target/sigma_source) * (source_image_l_norm - mu_source)) + mu_target

    # source_image_l_linshift = (source_image_l_linshift_pre - np.min(source_image_l_linshift_pre)) / (np.max(source_image_l_linshift_pre) - np.min(source_image_l_linshift_pre))

    source_image_l_linshift = source_image_l_linshift_pre.copy()

    source_image_l_linshift *= 255.0
    source_image_l_linshift = np.clip(source_image_l_linshift, 0, 255)

    source_image_linshift = np.dstack((source_image_l_linshift, a, b))

    return source_image_linshift

source_img = cv2.imread(path + 'artistic_image.jpg')
target_img = cv2.cvtColor(cv2.imread(path + 'target_image.jpg'), cv2.COLOR_BGR2LAB)[:, :, 0]

source_img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

source_image_linshift = linshift_match_images(source_img_rgb, target_img)

source_image_linshift_uint = np.array(source_image_linshift, dtype=np.uint8)

#cv2.imwrite('source_image_linshift.jpg', cv2.cvtColor(source_image_linshift_uint, cv2.COLOR_LAB2BGR))


def compute_sd(image, neighborhood_size=5):
    amt_to_pad = (neighborhood_size - 1) // 2
    y, x = image.shape
    sds = np.zeros((y, x), dtype=np.float64)

    padded = np.pad(image.copy(), ((amt_to_pad, amt_to_pad), (amt_to_pad, amt_to_pad)), mode='edge')
    padded = np.array(padded, dtype=np.float64)

    for i in range(amt_to_pad, y + amt_to_pad):
        for j in range(amt_to_pad, x + amt_to_pad):
            region = padded[i - amt_to_pad:i + amt_to_pad + 1, j - amt_to_pad:j + amt_to_pad + 1]
            sd = np.std(np.array(region, dtype=np.float64))
            sds[i - amt_to_pad, j - amt_to_pad] = sd
    return sds

source_img_l = (source_image_linshift.copy())[:, :, 0]
sds_source = compute_sd(source_img_l, 5)

target_img = cv2.cvtColor(cv2.imread(path + 'target_image.jpg'), cv2.COLOR_BGR2LAB)[:, :, 0]
sds_target = compute_sd(target_img, 5)


#jitter sampling

def jitter_sampling(image_lab, num_colors, sds):
    h, w, _ = image_lab.shape
    image_lab = np.array(image_lab, dtype=np.float64)

    l, a, b = image_lab[:, :, 0].copy(), image_lab[:, :, 1].copy(), image_lab[:, :, 2].copy()

    grid_size = int(np.ceil(np.sqrt(num_colors)))
    grid_height = h // grid_size
    grid_width = w // grid_size

    sampled_colors = []

    for i in range(grid_size):
        for j in range(grid_size):
            top = i * grid_height
            bottom = (i + 1) * grid_height
            left = j * grid_width
            right = (j + 1) * grid_width

            rand_y = random.randint(top, bottom - 1)
            rand_x = random.randint(left, right - 1)

            val = np.array([l[rand_y, rand_x], a[rand_y, rand_x], b[rand_y, rand_x], rand_y, rand_x, sds[rand_y, rand_x]], dtype=np.float64)
            sampled_colors.append(val)
    
    if len(sampled_colors) > num_colors:
        sampled_colors = random.sample(sampled_colors, num_colors)

    return sampled_colors

image_path = source_image_linshift.copy()
num_colors = 200 #As mentioned in paper

sampled_colors = jitter_sampling(image_path, num_colors, sds_source.copy())
sampled_colors = np.array(sampled_colors, dtype=np.float64)


def calc_distance_val(p1, p2, stds, stdt):
    diff = (0.5)*abs(p1 - p2) + (0.5)*abs(stds - stdt)
    return diff


def coloring(image, sds_source, sds_target):
    image = np.array(image, dtype=np.float64)
    colored_image_lab = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float64)

    for i in range(colored_image_lab.shape[0]):
        for j in range(colored_image_lab.shape[1]):
            sd = sds_target[i,j]

            colored_image_lab[i, j, 0] = image[i, j]
            min_lum = np.inf
            for color in sampled_colors:
                
                distance = calc_distance_val(color[0], image[i, j], sds_source[int(color[3]), int(color[4])], sd)

                if distance < min_lum:
                    min_lum = distance
                    colored_image_lab[i, j, 1], colored_image_lab[i, j, 2]  = color[1], color[2]
                else:
                    continue
    return colored_image_lab

target_image_to_color = cv2.cvtColor(cv2.imread(path + 'target_image.jpg'), cv2.COLOR_BGR2LAB)[:, :, 0]
colored_image_lab = coloring(target_image_to_color, sds_source, sds_target)

colored_image_lab_uint = np.array(colored_image_lab, dtype=np.uint8)

cv2.imwrite(path + 'colored_image_wo_swatch.jpg', cv2.cvtColor(colored_image_lab_uint, cv2.COLOR_LAB2BGR))
cv2.imwrite('colored_image_wo_swatch.jpg', cv2.cvtColor(colored_image_lab_uint, cv2.COLOR_LAB2BGR))
print('Target image with colors and without swatches is generated\n')


#With Swatches

### Step 1 ###

def jitter_sampling_swatch(img, image_lab, num_colors, y_add, x_add):
    h, w, _ = image_lab.shape
    l, a, b = img[:, :, 0].copy(), img[:, :, 1].copy(), img[:, :, 2].copy()

    grid_size = int(np.ceil(np.sqrt(num_colors)))
    grid_height = h // grid_size
    grid_width = w // grid_size

    sampled_colors = []
    locations = []

    for i in range(grid_size):
        for j in range(grid_size):
            top = i * grid_height
            bottom = (i + 1) * grid_height
            left = j * grid_width
            right = (j + 1) * grid_width

            rand_y = random.randint(top, bottom - 1)
            rand_x = random.randint(left, right - 1)

            val = np.array([l[rand_y + y_add, rand_x + x_add], a[rand_y + y_add, rand_x + x_add], b[rand_y + y_add, rand_x + x_add], rand_y + y_add, rand_x + x_add], dtype=np.float64)
            sampled_colors.append(val)
    
    if len(sampled_colors) > num_colors:
        sampled_colors = random.sample(sampled_colors, num_colors)

    return sampled_colors


#manual source swatch annotation
# source_swatches = [[59, 211, 346, 756], [406, 516, 622, 1000], [793, 388, 918, 569]] #
source_swatches = [[35, 20, 115, 105], [393, 586, 439, 636], [173, 533, 212, 606], [175, 585, 212, 611], [10, 188, 49, 307]]

def extract_colors(source_image, source_swatch, num_colors):
    y1, x1, y2, x2 = source_swatch
    swatch = source_image[y1:y2,x1:x2,:]
    colors = jitter_sampling_swatch(source_image, swatch, num_colors, y1, x1)
    return colors
    

source_image = source_image_linshift.copy()
source_image_lab = source_image_linshift.copy()
target_image = cv2.cvtColor(cv2.imread(path + 'target_image.jpg'), cv2.COLOR_BGR2LAB)[:, :, 0]

list_colors_per_swatch = []
for swatch in source_swatches:
    list_colors_per_swatch.append(extract_colors(source_image_lab, swatch, 50))



def compute_ssd_patch(source_patch, target_patch):
    return np.sum((source_patch - target_patch) ** 2)

def color_transfer(source_image, target_image, target_swatch, target_color_image, colors, sds_source, sds_target, window_size=5):
    padding = window_size//2

    # print(target_image.shape)
    # print(source_image.shape)

    padded_source = np.pad(source_image, ((padding, padding), (padding, padding)), mode='edge')
    padded_source = np.array(padded_source, dtype=np.float64)
    # print(padded_source.shape)

    padded_target = np.pad(target_image, ((padding, padding), (padding, padding)), mode='edge')
    padded_target = np.array(padded_target, dtype=np.float64)
    # print(padded_target.shape)

    target_image = np.array(target_image, dtype=np.float64)

    y1, x1, y2, x2 = target_swatch
    y1 += padding
    x1 += padding
    y2 += padding
    x2 += padding

    source_patches = []
    std_source = []

    for color in colors:
        s_y, s_x = int(color[3]), int(color[4])
        s_y += padding
        s_x += padding
        source_patch = padded_source[s_y - padding:s_y + padding+1, s_x - padding:s_x + padding+1]
        source_patches.append(source_patch)
        std_source.append(np.std(source_patch))
    source_patches = np.array(source_patches)

    for y in range(y1, y2):
        for x in range(x1, x2):
            target_patch = padded_target[y - padding:y + padding+1, x - padding:x + padding+1]


            if(target_patch.shape[1] == 4 or source_patch.shape[1] == 4):
                print(y, x)

            min_ssd = np.inf

            std_target = np.std(target_patch)

            best_match_color = None

            for color in range(len(colors)):
                source_patch = source_patches[color]
                color_diff = abs(target_patch[2, 2] - source_patch[2, 2])

                # ssd = compute_ssd_patch(source_patch, target_patch)

                ssd = (0.5)*color_diff + (0.5)*(abs(std_source[color] - std_target))

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_match_color = colors[color]
            
            target_color_image[y-padding, x-padding] = np.array([target_image[y-padding, x-padding], best_match_color[1].copy(), best_match_color[2].copy()])
    return target_color_image


h, w = target_image.shape
l_ch = target_image.copy()
a_ch, b_ch = np.zeros((h,w)), np.zeros((h,w))
target_color_image = np.dstack((l_ch, a_ch, b_ch))


#manual target swatch annotation
# target_swatches = [[14, 67, 87, 402], [181, 163, 265, 395], [303, 312, 330, 468]] #
target_swatches = [[150, 110, 210, 200], [355, 70, 405, 195], [263, 74, 305, 150], [314, 150, 323, 265], [25, 300, 115, 375]] 

# source_img = cv2.imread(path + 'source_image.jpg')
source_img = source_image_linshift.copy()
source_img = np.array(source_img, dtype=np.uint8)
source_image_rgb = cv2.cvtColor(source_img, cv2.COLOR_LAB2RGB)
source_image_lab = linshift_match_images(source_image_rgb, l_ch)

source_image_lab_uint = np.array(source_image_lab, dtype=np.uint8)

sds_source = compute_sd(source_image_lab[:, :, 0].copy())
sds_target = compute_sd(l_ch.copy())


for color, target_swatch in zip(list_colors_per_swatch, target_swatches):
    target_color_image = color_transfer(source_image_lab[:, :, 0].copy(), l_ch, target_swatch, target_color_image, color, sds_source, sds_target)
    
target_color_image_save = np.array(target_color_image, dtype=np.uint8)

cv2.imwrite(path + 'target_image_with_swatch_from_source.jpg', cv2.cvtColor(target_color_image_save, cv2.COLOR_LAB2BGR))
cv2.imwrite('target_image_with_swatch_from_source.jpg', cv2.cvtColor(target_color_image_save, cv2.COLOR_LAB2BGR))
print('Target image with colors and swatches from source is generated\n')


### Step 2 ###

#in step 2, source swatches will be the swatches that are target in step 1
# src_swatches = [[14, 67, 87, 402], [181, 163, 265, 395], [303, 312, 330, 468]] #
src_swatches = [[150, 110, 210, 200], [355, 70, 405, 195], [263, 74, 305, 150], [314, 150, 323, 265], [25, 300, 115, 375]] 
src_colors = []

for swatch in src_swatches:
    color = extract_colors(target_color_image, swatch, 50)
    src_colors.extend(color)

target_swatch = [0, 0, target_color_image[:, :, 0].shape[0], target_color_image[:, :, 0].shape[1]]
target_color_image = color_transfer(target_color_image[:, :, 0], target_color_image[:, :, 0], target_swatch, target_color_image, src_colors, sds_source, sds_target)

target_color_image_save = np.array(target_color_image, dtype=np.uint8)

cv2.imwrite(path + 'target_image_with_swatch_from_target.jpg', cv2.cvtColor(target_color_image_save, cv2.COLOR_LAB2BGR))
cv2.imwrite('target_image_with_swatch_from_target.jpg', cv2.cvtColor(target_color_image_save, cv2.COLOR_LAB2BGR))
print('Target image with colors and with swatches is generated\n')









