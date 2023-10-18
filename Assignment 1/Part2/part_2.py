import cv2
import numpy as np

image_path = input('Give folder name containing the source image: ')

path = '../' + str(image_path) + '/'
print('')

#2 Quantization

#Median Cut

img_bgr = cv2.imread(path + 'source.jpg') #Image is being read in BGR color palette
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.imread(path + 'source.jpg', 0)

artistic_image = cv2.imread(path + 'artistic_image.jpg')
artistic_image = cv2.cvtColor(artistic_image, cv2.COLOR_BGR2RGB)

median_cut_image = artistic_image.copy()

def median_cut_quantize(img, flattened_arr):
    r_avg = np.mean(flattened_arr[:,0])
    g_avg = np.mean(flattened_arr[:,1])
    b_avg = np.mean(flattened_arr[:,2])
    
    for data in flattened_arr:
        img[data[3], data[4]] = [r_avg, g_avg, b_avg]

def split_into_bins(img, flattened_arr, depth):  
    if depth == 0:
        median_cut_quantize(img, flattened_arr)
        return  
    ranges = np.ptp(flattened_arr, axis=0)
    r_range, g_range, b_range = ranges[0], ranges[1], ranges[2]

    rgb_val = [r_range, g_range, b_range]
    range_idx = rgb_val.index(max(rgb_val))

    flattened_arr = flattened_arr[flattened_arr[:,range_idx].argsort()]
    median_index = int((len(flattened_arr)+1)/2)

    split_into_bins(img, flattened_arr[0:median_index], depth-1)
    split_into_bins(img, flattened_arr[median_index:], depth-1)


flattened_img_array = []
for rindex, rows in enumerate(median_cut_image):
    for cindex, color in enumerate(rows):
        flattened_img_array.append([color[0],color[1],color[2],rindex, cindex]) 
        
flattened_img_array = np.array(flattened_img_array)

colors = 5 #2^n colors #5 indicates 32 colors
        
split_into_bins(median_cut_image, flattened_img_array, colors) 

cv2.imwrite(path + f'median_cut_quantized_image_{2**colors}.jpg', cv2.cvtColor(median_cut_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(f'median_cut_quantized_image_{2**colors}.jpg', cv2.cvtColor(median_cut_image, cv2.COLOR_RGB2BGR))
print(f'Median cut image is genearted with {2**colors} colors')


#Floyd Steinberg Dithering

quantized_image = median_cut_image.copy()

original_image = artistic_image.copy()

flattened_array = median_cut_image.reshape(-1, 3)
unique_pixel_values = np.array(np.unique(flattened_array, axis=0), dtype=np.float64)

def calc_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

def nearestpixel(pix):
    min_distance = float('inf')
    closest_pixel = None
    
    for unique_pixel in unique_pixel_values:
        distance = calc_distance(pix, unique_pixel)
        if distance < min_distance:
            min_distance = distance
            closest_pixel = unique_pixel
    return closest_pixel

def floyd_steinberg_dithering(org_image, quant_image):
    new_image = np.array(org_image.copy(), dtype=np.float64)
    height, width, _ = new_image.shape

    for y in range(height):
        for x in range(width):
            pix = new_image[y, x].copy()
            new_pixel = nearestpixel(pix.copy())
            new_image[y, x] = new_pixel
            
            quantization_error = pix - new_pixel

            if x + 1 < width:
                new_image[y, x + 1] = new_image[y, x + 1] + quantization_error * 7 / 16
            if y + 1 < height:
                if x - 1 >= 0:
                    new_image[y + 1, x - 1] = new_image[y + 1, x - 1] + quantization_error * 3 / 16
                quant_image[y + 1, x] = quant_image[y + 1, x] + quantization_error * 5 / 16
                if x + 1 < width:
                    new_image[y + 1, x + 1] = new_image[y + 1, x + 1] + quantization_error * 1 / 16
    
    dithered_image = np.array(new_image, dtype=np.uint8)
    return dithered_image

dithered_image = floyd_steinberg_dithering(original_image, quantized_image)

cv2.imwrite(path + 'dithered_image.jpg', cv2.cvtColor(dithered_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('dithered_image.jpg', cv2.cvtColor(dithered_image, cv2.COLOR_RGB2BGR))
print('Dithered image with same quantization level as median cut image\n')



#Qunatization error

src_img = artistic_image.copy() #In RGB
median_cut_img = median_cut_image.copy() #In RGB
dithered_img = dithered_image.copy() # In RGB

error_median_cut = np.mean(np.abs(median_cut_img - src_img))
error_dithering = np.mean(np.abs(dithered_img - src_img))

print('Mean error of Median cut image with Artistic image is ', error_median_cut)
print('\nMean error of Dithered image with Artistic image is ', error_dithering)




