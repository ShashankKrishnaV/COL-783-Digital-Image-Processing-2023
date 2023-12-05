import numpy as np
import cv2
import math
from math import cos
from numpy import r_
from PIL import Image
import os

energy_map = None
gif_idx = 0

def cosp(i,j,n):
    output = 0
    output = cos(((2*i)+1)*j*math.pi/(2*n))
    return output

def convolveDCT(f,n,u,v,a,b):
    sumd = 0                               
    for x in r_[0:n]:
        for y in r_[0:n]:
            u = u%n
            v = v%n
            sumd += f[x+a,y+b]*cosp(x,u,n)*cosp(y,v,n)    
    if u == 0: sumd *= 1/math.sqrt(8) 
    else: sumd *= math.sqrt(2/8)
    if v == 0: sumd *= 1/math.sqrt(8)
    else: sumd *= math.sqrt(2/8)

    return sumd

def convolveIDCT(dctmatrix,n,x,y,a,b): 
    sumd = 0                           
    for u in r_[0:n]:
        for v in r_[0:n]:
            val1 = 1
            val2 = 1
            x = x%n
            y = y%n
            if u == 0: val1 = 1/math.sqrt(8)
            else: val1=math.sqrt(2/8)
            if v == 0: val2 = 1/math.sqrt(8)
            else: val2=math.sqrt(2/8)
            sumd += dctmatrix[u+a,v+b]*val1*val2*cosp(x,u,n)*cosp(y,v,n)   

    return sumd

def compute_energy_map(im):
    im = np.uint8(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    r_flag = False
    c_flag = False

    if im.shape[1]%8 !=0:
        extra_cols = (((im.shape[1] // 8)+1)*8) - im.shape[1]
        for i in range(extra_cols):  #Hard coded for 706->712
            im = np.hstack((im, np.expand_dims(im[:, -1], axis=1)))
        c_flag = True
    if im.shape[0]%8 != 0:
        extra_rows = (((im.shape[0] // 8)+1)*8) - im.shape[0]
        for i in range(extra_rows): #Hard coded for 532->536
            im = np.vstack((im, [im[-1]]))
        r_flag = True

    n = 8
    sumd = 0

    dctmatrix = np.zeros(np.shape(im))
    im = im.astype(np.int16)
    im = im-128
    im2 = np.zeros(np.shape(im))

    for a in r_[0:np.shape(im)[0]:n]:
        for b in r_[0:np.shape(im)[1]:n]:
            for u in r_[a:a+n]:
                for v in r_[b:b+n]:
                    dctmatrix[u,v] = convolveDCT(im,n,u,v,a,b)
    np.around(dctmatrix)

    Quant = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    factor = 4
    for a in r_[0:np.shape(im)[0]:n]:
        for b in r_[0:np.shape(im)[1]:n]:
            dctmatrix[a:a+n,b:b+n] = dctmatrix[a:a+n,b:b+n]/Quant*factor

    for a in r_[0:np.shape(dctmatrix)[0]:n]:
        for b in r_[0:np.shape(dctmatrix)[1]:n]:
            for x in r_[a:a+n]:
                for y in r_[b:b+n]:
                    im2[x,y] = convolveIDCT(dctmatrix,n,x,y,a,b)

    im2 = im2 + 128

    if r_flag:
        for i in range(extra_rows):
            height, width = im2.shape[:2]
            im2 = im2[:height-1, :]
    if c_flag:
        for i in range(extra_cols):
            im2 = im2[:, :-1]

    energy_map = np.absolute(cv2.Scharr(im2, -1, 1, 0)) + np.absolute(cv2.Scharr(im2, -1, 0, 1))

    return energy_map

def seams_carving(in_image, out_image, out_height, out_width):
    in_height, in_width = in_image.shape[: 2]

    delta_row, delta_col = int(out_height - in_height), int(out_width - in_width)

    if delta_col < 0:
        out_image = seams_removal(out_image, delta_col * -1, 'col')
    elif delta_col > 0:
        out_image = seams_insertion(out_image, delta_col, 'col')

    if delta_row < 0:
        out_image = rotate_image(out_image, 1)
        out_image = seams_removal(out_image, delta_row * -1, 'row')
        out_image = rotate_image(out_image, 0)
    elif delta_row > 0:
        out_image = rotate_image(out_image, 1)
        out_image = seams_insertion(out_image, delta_row, 'row')
        out_image = rotate_image(out_image, 0)

    return out_image

def seams_insertion(out_image, num_pixel, toadd):
    temp_image = np.copy(out_image)
    seams_record = []

    global energy_map
    global gif_idx

    if toadd == 'col':
        energy_map = calc_energy_map(out_image)
        cv2.imwrite('energy_map/energy_map.jpg', energy_map)

    else:
        if energy_map is None:
            out_dummy = rotate_image(out_image, 0)
            energy_map = calc_energy_map(out_dummy)
            cv2.imwrite('energy_map/energy_map.jpg', energy_map)
        energy_map = rotate_mask(energy_map, 1)

    energy_dummy = np.copy(energy_map)

    out_image_dummy = out_image.copy()

    for dummy in range(num_pixel):        
        cumulative_map = cumulative_map_backward(energy_dummy)
        seam_idx = find_seam(cumulative_map)
        seams_record.append(seam_idx)
        out_image_dummy, _ = delete_seam(out_image_dummy, seam_idx, toadd)

        energy_dummy = delete_seam_energy(seam_idx, energy_dummy)

    # out_image = np.copy(temp_image)
    n = len(seams_record)
    for dummy in range(n):
        seam = seams_record.pop(0)
        out_image, energy_map, gif_image = add_seam(out_image, seam, energy_map, toadd)
        seams_record = update_seams(seams_record, seam)

        cv2.imwrite(f'output_gif/gif_{gif_idx}.jpg', gif_image)
        gif_idx += 1

    energy_map = energy_map

    return out_image

def seams_removal(out_image, num_pixel, toremove):
    global energy_map
    global gif_idx

    if toremove == 'col':
        energy_map = calc_energy_map(out_image)
        cv2.imwrite('energy_map/energy_map.jpg', energy_map)

    else:
        if energy_map is None:
            out_dummy = rotate_image(out_image, 0)
            energy_map = calc_energy_map(out_dummy)
            cv2.imwrite('energy_map/energy_map.jpg', energy_map)
        energy_map = rotate_mask(energy_map, 1)
    

    for dummy in range(num_pixel):
        cumulative_map = cumulative_map_forward(out_image, energy_map)
        seam_idx = find_seam(cumulative_map)

        out_image, gif_image = delete_seam(out_image, seam_idx, toremove)
        
        cv2.imwrite(f'output_gif/gif_{gif_idx}.jpg', gif_image)
        gif_idx += 1

        energy_map = delete_seam_energy(seam_idx, energy_map)

    energy_map = energy_map

    return out_image


def calc_energy_map(out_image):
    return compute_energy_map(out_image)

def cumulative_map_backward(energy_map):
    m, n = energy_map.shape
    output = np.copy(energy_map)
    for row in range(1, m):
        for col in range(n):
            output[row, col] = \
                energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
    return output

def cumulative_map_forward(out_image, energy_map):
    kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
    kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
    kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

    matrix_x = calc_neighbor_matrix(out_image, kernel_x)
    matrix_y_left = calc_neighbor_matrix(out_image, kernel_y_left)
    matrix_y_right = calc_neighbor_matrix(out_image, kernel_y_right)

    m, n = energy_map.shape
    output = np.copy(energy_map)
    for row in range(1, m):
        for col in range(n):
            if col == 0:
                e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                e_up = output[row - 1, col] + matrix_x[row - 1, col]
                output[row, col] = energy_map[row, col] + min(e_right, e_up)
            elif col == n - 1:
                e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                e_up = output[row - 1, col] + matrix_x[row - 1, col]
                output[row, col] = energy_map[row, col] + min(e_left, e_up)
            else:
                e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                e_up = output[row - 1, col] + matrix_x[row - 1, col]
                output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
    return output

def calc_neighbor_matrix(out_image, kernel):
    b, g, r = cv2.split(out_image)
    output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
             np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
             np.absolute(cv2.filter2D(r, -1, kernel=kernel))
    return output

def find_seam(cumulative_map):
    m, n = cumulative_map.shape
    output = np.zeros((m,), dtype=np.uint32)
    output[-1] = np.argmin(cumulative_map[-1])

    for row in range(m - 2, -1, -1):
        prv_x = output[row + 1]
        if prv_x == 0:
            output[row] = np.argmin(cumulative_map[row, : 2])
        else:
            output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
    return output


def delete_seam(out_image, seam_idx, toremove):
    # print(type(out_image))
    gif = np.array(out_image, dtype=np.uint8).copy()

    m, n = out_image.shape[: 2]
    output = np.zeros((m, n - 1, 3))
    for row in range(m):
        col = seam_idx[row]
        output[row, :, 0] = np.delete(out_image[row, :, 0], [col])
        output[row, :, 1] = np.delete(out_image[row, :, 1], [col])
        output[row, :, 2] = np.delete(out_image[row, :, 2], [col])

        gif[row, col, 0] = np.uint8(0)
        gif[row, col, 1] = np.uint8(0)
        gif[row, col, 2] = np.uint8(255)

    out_image = np.copy(output)

    # print(type(out_image))

    if toremove != 'col':
        gif = rotate_image(gif, 0)

    # print('Delete seam: ', out_image.shape)


    # image_rgb = cv2.cvtColor(np.uint8(gif), cv2.COLOR_BGR2RGB)

    # gif_pil = Image.fromarray(image_rgb)

    return out_image, gif

def delete_seam_energy(seam_idx, energy_map):
    m, n = energy_map.shape
    output = np.zeros((m, n - 1))
    for row in range(m):
        col = seam_idx[row]
        output[row, :] = np.delete(energy_map[row, :], [col])

    energy_map = np.copy(output)
    return energy_map


def add_seam(out_image, seam_idx, energy_map, toadd):
    m, n = energy_map.shape
    new_energy_map = np.zeros((m, n + 1))

    m, n = out_image.shape[: 2]
    output = np.zeros((m, n + 1, 3))
    for row in range(m):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(out_image[row, col: col + 2, ch])
                output[row, col, ch] = out_image[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = out_image[row, col:, ch]
            else:
                p = np.average(out_image[row, col - 1: col + 1, ch])
                output[row, : col, ch] = out_image[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = out_image[row, col:, ch]

        if col == 0:
            p = np.average(energy_map[row, col: col + 2])
            new_energy_map[row, col] = energy_map[row, col]
            new_energy_map[row, col + 1] = p
            new_energy_map[row, col + 1:] = energy_map[row, col:]
        else:
            p = np.average(energy_map[row, col - 1: col + 1])
            new_energy_map[row, : col] = energy_map[row, : col]
            new_energy_map[row, col] = p
            new_energy_map[row, col + 1:] = energy_map[row, col:]

    gif = np.array(output, dtype=np.uint8).copy()
    for row in range(m):
        col = seam_idx[row]
        gif[row, col, 0] = np.uint8(0)
        gif[row, col, 1] = np.uint8(0)
        gif[row, col, 2] = np.uint8(255) 

    if toadd != 'col':
        gif = rotate_image(gif, 0)

    # image_rgb = cv2.cvtColor(np.uint8(gif), cv2.COLOR_BGR2RGB)

    # gif_pil = Image.fromarray(image_rgb)

    out_image = np.copy(output)

    return out_image, new_energy_map, gif


def update_seams(remaining_seams, current_seam):
    output = []
    for seam in remaining_seams:
        seam[np.where(seam >= current_seam)] += 2
        output.append(seam)
    return output


def rotate_image(image, ccw):
    m, n, ch = image.shape
    output = np.zeros((n, m, ch))
    if ccw:
        image_flip = np.fliplr(image)
        for c in range(ch):
            for row in range(m):
                output[:, row, c] = image_flip[row, :, c]
    else:
        for c in range(ch):
            for row in range(m):
                output[:, m - 1 - row, c] = image[row, :, c]
    return output

def rotate_mask(mask, ccw):
    m, n = mask.shape
    output = np.zeros((n, m))
    if ccw > 0:
        image_flip = np.fliplr(mask)
        for row in range(m):
            output[:, row] = image_flip[row, : ]
    else:
        for row in range(m):
            output[:, m - 1 - row] = mask[row, : ]
    return output

def save_result(filename, out_image):
    cv2.imwrite(filename, out_image.astype(np.uint8))


def seamcarving(image, new_h, new_w):
	image = image.astype(np.float64)
	old_h, old_w = image.shape[: 2]
	output_image = image.copy()

	output_image = seams_carving(image, output_image, new_h, new_w)

	return output_image


#main 

image_path = input('Provide input image path: ')

output_path = 'output/output_' + os.path.basename(image_path)

new_h = int(input('New height of the image: '))
new_w = int(input('New width of the image: '))

intial_image = cv2.imread(image_path)
final_image = seamcarving(intial_image, new_h, new_w)

save_result(output_path, final_image)











