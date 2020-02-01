import numpy as np
import glob
import os
import random
import re
import math
import multiprocessing
import imageio
from pathlib import Path


def load_pfm(filename):
     file = open(filename, 'r', newline='', encoding='latin-1')
     color = None
     width = None
     height = None
     scale = None
     endian = None

     header = file.readline().rstrip()
     if header == 'PF':
         color = True
     elif header == 'Pf':
         color = False
     else:
         raise Exception('Not a PFM file.')

     dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
     if dim_match:
         width, height = map(int, dim_match.groups())
     else:
         raise Exception('Malformed PFM header.')

     scale = float(file.readline().rstrip())
     if scale < 0: # little-endian
         endian = '<'
         scale = -scale
     else:
         endian = '>' # big-endian

     data = np.fromfile(file, endian + 'f')
     shape = (height, width, 3) if color else (height, width)

     file.close()

     return np.reshape(data, shape), scale


def bilinear_interp_val(vmap, y, x):
    '''
        bilinear interpolation on a 2D map
    '''
    h, w = vmap.shape
    x1 = int(x)
    x2 = x1 + 1
    x2 = w-1 if x2 > (w-1) else x2
    y1 = int(y)
    y2 = y1 + 1
    y2 = h-1 if y2 > (h-1) else y2
    Q11 = vmap[y1,x1]
    Q21 = vmap[y1,x2]
    Q12 = vmap[y2,x1]
    Q22 = vmap[y2,x2]
    return Q11 * (x2-x) * (y2-y) + Q21 * (x-x1) * (y2-y) + Q12 * (x2-x) * (y-y1) + Q22 * (x-x1) * (y-y1)


def get_3d_pos_xy(y_prime, x_prime, depth, focal_length=1050., w=960, h=540):
    '''
        depth pop up
    '''
    y = (y_prime - h / 2.) * depth / focal_length
    x = (x_prime - w / 2.) * depth / focal_length
    return [x, y, depth]


def gen_datapoint(fname_disparity, fname_disparity_next_frame, fname_disparity_change, fname_optical_flow, \
                  image, image_next_frame, n = 8192, max_cut = 35, focal_length=1050.):

    np.random.seed(601985)

    ##### generate needed data
    disparity_np, _ = load_pfm(fname_disparity)
    disparity_next_frame_np, _ = load_pfm(fname_disparity_next_frame)
    disparity_change_np, _ = load_pfm(fname_disparity_change)
    optical_flow_np, _ = load_pfm(fname_optical_flow)
    rgb_np = imageio.imread(image)[:, :, ::-1] / 255.
    rgb_next_frame_np = imageio.imread(image_next_frame)[:, :, ::-1] / 255.

    depth_np = focal_length / disparity_np
    depth_next_frame_np = focal_length / disparity_next_frame_np
    future_depth_np = focal_length / (disparity_np + disparity_change_np)
    
    h, w = disparity_np.shape

    ##### only take pixels that are near enough
    try:
        depth_requirement = depth_np < max_cut
    except:
        return None

    satisfy_pix1 = np.column_stack(np.where(depth_requirement))
    if satisfy_pix1.shape[0] < n:
        return None
    sample_choice1 = np.random.choice(satisfy_pix1.shape[0], size=n, replace=False)
    sampled_pix1_y = satisfy_pix1[sample_choice1, 0]
    sampled_pix1_x = satisfy_pix1[sample_choice1, 1]

    ##### flow generation
    current_pos1 = np.array([get_3d_pos_xy( sampled_pix1_y[i], sampled_pix1_x[i], \
                                           depth_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i])] ) for i in range(n)])
    current_rgb1 = np.array([[rgb_np[h-1-int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 0], \
                              rgb_np[h-1-int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 1], \
                              rgb_np[h-1-int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 2]] for i in range(n)])
    sampled_optical_flow_x = np.array([ optical_flow_np[ int( sampled_pix1_y[i] ), int( sampled_pix1_x[i] ) ][0] \
                                       for i in range(n)])
    sampled_optical_flow_y = np.array([ optical_flow_np[ int( sampled_pix1_y[i] ), int( sampled_pix1_x[i] ) ][1] \
                                       for i in range(n)])
    future_pix1_x = sampled_pix1_x + sampled_optical_flow_x
    future_pix1_y = sampled_pix1_y - sampled_optical_flow_y
    future_pos1 = np.array([get_3d_pos_xy( future_pix1_y[i], future_pix1_x[i], \
                                          future_depth_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i])] ) \
                            for i in range(n)])

    flow = future_pos1 - current_pos1
    
    ##### record original pixels positions
    sampled_pix1_yx = np.column_stack([sampled_pix1_y, sampled_pix1_x])

    ##### only take pixels that are near enough
    try:
        depth_requirement = depth_next_frame_np < max_cut
    except:
        return None

    satisfy_pix2 = np.column_stack(np.where(depth_next_frame_np < max_cut))
    if satisfy_pix2.shape[0] < n:
        return None
    sample_choice2 = np.random.choice(satisfy_pix2.shape[0], size=n, replace=False)
    sampled_pix2_y = satisfy_pix2[sample_choice2, 0]
    sampled_pix2_x = satisfy_pix2[sample_choice2, 1]

    ##### mask, judge whether point move out of fov or occluded by other object after motion
    current_pos2 = np.array([get_3d_pos_xy( sampled_pix2_y[i], sampled_pix2_x[i], \
                                           depth_next_frame_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i])] ) for i in range(n)])
    current_rgb2 = np.array([[rgb_next_frame_np[h-1-int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 0], \
                              rgb_next_frame_np[h-1-int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 1], \
                              rgb_next_frame_np[h-1-int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 2]] for i in range(n)])
    future_pos1_depth = future_depth_np[sampled_pix1_y, sampled_pix1_x]
    future_pos1_foreground_depth = np.zeros_like(future_pos1_depth)
    valid_mask_fov1 = np.ones_like(future_pos1_depth, dtype=bool)
    for i in range(future_pos1_depth.shape[0]):
        if future_pix1_y[i] > 0 and future_pix1_y[i] < h and future_pix1_x[i] > 0 and future_pix1_x[i] < w:
            future_pos1_foreground_depth[i] = bilinear_interp_val(depth_next_frame_np, future_pix1_y[i], future_pix1_x[i])
        else:
            valid_mask_fov1[i] = False
    valid_mask_occ1 = (future_pos1_foreground_depth - future_pos1_depth) > -5e-1

    mask1 = valid_mask_occ1 & valid_mask_fov1

    current_pos1[np.where(np.isnan(current_pos1)!=0)] = 0
    current_pos2[np.where(np.isnan(current_pos2)!=0)] = 0
    current_rgb1[np.where(np.isnan(current_rgb1)!=0)] = 0
    current_rgb2[np.where(np.isnan(current_rgb2)!=0)] = 0
    flow[np.where(np.isnan(flow)!=0)] = 0
    mask1[np.where(np.isnan(mask1)!=0)] = 0


    return current_pos1, current_pos2, current_rgb1, current_rgb2, flow, mask1


def proc_one_scene(s, input_dir_1, input_dir_2, output_dir):
    if s[-1] == '/':
        s = s[:-1]
    dis_split = s.split('/')
    train_or_test = dis_split[-4]
    ABC = dis_split[-3]
    scene_idx = dis_split[-2]
    left_right = dis_split[-1]
    for v in range(6, 15):
        fname = os.path.join(output_dir, train_or_test + '_' + ABC + '_' + scene_idx + '_' + left_right + '_' + str(v).zfill(4) + '-{}'.format(0) + '.npz')

        fname_disparity = os.path.join(input_dir_1, 'disparity', train_or_test, ABC, scene_idx, left_right, str(v).zfill(4) + '.pfm')
        fname_disparity_next_frame = os.path.join(input_dir_1, 'disparity', train_or_test, ABC, scene_idx, left_right, str(v+1).zfill(4) + '.pfm')
        fname_image = os.path.join(input_dir_1, 'frames_finalpass', train_or_test, ABC, scene_idx, left_right, str(v).zfill(4) + '.png')
        fname_image_next_frame = os.path.join(input_dir_1, 'frames_finalpass', train_or_test, ABC, scene_idx, left_right, str(v+1).zfill(4) + '.png')
        fname_disparity_change = os.path.join(input_dir_2, 'disparity_change', train_or_test, ABC, scene_idx, 'into_future', left_right, str(v).zfill(4) + '.pfm')
        L_R = 'L' if left_right == 'left' else 'R'
        fname_optical_flow = os.path.join(input_dir_2, 'optical_flow', train_or_test, ABC, scene_idx, 'into_future', left_right, 'OpticalFlowIntoFuture_' + str(v).zfill(4) + '_' + L_R + '.pfm')

        d = gen_datapoint(fname_disparity, fname_disparity_next_frame, fname_disparity_change, fname_optical_flow, fname_image, fname_image_next_frame, focal_length=1050.)
        if d is not None:
            print(fname)
            np.savez_compressed(fname, points1=d[0], \
                                       points2=d[1], \
                                       color1=d[2], \
                                       color2=d[3], \
                                       flow=d[4], \
                                       mask=d[5])
        else:
            print('N')


pool = multiprocessing.Pool(processes=8)

INPUT_DIR_1 = '/datasets'
INPUT_DIR_2 = '/data'
OUTPUT_DIR = '/home/tony/datasets/FT3D_point_cloud_data'

if not os.path.exists(OUTPUT_DIR):
    os.system('mkdir -p {}'.format(OUTPUT_DIR))

random.seed('chosenone')

disparity_scenes_train = sorted(list(glob.glob(os.path.join(INPUT_DIR_1, 'disparity/TRAIN/*/*/*/'))))
disparity_scenes_test = sorted(list(glob.glob(os.path.join(INPUT_DIR_1, 'disparity/TEST/*/*/*/'))))
disparity_scenes = random.sample(disparity_scenes_train, 2223) + random.sample(disparity_scenes_test, 223)


for s in disparity_scenes:
    proc_one_scene(s, INPUT_DIR_1, INPUT_DIR_2, OUTPUT_DIR)


pool.close()
pool.join()

