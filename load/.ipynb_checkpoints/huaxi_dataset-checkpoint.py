import os
import sys
sys.path.append('/mnt/data/oss_beijing/qianyi/jiaorushi/GLS/code/Oncocardiology_huaxi/data_preprocess/')
import random
import numpy as np
import pandas as pd
import skimage
import SimpleITK as sitk
import glob

import torch
from numpy.random import randint
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as f
import cv2
from PIL import Image
from scipy.ndimage.measurements import label
from skimage.exposure import match_histograms
from lib_for_preprocess.contrastenhancement import *

from  albumentations  import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur,RandomContrast, RandomBrightness, RandomBrightnessContrast, PixelDropout, Flip, OneOf, Compose
)


# augumentations
def strong_aug():
    return Compose([
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
        GaussNoise(var_limit=(1.0, 2.0), p=0),
        MotionBlur(p=0),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=25, border_mode=0, p=0.5),
        RandomContrast(0.3,p=0),
        PixelDropout(dropout_prob=0.005, p=0)])
        #RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=1)])
    
def noise_aug():
    return Compose([
        GaussNoise(var_limit=(1.0, 2.0), p=0),
        MotionBlur(p=1),
        RandomContrast(0.3,p=0)])

imgaug = strong_aug()

def strong_aug_pretrain():
    return Compose([
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
        GaussNoise(var_limit=(0, 0.005), p=0.5),
        MotionBlur(p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=25, border_mode=0, p=0.5),
        RandomContrast(0.1,p=0.5),
        PixelDropout(dropout_prob=0.005, p=0)])
        #RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=1)])

imgaug_pretrain = strong_aug_pretrain()


def choose_clip(array, len_clip, interval=1):
    len_vdo = len(array)
    end_slice = len_vdo - (len_clip * interval)
    if end_slice <=0:
        # print('<0', interval, array.shape)
        return array
    else:
        start = np.random.randint(0, end_slice)
        # print(start, end_slice, interval, array.shape)
        # print(array[start:start+(len_clip * interval):interval].shape)
        return array[start:start+(len_clip * interval):interval]
    
def remove_noise(seg_volume, label_volume):
    label_volume, label_n = label(label_volume)
    if label_n > 1:
        max_size = 0
        max_label = 0
        for i in range(1, label_n + 1):
            if np.sum(label_volume == i) > max_size:
                max_size = np.sum(label_volume == i)
                max_label = i
        # print('max_size:', max_size, 'max_label:', max_label)
        for i in range(1, label_n + 1):
            if i != max_label:
                seg_volume[label_volume == i] = 0
    return seg_volume

def erode(mask):
    kernel = np.ones((7, 7), np.uint8)
    new_mask = np.zeros_like(mask)
    classes = list(range(np.amax(mask)+1))
    for num in classes:
        mask_copy = mask.copy()
        mask_copy = mask_copy == num
        mask_copy = np.array(mask_copy).astype('uint8')
        mask_copy = cv2.erode(mask_copy, kernel)
        new_mask[mask_copy != 0] = 1
    return new_mask
        
    

def simulated_noise(video, noise=0.05):
    n = video.shape[0] * video.shape[1] * video.shape[2]
    ind = np.random.choice(n, round(noise * n), replace=False)
    f = ind % video.shape[0]
    ind //= video.shape[0]
    i = ind % video.shape[1]
    ind //= video.shape[1]
    j = ind
    video[f, i, j] = 0
    return video

def setup_config(default_path="config.yaml"):
    # Setup Config files
    if os.path.exists(default_path):
        with open(default_path, "r") as f:
            config = yaml.load(f)
            return config
    try:
        if not os.path.exists(default_path):
            print("Path is not exists, check!")
            return None
    except Exception:
        print("Error: File not find")
        

            
# funcs for pretrain
def patch_rand_drop(x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    c, h, w, z = x.shape
    mask = np.zeros_like(x)
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if x_rep is None:
            x_uninitialized = np.random.normal(
                size=(c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s)).astype(x.dtype)
            x_uninitialized = (x_uninitialized - np.min(x_uninitialized)) / (
                np.max(x_uninitialized) - np.min(x_uninitialized)
            )
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
        mask[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = 1
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
    return x, mask


def rot_rand(args, x_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    device = torch.device(f"cuda:{args.local_rank}")
    x_rot = torch.zeros(img_n).long().to(device)
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot


def aug_rand(args, samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(args, x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(args, x_aug[i], x_aug[idx_rnd])
    return x_aug

def splitFrames(videoFileName):
    #time1 = time.time()
    cap = cv2.VideoCapture(videoFileName) 
    num = 1
    frame_list = []
    while True:
        success, data = cap.read()
        if not success:
            break
        #print('data', data.shape)
        im = Image.fromarray(data)
        #im.save('avi_save/' +str(num)+".jpg")
        #print(num)
        num = num + 1
        frame_list.append(data[...,0:1])
    cap.release()
    array = np.concatenate(frame_list, axis=-1)
    return array

def load_file(path):
    if path.endswith('.npy'):
        array = np.load(path)
    elif path.endswith('.npz'):
        array = np.load(path)['array']
    elif path.endswith('.avi'):
        array = splitFrames(path)
        array = array.transpose((2,0,1))
    elif path.endswith('.nii.gz'):
        ds = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(ds)
    # if len(array.shape) == 4:
    #     array = array[:,1,:,:]
    return array

def gen_uncertainty_repeat(times, img, vendor_dict):
    image = (img*255).astype('uint8')
    repeat = []
    gamma_list = np.linspace(0.5, 1.5, num=times)
    for gamma in gamma_list:
        img_noise = gammacorrection(image, gamma, vendor_dict)
        repeat.append(img_noise[np.newaxis,:,:])
    repeat = np.concatenate(repeat, axis=0)
    repeat = (repeat - np.amin(repeat))/(np.amax(repeat) - np.amin(repeat))
    return repeat

def gen_uncertainty_repeat_temporal(times, img, vendor_dict):
    image = (img*255).astype('uint8')
    repeat = []
    gamma_list = np.linspace(0.5, 1.5, num=times)
    for gamma in gamma_list:
        img_noise = gammacorrection(image, gamma, vendor_dict)
        repeat.append(img_noise[:,np.newaxis,:,:])
    repeat = np.concatenate(repeat, axis=1)
    repeat = (repeat - np.amin(repeat))/(np.amax(repeat) - np.amin(repeat))
    return repeat

def get_vendor_dict(codebook, root_folder):
    codebook = pd.read_csv(codebook)
    files = glob.glob(root_folder)
    vendor_dict = {}
    for file in files:
        name = file.split('/')[-1].split('.')[0]
        row = codebook[codebook['FileName']==name].iloc[0]
        vendor = row['UltrasoundSystem']
        if vendor not in vendor_dict.keys():
            vendor_dict[vendor] = []
        vendor_dict[vendor].append(file) 
    return vendor_dict

def hist_match(src, target):
    src = (((src-np.amin(src))/(np.amax(src)-np.amin(src)))*255).astype('uint8')
    target = (((target-np.amin(target))/(np.amax(target)-np.amin(target)))*255).astype('uint8')
    src_target = match_histograms(src, target, channel_axis=0)
    return src_target

def gammacorrection(src, gamma, vendor_dict):
    if np.random.randint(2):
        if np.random.randint(2):
            table = np.array([((i / 255.0) ** gamma) * 255
                              for i in range(0, 256)]).astype("uint8")
            src = cv2.LUT(src, table)
        elif vendor_dict is not None:
            vendor = random.choice(list(vendor_dict.keys()))
            file = random.choice(vendor_dict[vendor])
            target = np.load(file).transpose(1,0,2,3)[1:2]
            src = src[np.newaxis]
            src = hist_match(src, target)[0]
    return src
    
        



class Huaxi_Dataset(Dataset):
    def __init__(self, info_csv, split='train', size=256, slices_per_cycle=32, transform=imgaug, root_path=None):
        '''
        info_csv: the csv file for training, validation or testing
        '''
        self.info_csv = info_csv
        self.transform = transform
        self.size = size
        self.split = split
        self.root_path = root_path
        self.slices_per_cycle = slices_per_cycle
        self.chmb_dict = {'a2c':0, 'a3c':1, 'a4c':2}
        print('the length of ' + split, len(self.info_csv))

    def resize_ratio(self, array):
        shape = array.shape
        assert shape[1] < shape[2]
        array = array[:,:,-shape[1]:]
        return array

    def __getitem__(self, idx):
        row = self.info_csv.iloc[idx]
        chamber = row['chamber']
        cycle = row['png_path'].split('_')[-1]
        video_path = row['path'] + '_' + cycle[:-4] + '_' + chamber + '.npz'
        
        if self.root_path is None:
            paths = []
            path = random.choice(paths)
        else:
            path = self.root_path
        video_path = os.path.join(path, video_path)
        gls = row['gls']
        video = np.load(video_path)
        array, fps = video['array'], video['fps']
        if 'crop' not in self.root_path:
            array = self.resize_ratio(array)
        array = skimage.transform.resize(array, (self.slices_per_cycle, self.size, self.size), 1) 
        if self.transform and self.split == 'train':
            augs = self.transform(image=array.transpose((1,2,0)))
            array = augs['image'].transpose(2,0,1)
        array = (array - np.amin(array))/(np.amax(array) - np.amin(array))
        array = array[np.newaxis,:,:,:]
        array = np.repeat(array, 3, axis=0)
        return video_path, array, gls, self.chmb_dict[chamber]

    def __len__(self):
        return len(self.info_csv)
    
class Huaxi2d_Dataset(Dataset):
    def __init__(self, args, info_csv, split='train', size=224, transform=imgaug, root_path=None, seg_path='/mnt/workspace/jiaorushi/GLS/data/segment_npy_denoise/', uncertainty=False, HistMatch=False):
        '''
        info_csv: the csv file for training, validation or testing
        '''
        self.info_csv = info_csv
        self.transform = transform
        self.noise = noise_aug()
        self.size = size
        self.split = split
        self.root_path = root_path
        self.seg_path = seg_path
        #self.slices_per_cycle = slices_per_cycle
        self.chmb_dict = {'a2c':0, 'a3c':1, 'a4c':2,'mid':3, 'mv':3, 'apex':3}
        self.beita_lv = {'a2c':[1,0], 'a3c':[1,0], 'a4c':[1,0],'mid':[1,0], 'mv':[1,0], 'apex':[1,0]}
        self.beita_myo = {'a2c':[1,0], 'a3c':[1,0], 'a4c':[1,0],'mid':[1,1], 'mv':[1,1], 'apex':[1,1]}
        self.uncertainty=uncertainty
        self.args = args
        if HistMatch:
            self.vendor_dict = get_vendor_dict('/mnt/workspace/jiaorushi/GLS/data/RVENetDatabase/codebook.csv',
                                          '/mnt/workspace/jiaorushi/GLS/data/RVENetDatabase/preprocessed/train_*/*.npy')
        else:
            self.vendor_dict = None
        if self.args.prior:
            #self.pseudo_path = self.get_pseudo()
            self.pseudo_path = '/mnt/workspace/jiaorushi/GLS/data/Inference_seg_save/seg_CAMUS2d_train_a4c_a3c_a2c_mid_mv_5_0_False_False_False_240106/'
        #print('the length of ' + split, len(self.info_csv))

    def resize_ratio(self, array):
        shape = array.shape
        assert shape[1] < shape[2]
        array = array[:,:,-shape[1]:]
        return array

    def get_pseudo(self):
        files = glob.glob('/mnt/workspace/jiaorushi/GLS/data/Inference_seg_save/Pseudo_*/*/*.nii.gz')
        pseudo_dict = {}
        for file in files:
            pseudo_dict[file.split('/')[-1].split('.')[0]] = file
        return pseudo_dict

    def __getitem__(self, idx):
        row = self.info_csv.iloc[idx]
        chamber = row['chamber']
        cycle_number = row['cycle_number']
        cycle_video_path = row['path'] + '_' + str(cycle_number) + '_' + chamber + '.npz'
        #print(idx, cycle_video_path)
        seg_chamber = chamber
        # if chamber in ['apex', 'mid', 'mv']:
        #     seg_chamber = 'psax'
        if 'data' in row['path']:
            path = self.root_path
        else:
            path = '/mnt/workspace/jiaorushi/GLS/data/raw_cycles_zll/'
        file_path = os.path.join(path, cycle_video_path)
        gls = row['gls']
        video = np.load(file_path)
        array, fps = video['array'], video['fps']

        mask_path = self.seg_path + '/' + seg_chamber + '/' + '_'.join(row['path'].split('/')) + '_' + str(cycle_number) + '_' + seg_chamber + '.npy'
        if  os.path.exists(mask_path):
            cycle_segment = np.load(mask_path).astype(np.uint8)
        else:
            cycle_segment = None
        if self.args.prior:
            mask_path_pseudo = self.pseudo_path + '/' + seg_chamber + '/' + '_'.join(row['path'].split('/')) + '_' + str(cycle_number) + '_' + seg_chamber + '.nii.gz'            
            cycle_segment_pseudo = load_file(mask_path_pseudo).astype(np.uint8)
        else: 
            cycle_segment_pseudo = np.zeros_like(array)
        
        # if 'refined' or 'selection' or 'cwh' in self.seg_path:
        #     mask_path = self.seg_path + '/' + seg_chamber + '/' + '_'.join(row['path'].split('/')) + '_' + str(cycle_number) + '_' + seg_chamber + '.npy'
        #     if  os.path.exists(mask_path):
        #         cycle_segment = np.load(mask_path).astype(np.uint8)
        #     else:
        #         cycle_segment = np.zeros_like(array)
        # else:
        #     mask_path = self.seg_path + '/' + seg_chamber + '/' + '_'.join(row['path'].split('/')) + '_0.npy'
        #     if  os.path.exists(mask_path):
        #         segment = np.load(mask_path)
        #         cycle_segment = segment[int(row['start_slice']-1):int(row['end_slice']+1)].astype(np.uint8)
        #     else:
        #         cycle_segment = np.zeros_like(array)

        length = len(array)
        slice_num = np.random.randint(length)
        if row['select_slice'] != 'random':
            slice_num = int(row['select_slice'])
            frame_seg = cycle_segment[slice_num]
        else:
            frame_seg = cycle_segment_pseudo[slice_num]

        frame = array[slice_num]
        frame_seg = remove_noise(frame_seg, frame_seg>0)
        frame = skimage.transform.resize(frame, (self.size, self.size), 1) 
        frame_seg = skimage.transform.resize(frame_seg, (self.size, self.size), 0,anti_aliasing=False)
        if self.transform and self.split == 'train':
            augs = self.transform(image=frame, mask=frame_seg)
            frame, frame_seg = augs['image'], augs['mask']
        if self.uncertainty:
            repeat = gen_uncertainty_repeat(self.uncertainty, frame, self.vendor_dict)
        else:
            repeat = np.float64(-1.0)
        if self.split == 'train':
            gamma = np.random.rand(1) + 0.5
            frame_noise = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)
            frame_noise = (frame_noise - np.amin(frame_noise))/(np.amax(frame_noise) - np.amin(frame_noise))
            gamma = random.uniform(0.5, 1.5)
            frame = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)
        else:
            frame_noise = frame
        frame = (frame - np.amin(frame))/(np.amax(frame) - np.amin(frame))
        frame = frame[np.newaxis,:,:]
        frame_noise = frame_noise[np.newaxis,:,:]
        frame = np.repeat(frame, 3, axis=0)
        frame_noise = np.repeat(frame_noise, 3, axis=0)
        frame_seg = frame_seg[np.newaxis,:,:]
        return cycle_video_path, slice_num, frame, frame_noise, repeat, frame_seg, gls, self.chmb_dict[chamber], np.array(self.beita_lv[chamber]), np.array(self.beita_myo[chamber])
        #return slice_num

    def __len__(self):
        return len(self.info_csv)

class Huaxi2d_Dataset_temporal(Dataset):
    def __init__(self, args, info_csv, split='train', size=224, transform=imgaug, root_path=None, seg_path='/mnt/workspace/jiaorushi/GLS/data/segment_npy_denoise/', uncertainty=False, temporal_interval=3, HistMatch=False):
        '''
        info_csv: the csv file for training, validation or testing
        '''
        self.info_csv = info_csv
        self.transform = transform
        self.noise = noise_aug()
        self.size = size
        self.split = split
        self.root_path = root_path
        self.seg_path = seg_path
        #self.slices_per_cycle = slices_per_cycle
        self.chmb_dict = {'a2c':0, 'a3c':1, 'a4c':2,'mid':3, 'mv':3, 'apex':3}
        self.beita_lv = {'a2c':[1,0], 'a3c':[1,0], 'a4c':[1,0],'mid':[1,0], 'mv':[1,0], 'apex':[1,0]}
        self.beita_myo = {'a2c':[1,0], 'a3c':[1,0], 'a4c':[1,0],'mid':[1,1], 'mv':[1,1], 'apex':[1,1]}
        self.uncertainty=uncertainty
        self.temporal_interval = temporal_interval
        self.args = args
        if HistMatch:
            self.vendor_dict = get_vendor_dict('/mnt/workspace/jiaorushi/GLS/data/RVENetDatabase/codebook.csv',
                                          '/mnt/workspace/jiaorushi/GLS/data/RVENetDatabase/preprocessed/train_*/*.npy')
        else:
            self.vendor_dict = None
        if self.args.prior:
            self.pseudo_path = '/mnt/workspace/jiaorushi/GLS/data/flow_Inference_seg_save/cycle_TransUnet_cst_temporal_flow_woResidual_CAMUS2d_True_224_a4c_a2c_a3c_tpral_1_uncer_0_hist_False_cls_True_prior_True_1_pretrain_True_ws_7_240430'
            #self.pseudo_path = self.get_pseudo()
            #self.pseudo_path = '/mnt/workspace/jiaorushi/GLS/data/flow_Inference_seg_save/best_cycle_TransUnet_cst_temporal_flow_woResidual_huaxi2d_False_224_a4c_a2c_a3c_mid_mv_tpral_1_uncer_0_hist_False_cls_True_prior_True_1_pretrain_True_240408/'
            self.pseudo_path_mid_mv = '/mnt/workspace/jiaorushi/GLS/data/flow_Inference_seg_save/cycle_TransUnet_cst_temporal_flow_woResidual_huaxi2d_False_224_mid_mv_tpral_1_uncer_0_hist_False_cls_True_prior_True_1_pretrain_True_240413/'
        #print('the length of ' + split, len(self.info_csv))

    def resize_ratio(self, array):
        shape = array.shape
        assert shape[1] < shape[2]
        array = array[:,:,-shape[1]:]
        return array

    def get_pseudo(self):
        files = glob.glob('/mnt/workspace/jiaorushi/GLS/data/Inference_seg_save/Pseudo_*/*/*.nii.gz')
        pseudo_dict = {}
        for file in files:
            pseudo_dict[file.split('/')[-1].split('.')[0]] = file
        return pseudo_dict

    def __getitem__(self, idx):
        row = self.info_csv.iloc[idx]
        chamber = row['chamber']
        cycle_number = row['cycle_number']
        cycle_video_path = row['path'] + '_' + str(cycle_number) + '_' + chamber + '.npz'
        seg_chamber = chamber
        if 'data' in row['path']:
            path = self.root_path
        else:
            path = '/mnt/workspace/jiaorushi/GLS/data/raw_cycles_zll/'
        file_path = os.path.join(path, cycle_video_path)
        gls = row['gls']
        video = np.load(file_path)
        array, fps = video['array'], video['fps']
        
        mask_path = self.seg_path + '/' + seg_chamber + '/' + '_'.join(row['path'].split('/')) + '_' + str(cycle_number) + '_' + seg_chamber + '.npy'
        if  os.path.exists(mask_path):
            cycle_segment = np.load(mask_path).astype(np.uint8)
        else:
            cycle_segment = None
        if self.args.prior:
            #mask_path_pseudo = self.pseudo_path['_'.join(row['path'].split('/')) + '_' + str(cycle_number) + '_' + seg_chamber]
            if chamber in ['mid', 'mv']:
                pseudo_path = self.pseudo_path_mid_mv
            else:
                pseudo_path = self.pseudo_path
            mask_path_pseudo = pseudo_path + '/' + seg_chamber + '/' + '_'.join(row['path'].split('/')) + '_' + str(cycle_number) + '_' + seg_chamber + '.nii.gz'            
            cycle_segment_pseudo = load_file(mask_path_pseudo).astype(np.uint8)
        else: 
            cycle_segment_pseudo = np.zeros_like(array)
        
        if self.temporal_interval == 1:
            temporal = np.random.randint(5, 15)
        else: 
            temporal = self.temporal_interval
        length = len(array)
        if length <= temporal:
            temporal = 0

        if row['select_slice'] != 'random':
            slice_num = int(row['select_slice'])
            if slice_num-temporal < 0:
                if slice_num+temporal >= len(cycle_segment):
                    print('axiba', slice_num, temporal, len(cycle_segment))
                    next_frame = slice_num
                else:
                    next_frame = slice_num+temporal
            else:
                next_frame = slice_num-temporal
                
        else:
            slice_num = np.random.randint(temporal, length)
            next_frame = slice_num - temporal
        #print(next_frame, slice_num, self.temporal_interval, temporal)
        frame = array[[next_frame, slice_num]]
        if row['select_slice'] == 'random':
            frame_seg = cycle_segment_pseudo[[next_frame, slice_num]]
        else:
            frame_seg = np.array([cycle_segment_pseudo[next_frame],cycle_segment[slice_num]])
            #print(frame_seg.shape)
        frame_seg = remove_noise(frame_seg, frame_seg>0)
        frame = skimage.transform.resize(frame, (frame.shape[0], self.size, self.size), 1) 
        frame_seg = skimage.transform.resize(frame_seg, (frame_seg.shape[0], self.size, self.size), 0,anti_aliasing=False)
        if self.transform and self.split == 'train':
            frame = frame.transpose(1,2,0)
            frame_seg = frame_seg.transpose(1,2,0)
            augs = self.transform(image=frame, mask=frame_seg)
            frame, frame_seg = augs['image'], augs['mask']
            frame = frame.transpose(2,0,1)
            frame_seg = frame_seg.transpose(2,0,1)
        if self.uncertainty:
            repeat = gen_uncertainty_repeat_temporal(self.uncertainty, frame, self.vendor_dict)
        else:
            repeat = np.float64(-1.0)
        if self.split == 'train':
            gamma = np.random.rand(1) + 0.5
            frame_noise = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)
            frame_noise = (frame_noise - np.amin(frame_noise))/(np.amax(frame_noise) - np.amin(frame_noise))
            gamma = random.uniform(0.5, 1.5)
            frame = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)
        else:
            frame_noise = frame
        frame = (frame - np.amin(frame))/(np.amax(frame) - np.amin(frame))
        frame = frame[:,np.newaxis,:,:]
        frame_noise = frame_noise[:,np.newaxis,:,:]
        frame = np.repeat(frame, 3, axis=1)
        frame_noise = np.repeat(frame_noise, 3, axis=1)
        frame_seg = frame_seg[:,np.newaxis,:,:]
        #print(frame.shape, frame_noise.shape, repeat.shape, frame_seg.shape,)
        return cycle_video_path, slice_num, frame, frame_noise, repeat, frame_seg, gls, self.chmb_dict[chamber], np.array(self.beita_lv[chamber]), np.array(self.beita_myo[chamber])
        #return slice_num

    def __len__(self):
        return len(self.info_csv)
    
class Huaxi_Dataset_crop(Dataset):
    def __init__(self, info_csv, split='train', size=256, slices_per_cycle=32, transform=imgaug, root_path=None):
        '''
        info_csv: the csv file for training, validation or testing
        '''
        self.info_csv = info_csv
        self.transform = transform
        self.size = size
        self.split = split
        self.root_path = root_path
        self.slices_per_cycle = slices_per_cycle
        self.chmb_dict = {'a2c':0, 'a3c':1, 'a4c':2}
        print('the length of ' + split, len(self.info_csv))

    def resize_ratio(self, array):
        shape = array.shape
        assert shape[1] < shape[2]
        array = array[:,:,-shape[1]:]
        return array

    def __getitem__(self, idx):
        row = self.info_csv.iloc[idx]
        chamber = row['chamber']
        cycle = row['png_path'].split('_')[-1]
        video_path = row['path'] + '_' + cycle[:-4] + '_' + chamber + '.npz'
        
        if self.root_path is None:
            paths = []
            path = random.choice(paths)
        else:
            path = self.root_path
        video_path = os.path.join(path, video_path)
        gls = row['gls']
        video = np.load(video_path)
        array, fps = video['array'], video['fps']
        if 'crop' not in self.root_path:
            array = self.resize_ratio(array)
        array = skimage.transform.resize(array, (self.slices_per_cycle, self.size, self.size), 1) 
        if self.transform and self.split == 'train':
            augs = self.transform(image=array.transpose((1,2,0)))
            array = augs['image'].transpose(2,0,1)
        array = (array - np.amin(array))/(np.amax(array) - np.amin(array))
        array = array[np.newaxis,:,:,:]
        array = np.repeat(array, 3, axis=0)
        
        if gls <= 16:
            gls = 1
        else:
            gls = 0
        # elif gls <= 18:
        #     label = 1
        # else:
        #     label = 2
        return video_path, array, np.array([gls]), chamber

    def __len__(self):
        return len(self.info_csv)
    
class Open_Dataset(Dataset):
    def __init__(self, args, info_csv, split='train', size=256, slices_per_cycle=32, transform=imgaug_pretrain, root_path=None):
        '''
        info_csv: the csv file for training, validation or testing
        '''
        self.args = args
        self.info_csv = info_csv
        self.transform = transform
        self.size = size
        self.split = split
        self.root_path = root_path
        self.slices_per_cycle = slices_per_cycle
        self.chmb_dict = {'a2c':0, 'a3c':1, 'a4c':2}
        print('the length of ' + split, len(self.info_csv))
        

    def __getitem__(self, idx):
        row = self.info_csv.iloc[idx]
        file_path = row['path']
        #video_array = load_file(file_path)
        video_array = load_file(file_path)
        interval = np.random.randint(1,4)
        clip = choose_clip(video_array, self.slices_per_cycle, interval)
        clip = skimage.transform.resize(clip, (self.slices_per_cycle, self.size, self.size), 1) 
        # random frames
        order_label = np.random.randint(2)
        if order_label:
            order = random.sample(list(range(len(clip))),len(clip))
            clip = clip[order]
            # order_array = np.zeros((len(clip)))
            # change_order = np.random.randint(len(clip),size=2)
            # clip[change_order[0]], clip[change_order[1]] = clip[change_order[1]], clip[change_order[0]]
            # order_array[change_order[0]], order_array[change_order[1]] = 1, 1 
        if self.transform and self.split == 'train':
            augs = self.transform(image=clip.transpose((1,2,0)))
            clip = augs['image'].transpose(2,0,1)
        clip = ((clip - np.amin(clip))/(np.amax(clip) - np.amin(clip)))*255
        # print('clip', np.amax(clip))
        clip = clip[np.newaxis,:,:,:]
        #clip = np.repeat(clip, 3, axis=0)
        clip_aug, mask = patch_rand_drop(clip.copy(), max_drop=0.7)
        #print(np.sum(mask))
        # if 'Swin' in self.args.network:
        #     clip = clip.transpose((1,2,3,0))
        #     clip_aug = clip_aug.transpose((1,2,3,0))
        return file_path, clip, clip_aug, mask, np.array([order_label])

    def __len__(self):
        return len(self.info_csv)
    
    
class Open2dMAE_Dataset(Dataset):
    def __init__(self, args, info_csv, split='train', size=256, slices_per_cycle=32, transform=imgaug_pretrain, root_path=None):
        '''
        info_csv: the csv file for training, validation or testing
        '''
        self.args = args
        self.info_csv = info_csv
        self.transform = transform
        self.size = size
        self.split = split
        self.root_path = root_path
        self.slices_per_cycle = slices_per_cycle
        self.chmb_dict = {'a2c':0, 'a3c':1, 'a4c':2}
        print('the length of ' + split, len(self.info_csv))
        

    def __getitem__(self, idx):
        row = self.info_csv.iloc[idx]
        file_path = row['path']
        video_array = load_file(file_path)
        interval = np.random.randint(4,28)
        if len(video_array)-interval <= 0:
            interval = len(video_array) // 2
        start = np.random.randint(0,len(video_array)-interval)
        img = video_array[start:start+1]
        img_fut = video_array[start+interval:start+interval+1]
        imgs = np.concatenate([img, img_fut], axis=0)
        imgs = skimage.transform.resize(imgs, (2, self.size, self.size), 1) 
        
        if self.transform and self.split == 'train':
            augs = self.transform(image=imgs.transpose((1,2,0)))
            imgs = augs['image'].transpose(2,0,1)
        gamma = np.random.rand(1) + 0.5
        imgs = gammacorrection((imgs*255).astype('uint8'), gamma)
        imgs = ((imgs - np.amin(imgs))/(np.amax(imgs) - np.amin(imgs)))
        imgs = imgs[np.newaxis,:,:,:]
        imgs = np.repeat(imgs, 3, axis=0)
        # img = np.load('/mnt/workspace/jiaorushi/GLS/code/Oncocardiology_huaxi/DeepGLS/load/img_fut.npy')[0]
        # img = img[:, :, ::-1]
        # img = img.transpose((2,0,1))
        # img = skimage.transform.resize(img, (3, self.size, self.size), 1)
        # img = ((img - np.amin(img))/(np.amax(img) - np.amin(img)))
        return file_path, imgs[:,0], imgs[:,1]
        #return file_path, img, img

    def __len__(self):
        return len(self.info_csv)
    

class CAMUS2d_Dataset(Dataset):
    def __init__(self, args, info_csv, split='train', size=224, transform=imgaug, root_path=None, uncertainty=False, HistMatch=False):
        '''
        info_csv: the csv file for training, validation or testing
        '''
        self.info_csv = info_csv
        self.transform = transform
        self.noise = noise_aug()
        self.size = size
        self.split = split
        self.root_path = root_path
        self.chmb_dict = {'a2c':0, 'a3c':1, 'a4c':2,'mid':3, 'mv':3, 'apex':3}
        self.beita_lv = {'a2c':[1,0], 'a3c':[1,0], 'a4c':[1,0],'mid':[1,0], 'mv':[1,0], 'apex':[1,0]}
        self.beita_myo = {'a2c':[1,0], 'a3c':[1,0], 'a4c':[1,0],'mid':[1,1], 'mv':[1,1], 'apex':[1,1]}
        self.pseudo_path = '/mnt/workspace/jiaorushi/GLS/data/Inference_seg_save/seg_CAMUS2d_train_a4c_a3c_a2c_mid_mv_5_0_False_False_False_240106/'
        self.uncertainty=uncertainty
        #self.pseudo_path = 
        if HistMatch:
            self.vendor_dict = get_vendor_dict('/mnt/workspace/jiaorushi/GLS/data/RVENetDatabase/codebook.csv',
                                          '/mnt/workspace/jiaorushi/GLS/data/RVENetDatabase/preprocessed/train_*/*.npy')
        else:
            self.vendor_dict = None

    
    def __getitem__(self, idx):
        row = self.info_csv.iloc[idx]
        cycle_video_path = row['path']
        seg_path = cycle_video_path[:-7] + '_gt.nii.gz'
        gls = np.float64(-1.0)
        if '4CH' in cycle_video_path:
            seg_path_pseudo = self.pseudo_path + '/4CH/' + cycle_video_path.split('/')[-1][:-7] + '_gt.nii.gz'
            chamber = 'a4c'
        else:
            seg_path_pseudo = self.pseudo_path + '/2CH/' + cycle_video_path.split('/')[-1][:-7] + '_gt.nii.gz'
            chamber = 'a2c'
        array = load_file(cycle_video_path)
        cycle_segment = load_file(seg_path)
        #cycle_segment[cycle_segment==3] = 0
        length = len(array)
        if row['slice_num'] == 'random':
            ES = int(row['ED'])-1
            ED = int(row['ES'])-1
            choice = list(range(len(array)))
            choice.remove(ES)
            choice.remove(ED)
            slice_num = random.choice(choice)
            cycle_segment = load_file(seg_path_pseudo)
        else:
            slice_num = int(row['slice_num'])-1
        frame = array[slice_num]
        frame_seg = cycle_segment[slice_num]
        frame = (frame - np.amin(frame))/(np.amax(frame) - np.amin(frame))
        frame = skimage.transform.resize(frame, (self.size, self.size), 1) 
        frame_seg = skimage.transform.resize(frame_seg, (self.size, self.size), 0, anti_aliasing=False) 
        if self.uncertainty:
            repeat = gen_uncertainty_repeat(self.uncertainty, frame, self.vendor_dict)
        else:
            repeat = np.float64(-1.0)
        if self.transform and self.split == 'train':
            augs = self.transform(image=frame, mask=frame_seg)
            frame, frame_seg = augs['image'], augs['mask']
        if self.split == 'train':
            gamma = random.uniform(0.5, 1.5)
            frame_noise = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)
            gamma = random.uniform(0.5, 1.5)
            frame = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)
        else:
            frame_noise = frame
        frame_noise = (frame_noise - np.amin(frame_noise))/(np.amax(frame_noise) - np.amin(frame_noise))
        frame = (frame - np.amin(frame))/(np.amax(frame) - np.amin(frame))
        frame = frame[np.newaxis,:,:]
        frame_noise = frame_noise[np.newaxis,:,:]
        frame = np.repeat(frame, 3, axis=0)
        frame_noise = np.repeat(frame_noise, 3, axis=0)
        frame_seg = frame_seg[np.newaxis,:,:].astype('uint8')
        return cycle_video_path, slice_num, frame, frame_noise, repeat, frame_seg, gls, self.chmb_dict[chamber], np.array(self.beita_lv[chamber]), np.array(self.beita_myo[chamber])

    def __len__(self):
        return len(self.info_csv)
    
class CAMUS2d_Dataset_temporal(Dataset):
    def __init__(self, args, info_csv, split='train', size=224, transform=imgaug, root_path=None, uncertainty=False, temporal_interval=3, HistMatch=False):
        '''
        info_csv: the csv file for training, validation or testing
        '''
        self.info_csv = info_csv
        self.transform = transform
        self.noise = noise_aug()
        self.size = size
        self.split = split
        self.root_path = root_path
        self.chmb_dict = {'a2c':0, 'a3c':1, 'a4c':2,'mid':3, 'mv':3, 'apex':3}
        self.beita_lv = {'a2c':[1,0], 'a3c':[1,0], 'a4c':[1,0],'mid':[1,0], 'mv':[1,0], 'apex':[1,0]}
        self.beita_myo = {'a2c':[1,0], 'a3c':[1,0], 'a4c':[1,0],'mid':[1,1], 'mv':[1,1], 'apex':[1,1]}
        self.uncertainty=uncertainty
        self.temporal_interval = temporal_interval
        self.pseudo_path = '/mnt/workspace/jiaorushi/GLS/data/Inference_seg_save/Prior_TransUnet_cst_temporal_a4c_a3c_a2c_mid_mv_tpral_5_uncer_0_hist_False_cls_False_prior_False_240121/'
        if HistMatch:
            self.vendor_dict = get_vendor_dict('/mnt/workspace/jiaorushi/GLS/data/RVENetDatabase/codebook.csv',
                                          '/mnt/workspace/jiaorushi/GLS/data/RVENetDatabase/preprocessed/train_*/*.npy')
        else:
            self.vendor_dict = None

    def __getitem__(self, idx):
        #print('CAMUS')
        row = self.info_csv.iloc[idx]
        cycle_video_path = row['path']
        seg_path = cycle_video_path[:-7] + '_gt.nii.gz'
        gls = np.float64(-1.0)
        if '4CH' in cycle_video_path:
            seg_path_pseudo = self.pseudo_path + '/4CH/' + cycle_video_path.split('/')[-1][:-7] + '_gt.nii.gz'
            chamber = 'a4c'
        else:
            seg_path_pseudo = self.pseudo_path + '/2CH/' + cycle_video_path.split('/')[-1][:-7] + '_gt.nii.gz'
            chamber = 'a2c'
        array = load_file(cycle_video_path)
        cycle_segment = load_file(seg_path)
        
        #cycle_segment[cycle_segment==3] = 0
        if self.temporal_interval == 1:
            temporal = np.random.randint(5, 15)
        else: 
            temporal = self.temporal_interval
        length = len(array)
        if length <= temporal:
            temporal = 0
        cycle_segment_pseudo = load_file(seg_path_pseudo)
        if row['slice_num'] != 'random':
            slice_num = int(row['slice_num'])-1
            if slice_num-temporal < 0:
                if slice_num+temporal >= len(cycle_segment):
                    print('axiba', slice_num, temporal, len(cycle_segment))
                    next_frame = slice_num
                else:
                    next_frame = slice_num+temporal
            else:
                next_frame = slice_num-temporal
        else:
            choice = list(range(temporal, len(array)))
            slice_num = random.choice(choice)
            next_frame = slice_num - temporal
        #print(self.temporal_interval, temporal)
        #print(row['slice_num'], next_frame, slice_num, self.temporal_interval, temporal, len(array))
        frame = array[[next_frame, slice_num]]
        
        if row['slice_num'] == 'random':
            frame_seg = cycle_segment_pseudo[[next_frame, slice_num]]
        else:
            frame_seg = np.array([cycle_segment_pseudo[next_frame],cycle_segment[slice_num]])
        
        #print(frame.shape, frame_seg.shape)
        frame = (frame - np.amin(frame))/(np.amax(frame) - np.amin(frame))
        frame = skimage.transform.resize(frame, (frame.shape[0], self.size, self.size), 1) 
        frame_seg = skimage.transform.resize(frame_seg, (frame_seg.shape[0], self.size, self.size), 0, anti_aliasing=False) 
        if self.uncertainty:
            repeat = gen_uncertainty_repeat_temporal(self.uncertainty, frame, self.vendor_dict)
        else:
            repeat = np.float64(-1.0)
        if self.transform and self.split == 'train':
            frame = frame.transpose(1,2,0)
            frame_seg = frame_seg.transpose(1,2,0)
            augs = self.transform(image=frame, mask=frame_seg)
            frame, frame_seg = augs['image'], augs['mask']
            # new_shape = frame.shape
            # idx = np.where(np.array(new_shape)==2)[0][0]
            # idxes = list(range(3))
            # idxes.remove(idx)
            # idxes.insert(0, idx)
            # frame = frame.transpose(idxes)
            # frame_seg = frame_seg.transpose(idxes)
            frame = frame.transpose(2,0,1)
            frame_seg = frame_seg.transpose(2,0,1)
            #np.save('frame.npy', frame)
            
        if self.split == 'train':
            gamma = random.uniform(0.5, 1.5)
            frame_noise = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)
            gamma = random.uniform(0.5, 1.5)
            frame = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)
        else:
            frame_noise = frame
        frame_noise = (frame_noise - np.amin(frame_noise))/(np.amax(frame_noise) - np.amin(frame_noise))
        frame = (frame - np.amin(frame))/(np.amax(frame) - np.amin(frame))
        frame = frame[:,np.newaxis,:,:]
        frame_noise = frame_noise[:,np.newaxis,:,:]
        frame = np.repeat(frame, 3, axis=1)
        frame_noise = np.repeat(frame_noise, 3, axis=1)
        frame_seg = frame_seg[:,np.newaxis,:,:].astype('uint8')
        return cycle_video_path, slice_num, frame, frame_noise, repeat, frame_seg, gls, self.chmb_dict[chamber], np.array(self.beita_lv[chamber]), np.array(self.beita_myo[chamber])

    def __len__(self):
        return len(self.info_csv)
    

class update_pseudo(Dataset):
    def __init__(self, info_csv, size=224, segment_savepath=None, root_path=None):
        '''
        info_csv: the csv file for training, validation or testing
        '''
        self.info_csv = info_csv
        self.root_path = root_path
        self.size = size
        self.segment_savepath = segment_savepath
        self.chmb_dict = {'a2c':0, 'a3c':1, 'a4c':2, 'apex':3, 'mid':4, 'mv':5}

    def __getitem__(self, idx):
        row = self.info_csv.iloc[idx]
        chamber = row['chamber']
        cycle_number = row['cycle_number']
        cycle_video_path = row['path'] + '_' + str(cycle_number) + '_' + chamber + '.npz'
        seg_chamber = chamber
        if chamber in ['apex', 'mid', 'mv']:
            seg_chamber = 'psax'
        segment_save = self.segment_savepath + '/' + seg_chamber + '/' + '_'.join(row['path'].split('/')) + '_' + str(cycle_number) + '.npy'
        file_path = os.path.join(self.root_path, cycle_video_path)
        gls = row['gls']
        video = np.load(file_path)
        array, fps = video['array'], video['fps']
        array = skimage.transform.resize(array, (array.shape[0], self.size, self.size), 1) 
        array = (array - np.amin(array))/(np.amax(array) - np.amin(array))
        return cycle_video_path, array, gls, seg_chamber, segment_save 
        #return slice_num

    def __len__(self):
        return len(self.info_csv)
    
    







