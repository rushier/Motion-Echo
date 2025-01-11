import os
import sys
sys.path.append('/mnt/workspace/jiaorushi/GLS/code/Oncocardiology_huaxi/data_preprocess/')
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

def loadvideo_gray(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v[count, :, :] = frame

    #v = v.transpose((3, 0, 1, 2))

    return v

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


class Huaxi2d_Dataset_flow(Dataset):
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

        self.pseudo_path = '/mnt/workspace/jiaorushi/GLS/data/Inference_seg_save/Prior_TransUnet_cst_temporal_a4c_a3c_a2c_mid_mv_tpral_5_uncer_0_hist_False_cls_False_prior_False_240121/'
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
        mask_path_pseudo = self.pseudo_path + '/' + seg_chamber + '/' + '_'.join(row['path'].split('/')) + '_' + str(cycle_number) + '_' + seg_chamber + '.nii.gz'      
        gls = row['gls']
        video = np.load(file_path)
        array, fps = video['array'], video['fps']
        length = list(range(len(array)))
        slice_nums = random.choices(length, k=2)
        slice_num_1 = slice_nums[0]
        slice_num_2 = slice_nums[1]
        
        # length_1 = length[:len(array)//2]
        # print(len(array), length_1)
        # slice_num_1 = random.choice(length_1)
        # length_2 = length[len(array)//2:]
        # slice_num_2 = random.choice(length_2)
        cycle_segment_pseudo = load_file(mask_path_pseudo).astype(np.uint8)
        frame = array[[slice_num_1, slice_num_2]]
        mask = cycle_segment_pseudo[[slice_num_1, slice_num_2]]
        frame = skimage.transform.resize(frame, (frame.shape[0], self.size, self.size), 1) 
        mask = skimage.transform.resize(mask, (mask.shape[0], self.size, self.size), 0, anti_aliasing=False) 
        if self.transform and self.split == 'train':
            frame = frame.transpose(1,2,0)
            mask = mask.transpose(1,2,0)
            augs = self.transform(image=frame, mask=mask)
            frame, mask = augs['image'], augs['mask']
            frame, mask = frame.transpose(2,0,1), mask.transpose(2,0,1)
        if self.split == 'train':
            gamma = random.uniform(0.5, 1.5)
            frame = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)
        frame = (frame - np.amin(frame))/(np.amax(frame) - np.amin(frame))
        frame = frame[:,np.newaxis,:,:]
        frame = np.repeat(frame, 3, axis=1)
        mask = mask[:,np.newaxis,:,:]
        return cycle_video_path, slice_nums, frame, frame, frame, mask, 0, self.chmb_dict[chamber], 0, 0
        #return slice_num

    def __len__(self):
        return len(self.info_csv)

    
class CAMUS2d_Dataset_flow(Dataset):
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
        cycle_segment_pseudo = load_file(seg_path_pseudo).astype(np.uint8)
        
        length = list(range(len(array)))
        slice_nums = random.choices(length, k=2)
        slice_num_1 = slice_nums[0]
        slice_num_2 = slice_nums[1]
    
        frame = array[[slice_num_1, slice_num_2]]
        mask = cycle_segment_pseudo[[slice_num_1, slice_num_2]]
        frame = skimage.transform.resize(frame, (frame.shape[0], self.size, self.size), 1) 
        mask = skimage.transform.resize(mask, (mask.shape[0], self.size, self.size), 0, anti_aliasing=False) 
        if self.transform and self.split == 'train':
            frame = frame.transpose(1,2,0)
            mask = mask.transpose(1,2,0)
            augs = self.transform(image=frame, mask=mask)
            frame, mask = augs['image'], augs['mask']
            frame, mask = frame.transpose(2,0,1), mask.transpose(2,0,1)
        # if self.split == 'train':
        #     gamma = random.uniform(0.5, 1.5)
        #     frame = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)
        frame = (frame - np.amin(frame))/(np.amax(frame) - np.amin(frame))
        frame = frame[:,np.newaxis,:,:]
        frame = np.repeat(frame, 3, axis=1)
        mask = mask[:,np.newaxis,:,:]
        return cycle_video_path, slice_nums, frame, frame, frame, mask, 0, self.chmb_dict[chamber], 0, 0

    def __len__(self):
        return len(self.info_csv)
    
    
    
    
class Echo2d_Dataset_flow(Dataset):
    def __init__(self, args, info_csv, split='train', size=224, transform=imgaug, root_path=None, seg_path='/mnt/workspace/jiaorushi/GLS/data/segment_npy_denoise/', uncertainty=False, temporal_interval=3, HistMatch=False):
        '''
        info_csv: the csv file for training, validation or testing
        '''
        self.info_csv = info_csv
        self.tracing_csv = pd.read_csv('/mnt/workspace/jiaorushi/GLS/data/EchoNet-Dynamic/VolumeTracings.csv')
        self.tracing_dict = self.gen_tracing(self.tracing_csv)
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
        self.pseudo_path = '/mnt/workspace/jiaorushi/GLS/data/Inference_seg_save/Prior_TransUnet_cst_temporal_a4c_a3c_a2c_mid_mv_tpral_5_uncer_0_hist_False_cls_False_prior_False_240121/'
            
    def gen_tracing(self, csv_file):
        file_names = csv_file['FileName'].values.tolist()
        file_frame_dict = {}
        for i in range(len(csv_file)):
            row = csv_file.iloc[i]
            name = row['FileName']
            frame = row['Frame']
            if name not in file_frame_dict.keys():
                file_frame_dict[name] = []
            if frame not in file_frame_dict[name]:
                file_frame_dict[name].append(frame)
        return file_frame_dict
    
    def gen_mask(self, PID, video):
        frames = self.tracing_dict[PID+'.avi']
        masks_dict = {}
        for f in frames:
            file_csv = self.tracing_csv[self.tracing_csv['FileName']==PID+'.avi']
            file_csv = file_csv[file_csv['Frame'] == f]

            x1, y1, x2, y2 = file_csv['X1'].values.tolist(), file_csv['Y1'].values.tolist(), file_csv['X2'].values.tolist(), file_csv['Y2'].values.tolist()
            x = np.concatenate((x1[1:], np.flip(x2[1:])))
            y = np.concatenate((y1[1:], np.flip(y2[1:])))

            r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[1], video.shape[2]))
            mask = np.zeros((video.shape[1], video.shape[2]), np.float32)
            mask[r, c] = 1
            masks_dict[f] = mask
        return masks_dict

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
        stoped = False
        if 1:
        #while not stoped:
            if 1:
                row = self.info_csv.iloc[idx]
                lvef = row['EF']
                PID = row['FileName']
                file_path = self.root_path + '/' + PID + '.avi'
                array = loadvideo_gray(file_path)
                masks_dict = self.gen_mask(PID, array)
                stoped = True
                chamber = 'a4c'
            # except:
            #     idx = random.choice(list(range(len(self.info_csv))))
            #     print(PID)
        cycle_segment = np.zeros_like(array)
        array = load_file(file_path).squeeze()
        #cycle_segment = load_file(seg_path_pseudo)
        
        length = list(range(len(array)))
        slice_nums = random.choices(length, k=2)
        slice_num_1 = slice_nums[0]
        slice_num_2 = slice_nums[1]
        
        frame = array[[slice_num_1, slice_num_2]]
        mask = cycle_segment[[slice_num_1, slice_num_2]]
        
        frame = (frame - np.amin(frame))/(np.amax(frame) - np.amin(frame))
        frame = skimage.transform.resize(frame, (frame.shape[0], self.size, self.size), 1) 
        mask = skimage.transform.resize(mask, (mask.shape[0], self.size, self.size), 0, anti_aliasing=False) 
        if self.transform and self.split == 'train':
            frame = frame.transpose(1,2,0)
            mask = mask.transpose(1,2,0)
            augs = self.transform(image=frame, mask=mask)
            frame, mask = augs['image'], augs['mask']
            frame, mask = frame.transpose(2,0,1), mask.transpose(2,0,1)
            
        if self.split == 'train':
            gamma = random.uniform(0.5, 1.5)
            frame = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)
        
        frame = (frame - np.amin(frame))/(np.amax(frame) - np.amin(frame))
        frame = frame[:,np.newaxis,:,:]
        frame = np.repeat(frame, 3, axis=1)
        mask = mask[:,np.newaxis,:,:].astype('uint8')
        return file_path, slice_nums, frame, frame, frame, mask, 0, self.chmb_dict[chamber], 0, 0
        #return slice_num
    def __len__(self):
        return len(self.info_csv)

    
    







