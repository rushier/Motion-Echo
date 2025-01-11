import os
import random
import glob
import cv2
import numpy as np
import pandas as pd
import skimage
import SimpleITK as sitk
from numpy.random import randint
from PIL import Image
from scipy.ndimage.measurements import label
from skimage.exposure import match_histograms

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as f

from  albumentations  import (ShiftScaleRotate, RandomRotate90,
    Transpose, ShiftScaleRotate,
    GaussNoise, MotionBlur,PixelDropout, Flip, Compose
)

from .contrastenhancement import *

def strong_aug():
    return Compose([
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
        GaussNoise(var_limit=(1.0, 2.0), p=0),
        MotionBlur(p=0),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=25, border_mode=0, p=0.5),
        #RandomContrast(0.3,p=0),
        PixelDropout(dropout_prob=0.005, p=0)])
        #RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=1)])
    
def noise_aug():
    return Compose([
        GaussNoise(var_limit=(1.0, 2.0), p=0),
        MotionBlur(p=1)
        #RandomContrast(0.3,p=0)
        ])

imgaug = strong_aug()


def scale_3c(img_3d, mask_3d):
    zz, xx, yy = np.where(mask_3d)
    xx_max, xx_min = max(xx), min(xx)
    yy_max, yy_min = max(yy), min(yy)
    xx_min_random = random.choice(list(range(-10, 10)))
    xx_max_random = random.choice(list(range(10,50)))
    # yy_min_random = random.choice(list(range(-10, 10)))
    # yy_max_random = random.choice(list(range(10,50)))
    img_crop = img_3d[:,xx_min-xx_min_random:xx_max+xx_max_random, yy_min-10:yy_max+10]
    mask_crop = mask_3d[:,xx_min-xx_min_random:xx_max+xx_max_random, yy_min-10:yy_max+10]
    return img_crop, mask_crop

def fill_in(img_crop, mask_crop):
    shape = img_crop.shape
    max_shape = max(shape[1:])
    zero_mask = np.zeros((shape[0], max_shape, max_shape),dtype=mask_crop.dtype)
    zero_img = np.zeros((shape[0], max_shape, max_shape),dtype=img_crop.dtype)
    if np.argmax(shape[1:]) == 0:
        start = (max_shape - shape[2])//2
        zero_img[:,:,start:start+shape[2]] = img_crop
        zero_mask[:,:,start:start+shape[2]] = mask_crop
    elif np.argmax(shape[1:]) == 1:
        start = (max_shape - shape[1])//2
        zero_img[:,start:start+shape[1],:] = img_crop
        zero_mask[:,start:start+shape[1],:] = mask_crop
    return zero_img, zero_mask
    
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

def hist_match_Echo(Echo_csv='/ailab/group/pjlab-medai/qianyi/jiaorushi/GLS/data/EchoNet-Dynamic/FileList.csv', 
                    Echo_folder='/ailab/group/pjlab-medai/qianyi/jiaorushi/GLS/data/EchoNet-Dynamic/Videos/'):
    vendor_dict = {}
    Echo_csv = pd.read_csv(Echo_csv)
    files = Echo_csv['FileName'].values.tolist()
    files = [Echo_folder + f + '.avi' for f in files]
    vendor_dict['Echo-Dynamic'] = files
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
            #target = np.load(file).transpose(1,0,2,3)[1:2]
            target = load_file(file)[np.newaxis]
            src = src[np.newaxis]
            src = hist_match(src, target)[0]
    return src
    
        
class Huaxi3d_Dataset(Dataset):
    def __init__(self, args, info_csv, split='train', size=224, transform=imgaug, root_path=None, seg_path='segment_npy_denoise/'):
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
        self.chmb_dict = {'a2c':0, 'a3c':1, 'a4c':2,'mid':3, 'mv':3, 'apex':3}
        self.args = args
        self.vendor_dict = None
        self.length = 32
        if self.args.prior:
            if 'a4c' in self.args.chamber:
                #self.pseudo_path = '/ailab/user/jincheng/Oncocardiology_huaxi_new_new/data/flow_Inference_seg_save/cycle_TransUnet_cst_temporal_flow_woResidual_CAMUS2d_True_224_a4c_a2c_a3c_tpral_1_uncer_0_hist_False_cls_True_prior_True_1_pretrain_True_240523/'
                self.pseudo_path = '/ailab/user/jincheng/Oncocardiology_huaxi_new_new/data/flow_Inference_seg_save/SpaceTimeUnet_Echo3d_True_112_a4c_a2c_a3c_prior_True_pretrain_True_240704/'
            if 'mid' in self.args.chamber:
                #self.pseudo_path = '/ailab/group/pjlab-medai/qianyi/jiaorushi/GLS/data/flow_Inference_seg_save/best_cycle_TransUnet_cst_temporal_flow_woResidual_huaxi2d_False_224_a4c_a2c_a3c_mid_mv_tpral_1_uncer_0_hist_False_cls_True_prior_True_1_pretrain_True_240408/'
                self.pseudo_path = '/ailab/user/jincheng/Oncocardiology_huaxi_new_new/data/flow_Inference_seg_save/SpaceTimeUnet_Huaxi3d_False_112_mid_mv_prior_True_pretrain_True_240716/'
    def resize_ratio(self, array):
        shape = array.shape
        assert shape[1] < shape[2]
        array = array[:,:,-shape[1]:]
        return array

    def getitem(self, idx):
        row = self.info_csv.iloc[idx]
        chamber = row['chamber']
        cycle_number = row['cycle_number']
        
        seg_chamber = chamber
        if 'root_path' in self.info_csv.columns.tolist():
            path = row['root_path']
        elif 'data' in row['path']:
            path = self.root_path
        else:
            path = '/ailab/group/pjlab-medai/qianyi/jiaorushi/GLS/data/raw_cycles_zll/'
        try:
            cycle_video_path = row['path'] + '_' + str(cycle_number) + '_' + chamber + '.npz'
            file_path = path + '/' + cycle_video_path
            video = np.load(file_path)
        except:
            cycle_video_path = row['path'] + '_' + str(cycle_number) + '_unknown' + '.npz'
            file_path = path + '/' + cycle_video_path
            video = np.load(file_path)
        gls = row['gls']
        array, fps = video['array'], video['fps']
        
        mask_path = self.seg_path + '/' + seg_chamber + '/' + '_'.join(row['path'].split('/')) + '_' + str(cycle_number) + '_' + seg_chamber + '.npy'
        if  os.path.exists(mask_path):
            cycle_segment = np.load(mask_path).astype(np.uint8)

        if self.args.prior:
            mask_path_pseudo = glob.glob(self.pseudo_path  + '/*/*/' + '_'.join(row['path'].split('/')) + '_' + str(cycle_number) + '_*' + '.nii.gz')
            cycle_segment_pseudo = load_file(mask_path_pseudo[0]).astype(np.uint8)
        
        frame_seg = remove_noise(cycle_segment_pseudo, cycle_segment_pseudo>0)
    
        if not np.random.randint(0,3):
            array, frame_seg = scale_3c(array, cycle_segment_pseudo)
            array, frame_seg = fill_in(array, frame_seg)

        frame = skimage.transform.resize(array, (self.length, self.size, self.size), 1) 
        frame_seg = skimage.transform.resize(frame_seg, (self.length, self.size, self.size), 0,anti_aliasing=False)
        if self.transform and self.split == 'train':
            frame = frame.transpose(1,2,0)
            frame_seg = frame_seg.transpose(1,2,0)
            augs = self.transform(image=frame, mask=frame_seg)
            frame, frame_seg = augs['image'], augs['mask']
            frame = frame.transpose(2,0,1)
            frame_seg = frame_seg.transpose(2,0,1)
        
        if self.split == 'train':
            gamma = random.uniform(0.1, 2.0)
            frame = gammacorrection((frame*255).astype('uint8'), gamma, self.vendor_dict)

        frame = (frame - np.amin(frame))/(np.amax(frame) - np.amin(frame))
        frame = frame[np.newaxis, :,:,:]
        frame_seg = frame_seg[np.newaxis,:,:,:]
        return cycle_video_path, frame, frame_seg, gls, self.chmb_dict[chamber]
    
    def __getitem__(self, idx):
        stop = False
        data = self.getitem(idx)
        # while not stop:
        #     try:
        #         data = self.getitem(idx)
        #         stop = True
        #     except:
        #         idx = random.randint(0,len(self.info_csv))
        return data

    def __len__(self):
        return len(self.info_csv)
    
class Huaxi2d_Dataset(Dataset):
    def __init__(self, args, info_csv, split='train', size=224, transform=imgaug, root_path=None, seg_path='segment_npy_denoise/', uncertainty=False, HistMatch=False):
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
            self.vendor_dict = get_vendor_dict('RVENetDatabase/codebook.csv',
                                          'RVENetDatabase/preprocessed/train_*/*.npy')
        else:
            self.vendor_dict = None
        if self.args.prior:
            self.pseudo_path = 'Inference_seg_save/'

    def resize_ratio(self, array):
        shape = array.shape
        assert shape[1] < shape[2]
        array = array[:,:,-shape[1]:]
        return array


    def __getitem__(self, idx):
        row = self.info_csv.iloc[idx]
        chamber = row['chamber']
        cycle_number = row['cycle_number']
        cycle_video_path = row['path'] + '_' + str(cycle_number) + '_' + chamber + '.npz'
        seg_chamber = chamber
        if 'data' in row['path']:
            path = self.root_path
        else:
            path = 'raw_cycles_zll/'
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
    def __init__(self, args, info_csv, split='train', size=224, transform=imgaug, root_path=None, seg_path='segment_npy_denoise/', uncertainty=False, temporal_interval=3, HistMatch=False):
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
            self.vendor_dict = hist_match_Echo()
        else:
            self.vendor_dict = None
        if self.args.prior:
            self.pseudo_path = '/ailab/group/pjlab-medai/qianyi/jiaorushi/GLS/data/flow_Inference_seg_save/best_cycle_TransUnet_cst_temporal_flow_woResidual_huaxi2d_False_224_a4c_a2c_a3c_mid_mv_tpral_1_uncer_0_hist_False_cls_True_prior_True_1_pretrain_True_240408/'
    def resize_ratio(self, array):
        shape = array.shape
        assert shape[1] < shape[2]
        array = array[:,:,-shape[1]:]
        return array


    def __getitem__(self, idx):
        row = self.info_csv.iloc[idx]
        chamber = row['chamber']
        cycle_number = row['cycle_number']
        cycle_video_path = row['path'] + '_' + str(cycle_number) + '_' + chamber + '.npz'
        seg_chamber = chamber
        if 'data' in row['path']:
            path = self.root_path
        else:
            path = '/ailab/group/pjlab-medai/qianyi/jiaorushi/GLS/data/raw_cycles_zll/'
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

        frame = array[[next_frame, slice_num]]
        if row['select_slice'] == 'random':
            frame_seg = cycle_segment_pseudo[[next_frame, slice_num]]
        else:
            frame_seg = np.array([cycle_segment_pseudo[next_frame],cycle_segment[slice_num]])

        frame_seg = remove_noise(frame_seg, frame_seg>0)

        if self.split == 'train':
            if random.choice([0,0,1]):
                frame, frame_seg = scale_3c(frame, frame_seg)
                frame, frame_seg = fill_in(frame, frame_seg)

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

        return cycle_video_path, slice_num, frame, frame_noise, repeat, frame_seg, gls, self.chmb_dict[chamber], np.array(self.beita_lv[chamber]), np.array(self.beita_myo[chamber])


    def __len__(self):
        return len(self.info_csv)


    

    

    
    


    
    







