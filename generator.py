from torch.utils.data import DataLoader
from load import huaxi_dataset
# from load import flow_dataset
from networks import EchoNet, unet
from networks.SpaceTimeUnet import make_a_video
# from networks.TransUnet import vit_seg_modeling_4flowPretrain
from networks.TransUnet.vit_seg_modeling import VisionTransformer
from networks.TransUnet.vit_seg_modeling import  VisionTransformer_cst_temporal_flow_woResidual
from networks.TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.EMA_VFI import feature_extractor, flow_estimation
from functools import partial
import sys
sys.path.append('load/')

import pandas as pd
import torch
from utils import losses
from monai.losses import DiceLoss as diceloss
from monai.metrics import DiceMetric
import sklearn
import glob
import numpy as np


def GLS_prior(info_csv, pred_csv):
    paths = pred_csv['path'].values.tolist()
    cycles = pred_csv['cycle_num'].values.tolist()
    gls_pred = np.array(pred_csv['gls'].values.tolist())
    path_cycle = np.array(list(zip(paths, cycles)))
    selected = gls_pred < 0.25
    selected_cycle = list(path_cycle[selected])
    
    info_paths = info_csv['path'].values.tolist()
    info_cycles = info_csv['cycle_number'].values.tolist()
    info_path_cycle = np.array(list(zip(info_paths, info_cycles)))
    anno = np.array(info_csv['select_slice'].values.tolist())
    labeled = anno != 'random'
    labeled_cycle = np.where(labeled)
    selected_labeled = list(info_path_cycle[labeled_cycle])

    selected_cycle.extend(selected_labeled)
    return selected_cycle

def csv_prior(info_csv, selected_cycle):
    paths = info_csv['path'].values.tolist()
    cycles = info_csv['cycle_number'].values.tolist()
    path_cycle = np.array(list(zip(paths, cycles)))
    
    path_cycle = [list(p) for p in path_cycle]
    selected_cycle = [list(p) for p in selected_cycle]
    
    selected = np.array([p_c in selected_cycle for p_c in path_cycle])
    part_csv = info_csv.iloc[selected]
    return part_csv
    

class DataGenerator:

    def __init__(self, args):
        self.args = args

    def get_part_csv(self, csv_file, split, chamber, semi=False):
        del_split = []
        for s in ['train', 'val', 'test']:
            if s not in split:
                del_split.append(s)
        for s in del_split:
            csv_file = csv_file[csv_file['split'] != s]
        new_csv = pd.DataFrame([])
        #part_csv = part_csv[part_csv['chamber'] != 'apex']
        chamber_list = chamber.split('_')
        for c in chamber_list:
            part_csv = csv_file[csv_file['chamber'] == c]
            if not semi:
                part_csv = part_csv[part_csv['select_slice']!='random']
            if split != 'train':
                part_csv = part_csv[part_csv['select_slice']!='random']
            new_csv = pd.concat([new_csv, part_csv])   
        return new_csv
    
    def check_exists(self, paths, target_path):
        idx_list = []
        for i in range(len(paths)):
            path = paths[i]
            segment = glob.glob(target_path + '/*' + '/' + '_'.join(path.split('/')) + '_*.npy')
            if len(segment)!=0:
                idx_list.append(i)
        return idx_list
    
    def choose_one_cycle(self, cycles_name, raw_path):
        paths = list(set(raw_path))
        raw_path = np.array(raw_path)
        cycles_name = np.array(cycles_name)
        idx_list = []
        for path in paths:
            idxs = np.where(raw_path == path)[0]
            choose = len(idxs)//2
            idx = idxs[choose]
            idx_list.append(idx)
        assert len(paths) == len(idx_list)
        return idx_list

    def regression_huaxi_train(self):
        print('using the huaxi dataset')
        data = {}
        info_csv = pd.read_csv(self.args.info_csv)
        
        train_csv = sklearn.utils.shuffle(self.get_part_csv(info_csv, 'train', self.args.chamber), random_state=1)
        train_leq_18 = train_csv[train_csv['gls'] <=18]
        train_grt_18 = train_csv[train_csv['gls'] > 18]
        print(len(train_leq_18), len(train_grt_18))
        balanced_grt_18 = train_grt_18.sample(n=len(train_leq_18), random_state=1)
        train_balanced = sklearn.utils.shuffle(pd.concat([train_leq_18, balanced_grt_18]), random_state=1)
        print(len(train_balanced), len(info_csv))
        print(train_balanced.head())
        #train_csv = train_csv.head(20)
        val_csv = self.get_part_csv(info_csv, 'val', self.args.chamber)
        test_csv = self.get_part_csv(info_csv, 'test', self.args.chamber)
        if 'crop' not in self.args.root_path:
            print('full image')
            train_ds = huaxi_dataset.Huaxi_Dataset(train_balanced, 'train', self.args.size, self.args.slices_per_cycle, root_path = self.args.root_path)
            val_ds = huaxi_dataset.Huaxi_Dataset(val_csv, 'val', self.args.size, self.args.slices_per_cycle, transform = None, root_path = self.args.root_path)
            test_ds =  huaxi_dataset.Huaxi_Dataset(test_csv, 'test', self.args.size, self.args.slices_per_cycle, transform = None, root_path = self.args.root_path)
        else:
            print('crop image')
            train_ds = huaxi_dataset.Huaxi_Dataset_crop(train_balanced, 'train', self.args.size, self.args.slices_per_cycle, root_path = self.args.root_path)
            val_ds = huaxi_dataset.Huaxi_Dataset_crop(val_csv, 'val', self.args.size, self.args.slices_per_cycle, transform = None, root_path = self.args.root_path)
            test_ds =  huaxi_dataset.Huaxi_Dataset_crop(test_csv, 'test', self.args.size, self.args.slices_per_cycle, transform = None, root_path = self.args.root_path)
            
        data['train'] = train_ds
        data['val'] = val_ds
        data['test'] = test_ds
        return data
    
    def seg_huaxi2d_train(self):
        print('using the huaxi2d dataset for segmentation')
        data = {}
        
        info_csv = pd.read_csv(self.args.info_csv)
        test_csv = self.get_part_csv(info_csv, 'test', self.args.chamber, self.args.forward_uncer)
        print('the number of test samples',len(test_csv))
        zll_csv = pd.read_csv('train_val_test_zll_selected_cycles_1227.csv')
        wqh_csv = pd.read_csv('train_val_test_wqh_selected_cycles.csv')
        info_csv = pd.concat([info_csv, zll_csv, wqh_csv])
        
        if self.args.prior:
            train_csv = self.get_part_csv(info_csv, 'train', self.args.chamber, True)
            val_csv = self.get_part_csv(info_csv, 'val', self.args.chamber, False)
            if self.args.GLS_prior:
                pred_csv_zll = pd.read_csv('zll_a4c_a2c_a3c_mid_mv_train_val_test.csv')
                pred_csv_zhaoli = pd.read_csv('zhaoli_a4c_a2c_a3c_mid_mv_train_val_test.csv')
                pred_csv_wqh = pd.read_csv('wqh_a4c_a2c_a3c_mid_mv_train_val_test.csv')
                print('training samples', len(train_csv))
                pred_csv = pd.concat([pred_csv_zhaoli, pred_csv_zll, pred_csv_wqh])
                selected_cycles = GLS_prior(info_csv, pred_csv)
                train_csv = csv_prior(train_csv, selected_cycles)
                print('training samples after gls prior', len(train_csv))
        else:
            train_csv = self.get_part_csv(info_csv, 'train', self.args.chamber, False)
            val_csv = self.get_part_csv(info_csv, 'val', self.args.chamber, False)
            
        
        print('the number of training samples',len(train_csv))
        print('the number of val samples',len(val_csv))

        if self.args.temporal_interval > 0:
            train_ds = huaxi_dataset.Huaxi2d_Dataset_temporal(self.args, train_csv , 'train', self.args.size, root_path = self.args.root_path, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, temporal_interval=self.args.temporal_interval, HistMatch=self.args.HistMatch)
            val_ds = huaxi_dataset.Huaxi2d_Dataset_temporal(self.args, val_csv, 'val', self.args.size, root_path = self.args.root_path, transform=None, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, temporal_interval=self.args.temporal_interval, HistMatch=False)
            test_ds = huaxi_dataset.Huaxi2d_Dataset_temporal(self.args, test_csv, 'test', self.args.size, root_path = self.args.root_path, transform=None, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, temporal_interval=self.args.temporal_interval, HistMatch=False)
        else:
            train_ds = huaxi_dataset.Huaxi2d_Dataset(self.args, train_csv , 'train', self.args.size, root_path = self.args.root_path, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, HistMatch=self.args.HistMatch)
            val_ds = huaxi_dataset.Huaxi2d_Dataset(self.args, val_csv, 'val', self.args.size, root_path = self.args.root_path, transform=None, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, HistMatch=False)
            test_ds = huaxi_dataset.Huaxi2d_Dataset(self.args, test_csv, 'test', self.args.size, root_path = self.args.root_path, transform=None, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, HistMatch=False)
        data['train'] = train_ds
        data['val'] = val_ds
        data['test'] = test_ds
        data['train_csv'] = train_csv
        return data
    
    def cls_huaxi2d_train(self):
        print('using the huaxi2d dataset for cls')
        data = {}
        
        info_csv = pd.read_csv(self.args.info_csv)
        zll_csv = pd.read_csv('train_val_test_zll_selected_cycles_1227.csv')
        wqh_csv = pd.read_csv('train_val_test_wqh_selected_cycles.csv')
        info_csv = pd.concat([info_csv, zll_csv, wqh_csv])
        test_csv = info_csv[info_csv['split'] == 'test']
        val_csv = info_csv[info_csv['split'] == 'val']
        train_csv = info_csv[info_csv['split'] == 'train']
        print('the number of test samples',len(test_csv))
        print('the number of training samples',len(train_csv))
        print('the number of val samples',len(val_csv))

        if self.args.temporal_interval > 0:
            train_ds = huaxi_dataset.Huaxi2d_Dataset_temporal(self.args, train_csv , 'train', self.args.size, root_path = self.args.root_path, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, temporal_interval=self.args.temporal_interval, HistMatch=self.args.HistMatch)
            val_ds = huaxi_dataset.Huaxi2d_Dataset_temporal(self.args, val_csv, 'val', self.args.size, root_path = self.args.root_path, transform=None, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, temporal_interval=self.args.temporal_interval, HistMatch=False)
            test_ds = huaxi_dataset.Huaxi2d_Dataset_temporal(self.args, test_csv, 'test', self.args.size, root_path = self.args.root_path, transform=None, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, temporal_interval=self.args.temporal_interval, HistMatch=False)
        else:
            train_ds = huaxi_dataset.Huaxi2d_Dataset(self.args, train_csv , 'train', self.args.size, root_path = self.args.root_path, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, HistMatch=self.args.HistMatch)
            val_ds = huaxi_dataset.Huaxi2d_Dataset(self.args, val_csv, 'val', self.args.size, root_path = self.args.root_path, transform=None, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, HistMatch=False)
            test_ds = huaxi_dataset.Huaxi2d_Dataset(self.args, test_csv, 'test', self.args.size, root_path = self.args.root_path, transform=None, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, HistMatch=False)
        data['train'] = train_ds
        data['val'] = val_ds
        data['test'] = test_ds
        data['train_csv'] = train_csv
        return data

    def Huaxi3d(self):
        print('using the huaxi3d dataset for segmentation')
        data = {}
        info_csv = pd.read_csv(self.args.info_csv)
        test_csv = self.get_part_csv(info_csv, 'test', self.args.chamber, False)
        print('the number of test samples',len(test_csv))
        zll_csv = pd.read_csv('train_val_test_zll_selected_cycles_1227.csv')
        wqh_csv = pd.read_csv('train_val_test_wqh_selected_cycles.csv')
        info_csv = pd.concat([info_csv, zll_csv, wqh_csv])
        train_csv = self.get_part_csv(info_csv, 'train', self.args.chamber, True)
        val_csv = self.get_part_csv(info_csv, 'val', self.args.chamber, False)
        # if self.args.prior:
        #     train_csv = self.get_part_csv(info_csv, 'train', self.args.chamber, True)
        #     val_csv = self.get_part_csv(info_csv, 'val', self.args.chamber, False)
        #     if self.args.chamber != 'mid_mv':
        #         pred_csv = pd.read_csv('a4c_a3c_a2c_train_val_test.csv')
        #     else:
        #         pred_csv = pd.read_csv('mid_mv_prior_True_pretrain_True_240716_all_mid_mv_apex_train_val_test.csv')
        #         pred_csv = pred_csv[pred_csv['root_path'] == 'raw_cycles_HSCT/']
        #     pred_csv = pred_csv.dropna(subset=['gls', 'gls_gt'])
        #     print(pred_csv.head())
        #     info_csv = info_csv[info_csv['root_path'] == 'raw_cycles_HSCT/']
        #     selected_cycles = GLS_prior(info_csv, pred_csv)
        #     train_csv = csv_prior(train_csv, selected_cycles)
            
        print('the number of training samples',len(train_csv))
        print('the number of val samples',len(val_csv))

        train_ds = huaxi_dataset.Huaxi3d_Dataset(self.args, train_csv , 'train', self.args.size, root_path = self.args.root_path)
        val_ds = huaxi_dataset.Huaxi3d_Dataset(self.args, val_csv, 'val', self.args.size, root_path = self.args.root_path, transform=None)
        test_ds = huaxi_dataset.Huaxi3d_Dataset(self.args, test_csv, 'test', self.args.size, root_path = self.args.root_path, transform=None)
        
        print('the number of train samples', len(train_ds))
        data['train'] = train_ds
        data['val'] = val_ds
        data['test'] = test_ds
        data['train_csv'] = train_csv
        return data
    
   
    
    def flow_huaxi2d_train(self):
        print('using the huaxi2d dataset for flow estimation')
        data = {}
        
        info_csv = pd.read_csv(self.args.info_csv)
        test_csv = info_csv[info_csv['split']=='test']
        print('the number of test samples',len(test_csv))
        zll_csv = pd.read_csv('train_val_test_zll_selected_cycles_1227.csv')
        wqh_csv = pd.read_csv('train_val_test_wqh_selected_cycles.csv')
        info_csv = pd.concat([info_csv, zll_csv, wqh_csv])
        train_csv = self.get_part_csv(info_csv, 'train', self.args.chamber, semi=True)
        val_csv = self.get_part_csv(info_csv, 'val', self.args.chamber, semi=True)
        print('the number of training samples',len(train_csv))
        print('the number of val samples',len(val_csv))

        train_ds = flow_dataset.Huaxi2d_Dataset_flow(self.args, train_csv , 'train', self.args.size, root_path = self.args.root_path, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, temporal_interval=self.args.temporal_interval, HistMatch=self.args.HistMatch)
        val_ds = flow_dataset.Huaxi2d_Dataset_flow(self.args, val_csv, 'val', self.args.size, root_path = self.args.root_path, transform=None, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, temporal_interval=self.args.temporal_interval, HistMatch=False)
        test_ds = flow_dataset.Huaxi2d_Dataset_flow(self.args, test_csv, 'test', self.args.size, root_path = self.args.root_path, transform=None, seg_path=self.args.update_folder, uncertainty=self.args.forward_uncer, temporal_interval=self.args.temporal_interval, HistMatch=False)
        
        print('the number of train samples', len(train_ds))
        data['train'] = train_ds
        data['val'] = val_ds
        data['test'] = test_ds
        data['train_csv'] = train_csv
        return data
              

class NetGenerator:
    
    def __init__(self, args):
        self.args = args

    def EchoNet(self):
        print('using EchoNet')
        net = EchoNet.load_EchoNet(self.args.model_name, True, self.args.add_classfy)
        return net
    
    def TransUnet(self):
        print('using TransUnet for segmentation')
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = self.args.num_classes
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(self.args.size / 16), int(self.args.size / 16))
        net = VisionTransformer(self.args, config_vit, img_size=self.args.size, num_classes=config_vit.n_classes)
        return net

    
    def TransUnet_cst_temporal_flow_woResidual(self):
        print('using TransUnet_cst_tamporal_flow_noResidual for segmentation')
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = self.args.num_classes
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(self.args.size / 16), int(self.args.size / 16))
        net = VisionTransformer_cst_temporal_flow_woResidual(self.args, config_vit, img_size=self.args.size, num_classes=config_vit.n_classes)
        return net

    
    def UNet(self):
        net = unet.UNet(3,self.args.num_classes)
        return net
    
    def UNet_cst_temporal_flow_woResidual(self):
        print('using UNet_cst_temporal_flow_woResidual for segmentation')
        net = unet.UNet_cst_temporal_flow_woResidual(self.args, 3,self.args.num_classes)
        return net

    def SpaceTimeUnet(self):
        print('using SpaceTimeUnet')
        net = make_a_video.SpaceTimeUnet(dim = 64,
                                                 channels = self.args.in_channels,
                                                 out_channels = self.args.num_classes,
                                                 dim_mult = (1, 2, 4, 8),
                                                 resnet_block_depths = (1, 1, 1, 2),
                                                 temporal_compression = (False, False, False, True),
                                                 self_attns = (False, False, False, True),
                                                 condition_on_timestep = False,
                                                 attn_pos_bias = False,
                                                 flash_attn = True)
        return net

    
    
def init_model_config(F=32, W=7, depth=[2, 2, 2, 4, 4]):
    '''This function should not be modified'''
    return { 
        'embed_dims':[F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads':[8*F//32, 16*F//32],
        'mlp_ratios':[4, 4],
        'qkv_bias':True,
        'norm_layer':partial(torch.nn.LayerNorm, eps=1e-6), 
        'depths':depth,
        'window_sizes':[W, W]
    }, {
        'embed_dims':[F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths':depth,
        'num_heads':[8*F//32, 16*F//32],
        'window_sizes':[W, W],
        'scales':[4, 8, 16],
        'hidden_dims':[4*F, 4*F],
        'c':F
    }

MODEL_CONFIG = {
    'LOGNAME': 'ours',
    'MODEL_TYPE': (feature_extractor, flow_estimation),
    'MODEL_ARCH': init_model_config(
        F = 32,
        W = 7,
        depth = [2, 2, 2, 4, 4]
    )
}
        
        

class LossGenerator:

    def __init__(self, args):
        self.args = args

    def MSELoss(self):
        print('using MSELoss')
        loss = torch.nn.MSELoss()
        return loss
    
    def BCELoss(self):
        print('using MSELoss')
        loss = torch.nn.BCEWithLogitsLoss()
        return loss

    def BMCLoss(self):
        print('using BMCLoss')
        loss = losses.BMCLoss(self.args.init_noise_sigma)
        return loss
    
    def ClassfyLoss(self):
        print('using MSELoss and CELoss')
        loss_MSE = torch.nn.MSELoss()
        loss_CE = torch.nn.CrossEntropyLoss()
        return loss_MSE, loss_CE
    
    def SmoothL1Loss(self):
        print('using smoothL1 and CELoss')
        loss_MSE = torch.nn.SmoothL1Loss()
        loss_CE = torch.nn.CrossEntropyLoss()
        return loss_MSE, loss_CE
    
    def DiceLoss(self):
        # if self.args.weighted_dice or self.args.forward_uncer is not None:
        #     print('using weighted dice loss')
        #     loss = losses.weighted_dice()
        # else:
        if 1:
            print('using dice loss')
            loss = diceloss(softmax=True)
        return loss
    
    def CELoss(self):
        if self.args.weighted_dice or self.args.forward_uncer is not None:
            print('using weighted ce loss')
            loss = losses.pixel_weighted_ce()
        else:
            print('using ce loss')
            loss = torch.nn.CrossEntropyLoss()
        return loss
    
    def SSLLoss(self):
        print('using multilabel loss, reconstruction loss for ssl')
        loss = losses.SSLLoss()
        return loss
        
    
    
        
        
