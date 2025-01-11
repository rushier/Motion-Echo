import sys
import numpy as np
from sklearn.metrics import precision_recall_curve, accuracy_score
import random
import os
import pandas as pd
import time
import csv
import argparse
import tempfile
import math
import warnings
import datetime
import matplotlib.pyplot as plt
import sklearn

import generator
import test
from utils.utils import AverageMeter, distributed_all_gather, get_logger
from utils import ramps

from xml.dom.expatbuilder import theDOMImplementation
from torch.utils.data import DataLoader,Dataset,ConcatDataset
import torch   
#import torch.nn as nn
#from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from tensorboardX import SummaryWriter
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.utils import one_hot
warnings.filterwarnings("ignore")
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank%torch.cuda.device_count()
    else:
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    dist.init_process_group(backend=args.dist_backend,init_method=args.dist_url,world_size=args.world_size,rank=args.rank)
    dist.barrier()

def write2csv(node_info,pred):
    node_info = [ node_info[n].split('_').append(pred[n]) for n in range(len(node_info))]
    return node_info


def get_dataloader(args):
    gen = generator.DataGenerator(args)
    loader_name = args.dataset 
    print(loader_name)
    dataset = getattr(gen, loader_name)()
    return dataset

def get_network(args):
    gen = generator.NetGenerator(args)
    loader_name = args.network
    net = getattr(gen, loader_name)()
    return net

def get_loss(args):
    gen = generator.LossGenerator(args)
    loader_name = args.loss
    loss = getattr(gen, loader_name)()
    return loss

def save_checkpoint(state, model_save_name, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = args.model_dir + "%s/"%(model_save_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)

def remove_dist(state_dict):
    new_dict = {}
    for key in list(state_dict.keys()):
            if key.startswith('module.'):
                new_key = key[7:]
                new_dict[new_key] = state_dict[key]
            else:
                print(key)
                new_dict[key] = state_dict[key]
    return new_dict

def main(args):
    init_distributed_mode(args=args)#启动多gpu
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
 
    dataset = get_dataloader(args)
    train_data = dataset['train']
    val_data = dataset['val']
    test_data = dataset['test']
    
    if args.added_CAMUS:
        print('CAMUS')
        args.dataset = 'CAMUS3d'
        args.info_csv =  'CAMUS_dataset_official.csv'
        CAMUS_dataset = get_dataloader(args)
        CAMUS_train_data = CAMUS_dataset['train']
        CAMUS_val_data = CAMUS_dataset['val']
        CAMUS_test_data = CAMUS_dataset['test']
    
        CAMUS_labeled_idxs = np.array(range(len(CAMUS_train_data))) + len(train_data)
        CAMUS_labeled_idxs = list(CAMUS_labeled_idxs)

        train_data = ConcatDataset([train_data, CAMUS_train_data])
        val_data = ConcatDataset([val_data, CAMUS_val_data])

    if args.added_Echo:
        print('Echo3d')
        args.dataset = 'Echo3d'
        args.info_csv =  'EchoNet_dynamic_seg.csv'
        args.root_path = 'EchoNet-Dynamic/Videos'
        Echo_dataset = get_dataloader(args)
        Echo_train_data = Echo_dataset['train']
        Echo_val_data = Echo_dataset['val']
        Echo_test_data = Echo_dataset['test']
    
        Echo_labeled_idxs = np.array(range(len(Echo_train_data))) + len(train_data)
        Echo_labeled_idxs = list(Echo_labeled_idxs)

        train_data = ConcatDataset([train_data, Echo_train_data])
        val_data = ConcatDataset([val_data, Echo_val_data])

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        train_sampler = None
        val_sampler = None
    train_dataloader = DataLoader(train_data,shuffle=(train_sampler is None),num_workers = args.nw, batch_size = args.bs,pin_memory=True, sampler=train_sampler,drop_last=True)
    val_dataloader = DataLoader(val_data,shuffle = False,num_workers = args.nw,batch_size=args.bs,sampler=val_sampler,drop_last=True)
    test_dataloader = DataLoader(test_data,shuffle = False,num_workers = args.nw,batch_size=args.bs,sampler=None,drop_last=True)

    net = get_network(args)
    net.to(args.device)
    ce_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.,10.,10.]).to(args.device) )
    dice_loss = get_loss(args)
    metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    pg = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=eval(args.ms), gamma=0.1)
    model_save_name = '_'.join([args.network, args.dataset, str(args.added_CAMUS), str(args.size), args.chamber, 'prior_' + str(args.prior), 'pretrain_' + str(args.pretrain is not None), datetime.date.today().strftime('%y%m%d')]) 
    lowerest_loss = float("inf")
    epoch_resume = 0
   
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)


        net.load_state_dict(remove_dist(checkpoint['state_dict']), strict=True)


    if args.continue_train:
        optimizer.load_state_dict(checkpoint['opt_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        epoch_resume = checkpoint["epoch"] + 1
        lowerest_loss = checkpoint["lowerest_loss"]
    if args.distributed:
        if args.syncBN:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(args.device) 
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu],find_unused_parameters=False)
        
        
    if args.test:
        val_loss = test(0,val_dataloader,net,metric,ce_loss, dice_loss,args.device)
        return
    
    log_path = args.model_dir + '/' + "%s/"%(model_save_name)
    if args.rank == 0:
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    logger = get_logger(log_path + '/logger.log')
    if args.rank == 0:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger.info(args)
        logger.info('the length of train_data {} and val_data {}'.format(str(len(train_data)), str(len(val_data))))
    
 
    if args.rank == 0:
        writer1 = SummaryWriter('tensorboard/'+model_save_name)

    for e in range(epoch_resume, args.epoch):
        if args.distributed:
            train_sampler.set_epoch(e)
        time1 = time.time()
        train_loss = train(e,train_dataloader,net,metric,ce_loss, dice_loss,optimizer,args.device,logger)
        time2 = time.time() - time1
        scheduler.step()

        if args.rank == 0:
            # logger.info('train Epoch: [{}]\t'
            #             'Loss {:.4f}\t'
            #             'Time {:.4f}\t'
            #             .format(
            #                 e, train_loss.item(), time2))
            writer1.add_scalar('train_loss', train_loss, global_step=e)
            


        if e%2 == 0:
            test_loss = val(e,test_dataloader,net,metric, ce_loss, dice_loss,args.device, logger)
            val_loss = val(e,val_dataloader,net,metric, ce_loss, dice_loss,args.device, logger)
            #print(all_pred)
            if args.rank == 0:
                # logger.info('val   Epoch: [{}]\t'
                #         'Loss {:.4f}\t'
                #         .format(e, val_loss.item()))
                writer1.add_scalar('val_loss', val_loss, global_step=e)
                if val_loss < lowerest_loss:
                    lowerest_loss = val_loss
                    save_checkpoint({
                        'epoch': e,
                        'state_dict': net.state_dict(),
                        'opt_dict': optimizer.state_dict(),
                        'scheduler_dict': scheduler.state_dict(),
                        'lowerest_loss': lowerest_loss
                    }, model_save_name, 'lowerest_loss' + '.pth.tar')
                if not os.path.exists('./imgs'):
                    os.makedirs('./imgs')
                save_figure = './imgs/' + str(e) + '_' + args.chamber
                if not os.path.exists(save_figure):
                    os.makedirs(save_figure)
                #plot_figure(all_labels, all_pred, save_figure, split='val')
                
        if args.rank == 0:
            if e % args.model_save_freq ==0:
                if args.distributed:
                    save_checkpoint({
                        'epoch': e,
                        'state_dict': net.state_dict(),
                        'opt_dict': optimizer.state_dict(),
                        'scheduler_dict': scheduler.state_dict(),
                        'lowerest_loss': lowerest_loss}, 
                        model_save_name,
                        'checkpoint_' + str(e) + '.pth.tar')
                else:
                    save_checkpoint({
                        'epoch': e,
                        'state_dict1': net.state_dict(),
                        'opt_dict': optimizer.state_dict(),
                        'scheduler_dict': scheduler.state_dict(),
                        'lowerest_loss': lowerest_loss}, 
                        model_save_name,
                        'checkpoint_' + str(e) + '.pth.tar')
                    
                
def train(ep, train_dataloader, model, metric, ce_loss, dice_loss, optimizer, device, logger):
    mean_loss = torch.zeros(1).to(device)
    model.train()
    batch_start = 0
    all_pred = np.zeros(((len(train_dataloader)) * args.bs))
    all_labels = np.zeros(((len(train_dataloader))*args.bs))
    best_mean_loss = 100000
    run_acc = AverageMeter()
    for step, data in enumerate(train_dataloader):
        file_path, frame, mask, gls, chamber = data
        frame = frame.type(torch.FloatTensor)
        frame = frame.to(device)    
        mask = mask.type(torch.LongTensor)
        mask = mask.to(device)
        mask[mask>2] = 0        
        outputs = model(frame)
        outputs_soft = torch.softmax(outputs, dim=1)
    

        mask_onehot = one_hot(mask, num_classes=3)
        loss = 0.5 * ce_loss(outputs, mask_onehot) + 1*dice_loss(
            outputs, mask_onehot)
        
        pseudo_outputs = torch.argmax(
            outputs_soft.detach(), dim=1, keepdim=False)


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        
        #calculate diecs each class
        metric.reset()
        metric(y_pred=one_hot(pseudo_outputs.unsqueeze(1),num_classes=3), y=mask)
        dice, not_nans = metric.aggregate()
        dice = dice.cuda(args.rank)
        if args.distributed:
            dice_list, not_nans_list = distributed_all_gather(
                [dice, not_nans], out_numpy=True)
            for al, nl in zip(dice_list, not_nans_list):
                run_acc.update(al, n=nl)
        else:
            run_acc.update(dice.cpu().numpy(), n=not_nans.cpu().numpy())

        array_dict = {'frame':frame.cpu().numpy(), 'mask':mask.cpu().detach().numpy(),'pred':pseudo_outputs.cpu().detach().numpy()}
        np.savez('training_exers.npz', **array_dict)
            
    if args.rank==0:
        logger.info('{} ep:{:0>3d} mean_loss:{:.4f} Dice lv {:.4f} Dice lvo {:.4f} Dice bg {:.4f} lr:{:.4f} '.format('Train', ep, mean_loss.item(), run_acc.avg[1], run_acc.avg[2], run_acc.avg[0], optimizer.param_groups[0]['lr']))
        torch.cuda.empty_cache()
        
    return mean_loss


def val(ep, train_dataloader, model, metric, ce_loss, dice_loss, device, logger):
    mean_loss = torch.zeros(1).to(device)
    model.eval()
    batch_start = 0
    all_pred = np.zeros(((len(train_dataloader)) * args.bs))
    all_labels = np.zeros(((len(train_dataloader))*args.bs))
    best_mean_loss = 100000
    run_acc = AverageMeter()
    with torch.no_grad():
        for step, data in enumerate(train_dataloader):
            file_path, frame, mask, gls, chamber = data
            frame = frame.type(torch.FloatTensor)
            frame = frame.to(device)    
            mask = mask.type(torch.LongTensor)
            mask = mask.to(device)        
            mask[mask>2] = 0        
            outputs = model(frame)
            outputs_soft = torch.softmax(outputs, dim=1)
            mask_onehot = one_hot(mask, num_classes=3)
            loss = 0.5 * ce_loss(outputs, mask_onehot) + dice_loss(
                outputs, mask_onehot)
            # cross pseudo losses
            pseudo_outputs = torch.argmax(
                outputs_soft.detach(), dim=1, keepdim=False)
            
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
            
                    #calculate diecs each class
            metric.reset()
            metric(y_pred=one_hot(pseudo_outputs.unsqueeze(1),num_classes=3), y=mask)
            dice, not_nans = metric.aggregate()
            dice = dice.cuda(args.rank)
            if args.distributed:
                dice_list, not_nans_list = distributed_all_gather(
                    [dice, not_nans], out_numpy=True)
                for al, nl in zip(dice_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(dice.cpu().numpy(), n=not_nans.cpu().numpy())
            
    if args.rank==0:
        logger.info('{} ep:{:0>3d} mean_loss:{:.4f} Dice lv {:.4f} Dice lvo {:.4f} Dice bg {:.4f}'.format('Val  ', ep, mean_loss.item(), run_acc.avg[1], run_acc.avg[2], run_acc.avg[0]))
        torch.cuda.empty_cache()
        
    return mean_loss

def test(ep, train_dataloader, model1, metric1, ce_loss, dice_loss, device):
    mean_loss = torch.zeros(1).to(device)
    model1.eval()
    batch_start = 0
    all_pred = np.zeros(((len(train_dataloader)) * args.bs))
    all_labels = np.zeros(((len(train_dataloader))*args.bs))
    best_mean_loss = 100000
    run_acc1 = AverageMeter()
    
    for step, data in enumerate(train_dataloader):
        file_path, frame, mask, gls, chamber = data
        # np.save('ipt.npy', video_array_aug.cpu().numpy())
        #print(target_order)
        #print(torch.max(video_array_aug))
        frame = frame.type(torch.FloatTensor)
        frame = frame.to(device)    
        mask = mask.type(torch.LongTensor)
        mask = mask.to(device)        
        
        outputs1 = model1(frame)
        outputs_soft1 = torch.softmax(outputs1, dim=1)        
        mask_onehot = one_hot(mask, num_classes=3)

        loss = 0.5 * ce_loss(outputs1, mask_onehot) + dice_loss(outputs1, mask_onehot)

        # cross pseudo losses
        pseudo_outputs1 = torch.argmax(
            outputs_soft1.detach(), dim=1, keepdim=False)
        
        print(file_path)
        save_folder = 'res_segmentations/CAMUStrain/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        
                #calculate diecs each class
        metric1.reset()
        metric1(y_pred=one_hot(pseudo_outputs1.unsqueeze(1),num_classes=3), y=mask)
        dice, not_nans = metric1.aggregate()
        dice = dice.cuda(args.rank)
        if args.distributed:
            dice_list, not_nans_list = distributed_all_gather(
                [dice, not_nans], out_numpy=True)
            for al, nl in zip(dice_list, not_nans_list):
                run_acc1.update(al, n=nl)
        else:
            run_acc1.update(dice.cpu().numpy(), n=not_nans.cpu().numpy())

    if args.rank==0:
        print('{} ep:{:0>3d} mean_loss:{:.4f} Dice lv {:.4f} Dice lvo {:.4f} Dice bg {:.4f}'.format('Val   ', ep, mean_loss.item(), run_acc1.avg[1], run_acc1.avg[2], run_acc1.avg[0]))
        torch.cuda.empty_cache()
        
    return mean_loss






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',default=None )
    parser.add_argument('--task',default='regression',help='seg or regression or ssl')
    parser.add_argument('--dataset',default='huaxi',help='huaxi or open')
    parser.add_argument('--phase',default='train',help='train or pretrain')
    parser.add_argument('--network',default='EchoNet',help='EchoNet or Swin_UNetr or Swin_SSL' )
    parser.add_argument('--model_name',default='r2plus1d_18' )
    parser.add_argument('--add_classfy',action='store_true')
    parser.add_argument('--continue_train',action='store_true')
    parser.add_argument('--prior',action='store_true')
    parser.add_argument('--loss',default='MSELoss' )
    parser.add_argument('--chamber',default='all' )
    parser.add_argument('--info_csv',default='Open_dataset.csv' )
    parser.add_argument('--vendor_dict',default=None)
    parser.add_argument('--csv_save',default=None,help='csv save path for the testing results' )
    parser.add_argument("--epoch", default=100,type=int)
    parser.add_argument("--initial_epoch", default=0,type=int)
    parser.add_argument("--pretrain", default=None,type=str)
    parser.add_argument("--lr", default=1e-3,type=float)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument("--bs", default=8,type=int,help='batch_size')
    parser.add_argument("--ms", default = '[30, 50]')
    parser.add_argument("--test",action='store_true')
    parser.add_argument("--size", default=256,type=int)
    parser.add_argument("--slices_per_cycle", default=32,type=int)
    parser.add_argument("--print_freq", default = 30, type=int)
    parser.add_argument('--model_dir',default='save_models/')
    parser.add_argument("--model_save_freq", default=2,type=int)
    parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
    
    # settings for distribution
    parser.add_argument("--seed", default=1,type=int)
    parser.add_argument("--nw", default=0,type=int)
    parser.add_argument("--local-rank", default=0,type=int)
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--rank", default=0)
    parser.add_argument('--distributed',default=True,type=bool)
    parser.add_argument("--device", default='cuda')
    parser.add_argument('--syncBN',type=bool,default=True)
    parser.add_argument('--world_size',default=2,type=int,help='number of distributed processes')
    parser.add_argument('--dist-url',default='env://',help='url uses to set up distributed training')
    
    
    # settings for Swin_SSL
    parser.add_argument("--spatial_dims", default = 3, type=int)
    parser.add_argument("--feature_size", default = 48, type=int)
    parser.add_argument("--dropout_path_rate", default = 0.3, type=int)
    parser.add_argument('--use_checkpoint',action='store_true')
    
    # settings for MAE_SSL
    parser.add_argument("--norm_pix_loss",action='store_true')
    parser.add_argument("--MAEPretrain", default=None,type=str)

    parser.add_argument("--in_channels", default = 1, type=int)
    parser.add_argument("--num_classes", default = 3, type=int)
    parser.add_argument("--added_CAMUS",action='store_true')
    parser.add_argument("--added_Echo",action='store_true')
        
    args = parser.parse_args()
    
    if torch.cuda.is_available() is False:
        raise EnvironmentError('not find gpu')
    
    main(args)

        
    




            






















