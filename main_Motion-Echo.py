import sys
import numpy as np
from sklearn.metrics import precision_recall_curve, accuracy_score
import random
import os
import pandas as pd
import time
import csv
import argparse

import math
import warnings
import datetime
import matplotlib.pyplot as plt


import generator
import test
from utils.utils import AverageMeter,  get_logger, TwoStreamBatchSampler
from utils import ramps, metrics
from utils.losses import  multi_scale_flow

from xml.dom.expatbuilder import theDOMImplementation
from torch.utils.data import DataLoader,Dataset,ConcatDataset
import torch   
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from tensorboardX import SummaryWriter
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.utils.enums import MetricReduction
from monai.networks.utils import one_hot
from medpy import metric

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

def calculate_metric_percase(pred, gt):
    max_num = np.amax(gt)
    values_list = []
    for i in range(1, max_num+1):
        p, g = pred==i, gt==i
        p[p > 0] = 1
        g[g > 0] = 1
        dc = metric.binary.dc(p, g)
        jc = metric.binary.jc(p, g)
        hd = metric.binary.hd95(p, g)
        asd = metric.binary.asd(p, g)
        values_list.extend([dc, jc, hd, asd])
    return values_list

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

def update_metric(metric, run_acc, pseudo_outputs, mask, num_classes):
    metric.reset()
    metric(y_pred=one_hot(pseudo_outputs.unsqueeze(1),num_classes=num_classes), y=mask)
    dice, not_nans = metric.aggregate()
    dice = dice.cuda()
    if 1:
        run_acc.update(dice.cpu().numpy(), n=not_nans.cpu().numpy())
    return run_acc, dice

def get_dataloader(args):
    gen = generator.DataGenerator(args)
    loader_name = args.task + '_' + args.dataset + '_' + args.phase
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
    
def write2csv(res_list, csv_name):
    folder = '/'.join(csv_name.split('/')[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = open(csv_name, 'w', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerows(res_list)
    file.close()  

def main(args):
    init_distributed_mode(args=args)
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
    Huaxi_train = dataset['train_csv']
    
    if args.dataset == 'huaxi2d' or args.dataset == 'Echo2d':
        slice_num = np.array(Huaxi_train['select_slice'].values.tolist())
        Huaxi_unlabeled_idxs = list(np.where(slice_num == 'random')[0])
        Huaxi_labeled_idxs = list(np.where(slice_num != 'random')[0])
        print('the number of labeled samples in training', len(Huaxi_labeled_idxs))
        print('the number of unlabeled samples in training', len(Huaxi_unlabeled_idxs))
    else:
        slice_num = np.array(Huaxi_train['slice_num'].values.tolist())
        Huaxi_unlabeled_idxs = list(np.where(slice_num == 'random')[0])
        Huaxi_labeled_idxs = list(np.where(slice_num != 'random')[0])
        print('the number of labeled samples in training', len(Huaxi_labeled_idxs))
        
    if args.added_CAMUS:
        print('CAMUS')
        args.dataset = 'CAMUS2d'
        args.info_csv =  'CAMUS_dataset_official.csv'
        CAMUS_dataset = get_dataloader(args)
        CAMUS_train_data = CAMUS_dataset['train']
        CAMUS_val_data = CAMUS_dataset['val']
        CAMUS_test_data = CAMUS_dataset['test']
    
        CAMUS_labeled_idxs = np.array(range(len(CAMUS_train_data))) + len(train_data)
        CAMUS_labeled_idxs = list(CAMUS_labeled_idxs)
        Huaxi_labeled_idxs.extend(CAMUS_labeled_idxs)

        train_data = ConcatDataset([train_data, CAMUS_train_data])
        val_data = ConcatDataset([val_data, CAMUS_val_data])

    if args.added_EchoDynamic:
        print('added_EchoDynamic')
        args.dataset = 'Echo2d'
        args.root_path = 'EchoNet-Dynamic/Videos'
        args.info_csv =  'EchoNet_dynamic_seg.csv'
        Echo_dataset = get_dataloader(args)
        Echo_train_data = Echo_dataset['train']
        Echo_val_data = Echo_dataset['val']
        Echo_test_data = Echo_dataset['test']
        Echo_unlabeled_idxs = np.array(range(len(Echo_train_data))) + len(train_data)
        Echo_unlabeled_idxs = list(Echo_unlabeled_idxs)
        Huaxi_unlabeled_idxs.extend(Echo_unlabeled_idxs)
        train_data = ConcatDataset([train_data, Echo_train_data])

    if not args.test and args.added_CAMUS:
        test_data = ConcatDataset([test_data, CAMUS_test_data])
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
        
    if not args.prior:
        train_dataloader = DataLoader(train_data,shuffle = True, batch_sampler=None,num_workers = args.nw,batch_size=args.bs,worker_init_fn=worker_init_fn)
    else:
        batch_sampler = TwoStreamBatchSampler(Huaxi_labeled_idxs, Huaxi_unlabeled_idxs, args.bs, args.bs//2)
        train_dataloader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers = args.nw, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(val_data,shuffle = False,batch_sampler=None,num_workers = args.nw,batch_size=args.bs,worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(test_data,shuffle = False,batch_sampler=None,num_workers = args.nw,batch_size=args.bs,worker_init_fn=worker_init_fn,drop_last=False)
    
    net = get_network(args)
    net.to(args.device)
    
    dice_loss = get_loss(args)
    args.loss = 'CELoss'
    ce_loss = get_loss(args)
    cls_ce_loss = torch.nn.CrossEntropyLoss()
        
    dsc_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    hd_metric = HausdorffDistanceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, percentile=95, get_not_nans=True)
    iou_metric = MeanIoU(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    pg = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=eval(args.ms), gamma=0.1) 
    model_save_name = 'cycle' + '_' + '_'.join([args.network, args.dataset, str(args.added_CAMUS), str(args.size), args.chamber, 'tpral_' + str(args.temporal_interval), 'uncer_' + str(args.forward_uncer),  'hist_'+ str(args.HistMatch), 'cls_' + str(args.classify), 'prior_' + str(args.prior), str(args.sample_ratio), 'pretrain_' + str(args.pretrain is not None), 'ws_'+str(args.ws), datetime.date.today().strftime('%y%m%d')])
    lowerest_loss = float("inf")
    epoch_resume = 0
    update_lowloss = 0
   
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['state_dict1'], strict=False)
            
    if args.distributed:
        if args.syncBN:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(args.device)       
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu],find_unused_parameters=True)
 
        
    if args.test:
        print('Huaxi 2d')
        print(args.pretrain, args.chamber)
        test_loss, res_list = val(args, 0,test_dataloader,net,dsc_metric,ce_loss, dice_loss,args.device, None)
        #print(np.array(res_list)[:,1])
        mean_lv = np.mean(np.array(res_list)[1:,2].astype(float))
        mean_lyo = np.mean(np.array(res_list)[1:,6].astype(float))
        csv_save_name = "%s/results_%s_%s_%s.csv"%('/'.join(args.pretrain.split('/')[:-1]), args.chamber, str(mean_lv)[:6], str(mean_lyo)[:6])
        write2csv(res_list, csv_save_name)
        return
    log_path = args.model_dir + '/' + "%s/"%(model_save_name)
    if args.rank == 0:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    if args.distributed:
        torch.distributed.barrier()
    logger = get_logger(log_path + '/logger.log')
    if args.rank == 0:
        logger.info(args)
        logger.info('the length of train_data {} and val_data {}'.format(str(len(train_data)), str(len(val_data))))
        logger.info('the length of train_dataloder {} and val_dataloader {}'.format(str(len(train_dataloader)), str(len(val_dataloader))))
    
 
    if args.rank == 0:
        writer1 = SummaryWriter('tensorboard/'+model_save_name)

    for e in range(args.initial_epoch, args.epoch):
        time1 = time.time()
        train_loss = train(e,train_dataloader,net,dsc_metric,ce_loss, dice_loss, optimizer,args.device,logger)
        time2 = time.time() - time1
        scheduler.step()
        if args.rank == 0:
            writer1.add_scalar('train_loss', train_loss, global_step=e)

        if args.distributed:
            save_model = net.module
        else:
            save_model = net
        if e%2 == 0:
            val_loss, _ = val(args, e,val_dataloader,net,dsc_metric,ce_loss, dice_loss,args.device, logger)
            if args.rank == 0:
                writer1.add_scalar('val_loss', val_loss, global_step=e)

                if val_loss < lowerest_loss:
                    update_lowloss = 0
                    lowerest_loss = val_loss
                    save_checkpoint({
                        'epoch': e,
                        'state_dict': save_model.state_dict(),
                        'opt_dict': optimizer.state_dict(),
                        'scheduler_dict': scheduler.state_dict(),
                        'lowerest_loss': lowerest_loss
                    }, model_save_name, 'lowerest_loss' + '.pth.tar')
                else:
                    update_lowloss += 1
                
        if args.rank == 0:
            if e % args.model_save_freq ==0:
                   save_checkpoint({
                        'epoch': e,
                        'state_dict1': save_model.state_dict(),
                        'opt_dict': optimizer.state_dict(),
                        'scheduler_dict': scheduler.state_dict(),
                        'lowerest_loss': lowerest_loss}, 
                        model_save_name,
                        'checkpoint_' + str(e) + '.pth.tar')
                    
                
def train(ep, train_dataloader, model, metric, ce_loss, dice_loss, optimizer, device, logger):
    flow_loss = torch.nn.L1Loss()
    mean_loss = torch.zeros(1).to(device)
    model.train()
    run_acc1 = AverageMeter()
    iter_num = ep * len(train_dataloader)
    for step, data in enumerate(train_dataloader):
        
        iter_num = iter_num + 1
        file_path, num, frame, frame_noise, frame_uncertainty, mask, gls, chamber, beita_lv, beita_myo = data
        if args.temporal_interval != 0:
            bs, slices, c, h, w = frame.shape
            frame = torch.cat([frame[:,:slices//2],frame[:,slices//2:]], 0).squeeze(1)
            frame_noise = torch.cat([frame_noise[:,:slices//2],frame_noise[:,slices//2:]], 0).squeeze(1)
            mask = torch.cat([mask[:,:slices//2],mask[:,slices//2:]], 0).squeeze(1)
        frame = frame.type(torch.FloatTensor)
        frame = frame.to(device)    
        frame_noise = frame_noise.type(torch.FloatTensor)
        frame_noise = frame_noise.to(device)
        mask = mask.type(torch.LongTensor)
        mask = mask.to(device)
        for i in range(mask.shape[0]//4):
            if torch.sum(mask[mask.shape[0]//2 + i]) == 0:
                print('error mask')
        if args.num_classes == 3:
            mask[mask>2] = 0
        chamber = chamber.type(torch.LongTensor)
        chamber = chamber.to(device)
        #chamber = one_hot(chamber.unsqueeze(-1), 4)   
        if args.forward_uncer != 0:
            frame_uncertainty = torch.cat([frame_uncertainty[:,:slices//2],frame_uncertainty[:,slices//2:]], 0).squeeze(1)
            frame_uncertainty = frame_uncertainty.type(torch.FloatTensor)
            frame_uncertainty = frame_uncertainty.to(device)

        flow_list, outputs1, warped1_reverse_list, warped_logits_reverse_list, chamber_pred1 = model(frame)
        warped1 = warped1_reverse_list[-1]
        warped_logits = warped_logits_reverse_list[-1]
        
        outputs_soft1 = torch.softmax(outputs1, dim=1)
        mask_onehot = one_hot(mask, num_classes=args.num_classes)
        pseudo_outputs1 = torch.argmax(
            outputs_soft1.detach(), dim=1, keepdim=False)
        pseudo_warped_logits1 = torch.argmax(
            warped_logits.detach(), dim=1, keepdim=False)

        start = mask_onehot.shape[0]//2
        if args.prior:
            semi = args.bs//2
            loss1 = 1 * ce_loss(outputs1[-start:][:semi], mask_onehot[start:][:semi]) + 1*dice_loss(outputs1[-start:][:semi], mask_onehot[start:][:semi])
            run_acc1,_ = update_metric(metric, run_acc1, pseudo_outputs1[-start:][:semi], mask_onehot[start:][:semi],args.num_classes)
        else:
            loss1 = 1 * ce_loss(outputs1[-start:], mask_onehot[start:]) + 1*dice_loss(outputs1[-start:], mask_onehot[start:])
            run_acc1,_ = update_metric(metric, run_acc1, pseudo_outputs1[-start:], mask_onehot[start:],args.num_classes)

        if args.classify:
            chamber_loss = ce_loss(chamber_pred1[start:], chamber)
            loss = 2 * loss1 + 0.5 * chamber_loss
        else:
            loss = 2 * loss1
            

        weight = 1 - ((ep+1)/args.epoch)**2
        loss_warp_img, loss_warp_label, loss_warp_unlabel = multi_scale_flow(warped1_reverse_list, warped_logits_reverse_list, frame, mask_onehot, outputs1, bs, ce_loss, dice_loss, flow_loss, reverse=True, prior=args.prior)
        loss = loss + (2*loss_warp_label +  2*loss_warp_unlabel + 40 * loss_warp_img) * weight * 0.1
   
        time4 = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()      
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
    if args.rank==0:
        if args.num_classes == 4:
            logger.info('{} ep:{:0>3d} mean_loss:{:.4f} Dice lv {:.4f} Dice lvo {:.4f} Dice la {:.4f} Dice bg {:.4f} lr:{:.4f} '.format('Train', ep, mean_loss.item(), run_acc1.avg[1], run_acc1.avg[2], run_acc1.avg[3], run_acc1.avg[0], optimizer.param_groups[0]['lr']))
        else:
            logger.info('{} ep:{:0>3d} mean_loss:{:.4f} Dice lv {:.4f} Dice lvo {:.4f} Dice bg {:.4f} lr:{:.4f} '.format('Train', ep, mean_loss.item(), run_acc1.avg[1], run_acc1.avg[2], run_acc1.avg[0], optimizer.param_groups[0]['lr']))
        torch.cuda.empty_cache()
    return mean_loss


def val(args, ep, train_dataloader, model, metric, ce_loss, dice_loss, device, logger):
    mean_loss = torch.zeros(1).to(device)
    model.eval()
    run_acc1 = AverageMeter()
    res_list = [['file_path',  'chamber', 'dsc_lv', 'jc_lv', 'hd_lv', 'asd_lv', 'dsc_myo', 'jc_myo', 'hd_myo', 'asd_myo', 'chamber_pred']]
    
    for step, data in enumerate(train_dataloader):
        file_path, num, frame, frame_noise, frame_uncertainty, mask, gls, chamber, beita_lv, beita_myo = data
        if args.temporal_interval != 0:
            bs, slices, c, h, w = frame.shape
            frame = torch.cat([frame[:,:slices//2],frame[:,slices//2:]], 0).squeeze(1)
            frame_noise = torch.cat([frame_noise[:,:slices//2],frame_noise[:,slices//2:]], 0).squeeze(1)
            mask = torch.cat([mask[:,:slices//2],mask[:,slices//2:]], 0).squeeze(1)
        frame = frame.type(torch.FloatTensor)
        frame = frame.to(device)    
        frame_noise = frame_noise.type(torch.FloatTensor)
        frame_noise = frame_noise.to(device)
        mask = mask.type(torch.LongTensor)
        mask = mask.to(device)
        if args.num_classes == 3:
            mask[mask>2] = 0
        chamber = chamber.type(torch.LongTensor)
        chamber = chamber.to(device)
        
        flow_list, outputs1, warped1_reverse_list, warped_logits_reverse_list, chamber_pred1 = model(frame)
        warped1 = warped1_reverse_list[-1]
        warped_logits = warped_logits_reverse_list[-1]
        outputs_soft1 = torch.softmax(outputs1, dim=1)
        mask_onehot = one_hot(mask, num_classes=args.num_classes)
        
        pseudo_outputs1 = torch.argmax(
            outputs_soft1.detach(), dim=1, keepdim=False)
        
        start = mask.shape[0]//2
        loss = 1 * ce_loss(outputs1[-start:], mask_onehot[start:]) + 1*dice_loss(outputs1[-start:], mask_onehot[start:])
        run_acc1, iter_dice = update_metric(metric, run_acc1, pseudo_outputs1[-start:], mask_onehot[start:],args.num_classes)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        
        if args.test:
            values = metrics.calculate_metric_percase(pseudo_outputs1[-start:].cpu().detach().numpy().squeeze(), mask.cpu().detach().numpy()[-start:].squeeze())
            info_list = [file_path[0].split('/')[-1], chamber.cpu().numpy()[0]]
            another_values = calculate_metric_percase(pseudo_outputs1[-start:].cpu().detach().numpy().squeeze()[np.newaxis].astype('uint8'), mask.cpu().detach().numpy()[start:].squeeze()[np.newaxis].astype('uint8'))
            info_list.extend(another_values)
            info_list.append(chamber_pred1[start:][0].cpu().detach().numpy())
            res_list.append(info_list)
            print(info_list)

    if args.rank==0:
        if logger is not None:
            if args.num_classes == 4:
                logger.info('{} ep:{:0>3d} mean_loss:{:.4f} Dice lv {:.4f} Dice lvo {:.4f} Dice la {:.4f} Dice bg {:.4f}'.format('Val   ', ep, mean_loss.item(), run_acc1.avg[1], run_acc1.avg[2], run_acc1.avg[3], run_acc1.avg[0],))
            else:
                logger.info('{} ep:{:0>3d} mean_loss:{:.4f} Dice lv {:.4f} Dice lvo {:.4f} Dice bg {:.4f}'.format('Val   ', ep, mean_loss.item(), run_acc1.avg[1], run_acc1.avg[2], run_acc1.avg[0],))
        else:
            print('{} ep:{:0>3d} mean_loss:{:.4f} Dice lv {:.4f} Dice lvo {:.4f} Dice bg {:.4f}'.format('Val   ', ep, mean_loss.item(), run_acc1.avg[1], run_acc1.avg[2], run_acc1.avg[0]))
        torch.cuda.empty_cache()
        
    return mean_loss, res_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',default=None )
    parser.add_argument('--task',default='regression',help='seg or regression or ssl')
    parser.add_argument("--update_folder",default='/mnt/workspace/jiaorushi/GLS/data/segment_npy_denoise/')
    parser.add_argument('--dataset',default='huaxi',help='huaxi or open')
    parser.add_argument('--phase',default='train',help='train or pretrain')
    parser.add_argument('--network',default='EchoNet',help='EchoNet or Swin_UNetr or Swin_SSL' )
    parser.add_argument('--model_name',default='r2plus1d_18' )
    parser.add_argument('--classify',action='store_true')
    parser.add_argument('--prior',action='store_true')
    parser.add_argument('--GLS_prior',action='store_true')
    parser.add_argument('--HistMatch',action='store_true')
    parser.add_argument('--loss',default='MSELoss' )
    parser.add_argument('--weighted_dice',action='store_true')
    parser.add_argument('--chamber',default='all' )
    parser.add_argument('--num_classes',default=4, type=int)
    parser.add_argument('--info_csv',default='')
    parser.add_argument('--csv_save',default=None,help='csv save path for the testing results' )
    parser.add_argument("--epoch", default=100,type=int)
    parser.add_argument("--initial_epoch", default=0,type=int)
    parser.add_argument("--pretrain", default=None,type=str)
    parser.add_argument("--continue_train", default=None,type=str)
    parser.add_argument("--lr", default=1e-3,type=float)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument("--bs", default=8,type=int,help='batch_size')
    parser.add_argument("--ms", default = '[30, 50]')
    parser.add_argument("--test",action='store_true')
    parser.add_argument("--size", default=256,type=int)
    parser.add_argument("--slices_per_cycle", default=32,type=int)
    parser.add_argument("--print_freq", default = 30, type=int)
    parser.add_argument('--model_dir',default='save_models/')
    parser.add_argument("--model_save_freq", default=1,type=int)
    parser.add_argument('--consistency', type=float,
                    default=0.3, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                    default=250.0, help='consistency_rampup')
    parser.add_argument('--sample_ratio', type=float,
                    default=1, help='training samples ratio')
    
    # settings for distribution
    parser.add_argument("--seed", default=1,type=int)
    parser.add_argument("--nw", default=4,type=int)
    parser.add_argument("--local-rank", default=0,type=int)
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--rank", default=0)
    parser.add_argument('--distributed',default=False,type=bool)
    parser.add_argument("--device", default='cuda')
    parser.add_argument('--syncBN',type=bool,default=True)
    parser.add_argument('--world_size',default=2,type=int,help='number of distributed processes')
    parser.add_argument('--dist-url',default='env://',help='url uses to set up distributed training')
    
    # settings for Semi
    parser.add_argument("--iterative_refine",action='store_true')
    parser.add_argument("--added_CAMUS",action='store_true')
    parser.add_argument("--added_EchoDynamic",action='store_true')
    parser.add_argument("--forward_uncer",default=None, type=int)
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    
 
    # settings for temporal coherence
    parser.add_argument("--temporal_interval", default=3, type=int)

    parser.add_argument("--save_folder", default=None, help='res_segmentations/CAMUS_Huaxi_a3c_TransUnet_pretrain/')
    parser.add_argument("--ws", default=7, type=int)
        
    args = parser.parse_args()
    
    if torch.cuda.is_available() is False:
        raise EnvironmentError('not find gpu')
    
    main(args)

        
    




            






















