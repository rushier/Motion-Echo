import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from topologylayer.nn import LevelSetLayer2D, TopKBarcodeLengths
from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance
from abc import ABC

def multi_scale_flow(merged_imgs, merged_logits, imgs_gt, labels_onehot, logits_pred, bs, ce_loss, dice_loss, flow_loss, reverse=True, prior=False):
    if reverse:
        merged_imgs = [torch.cat([x[bs:], x[:bs]]) for x in merged_imgs]
        merged_logits = [torch.cat([x[bs:], x[:bs]]) for x in merged_logits]
    #print(bs, merged_imgs[0].shape)
    loss_warp_img = 0
    loss_warp_label = 0
    loss_warp_unlabel = 0
    for i in range(len(merged_logits)):
    # if 1:
    #     i = -1
        # warp img loss
        loss_warp_img = loss_warp_img + flow_loss(merged_imgs[i], imgs_gt)
        # warp label loss
        if prior:
            loss_warp_label = loss_warp_label + ce_loss(merged_logits[i][bs:][:bs//2], labels_onehot[bs:][:bs//2]) + dice_loss(merged_logits[i][bs:][:bs//2], labels_onehot[bs:][:bs//2])
        else:
            loss_warp_label = loss_warp_label + ce_loss(merged_logits[i][bs:], labels_onehot[bs:]) + dice_loss(merged_logits[i][bs:], labels_onehot[bs:])
        # warp unlabel loss
        loss_warp_unlabel = loss_warp_unlabel + flow_loss(merged_logits[i][:bs], logits_pred[:bs])

    return loss_warp_img, loss_warp_label, loss_warp_unlabel

def bmc_loss(pred, target, noise_var):
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var)
    return loss

def huber_loss(x):
    d_x = x[:,:,:,1:]-x[:,:,:,:-1]
    d_y = x[:,:,1:,:]-x[:,:,:-1,:]
    err = (d_x**2).sum()+(d_y**2).sum()
    err /= 20.0
    tv_err = T.sqrt(0.01+err)
    return tv_err


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)
    
    
class SSLLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.order_loss = torch.nn.MultiLabelSoftMarginLoss(reduction='mean').cuda() # 要过sigmoid
        self.order_loss = torch.nn.BCEWithLogitsLoss().cuda()
        #self.order_loss = torch.nn.L1Loss().cuda()
        self.recon_loss = torch.nn.MSELoss(reduction='none').cuda()
        #self.contrast_loss = Contrast(args, batch_size).cuda()
        self.alpha1 = 1.0
        self.alpha2 = 1.0

    def __call__(self, output_order, target_order, output_recons, target_recons, recons_mask):
        order_loss = self.alpha1 * self.order_loss(output_order, target_order)
        #recon_loss = 0
        recon_loss = torch.sum((output_recons-target_recons) **2 * recons_mask)/torch.sum(recons_mask)
        total_loss = order_loss + recon_loss
        return order_loss, recon_loss
    
class weighted_dice(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, pred, label, weight=None):
        sumdice = 0
        smooth = 1e-6
        pred = torch.softmax(pred, dim=1)
        if weight is None:
            weight = 1
        for i in range(0, label.size(1)):
            pred_bin = (pred[:,i:i+1])*weight
            label_bin = (label[:,i:i+1])*weight
            pred_bin = pred_bin.contiguous().view(pred_bin.shape[0], -1)
            label_bin = label_bin.contiguous().view(label_bin.shape[0], -1)
            intersection = (pred_bin * label_bin).sum()
            dice = (2. * intersection + smooth) / (pred_bin.sum() + label_bin.sum() + smooth)
            sumdice += dice
        return 1-sumdice/label.size(1)
    
    
class pixel_weighted_ce(torch.nn.Module):
    def __init__(self, class_weight=None):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(size_average=None, reduction='none', weight=class_weight)
    def __call__(self, pred, label, weight=None):
        loss = self.loss(pred, label)
        if weight is not None:
            loss = loss * weight[:,0]
            loss = torch.sum(loss)/torch.sum(weight)
        else:
            loss = torch.mean(loss)
        return loss
    
def cal_uncertainty(preds, bs, times):
    preds = F.softmax(preds, dim=1)
    preds = preds.reshape(bs, times, preds.shape[1], 224, 224)
    preds = torch.mean(preds, dim=1)  #(batch, 2, 112,112,80)
    uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)
    return uncertainty

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

# class topo_loss(torch.nn.Module):
#     def __init__(self, max_k=20):
#         super().__init__()
#         self.dgminfo = LevelSetLayer2D(size=(224,224), sublevel=False, maxdim=1)
#         self.max_k = max_k
#     def one_class_loss(self, preds, beitas):
#         loss_all = 0
#         for pred, beita in zip(preds, beitas):
#             a = self.dgminfo(pred)
#             L0 = (TopKBarcodeLengths(dim=0, k=self.max_k)(a)**2).sum()
#             dim_1_sq_bars = TopKBarcodeLengths(dim=1, k=self.max_k)(a)**2
#             bar_signs = torch.ones(self.max_k).cuda()
#             bar_signs[:int(beita[1].item())] = -1
#             L1 = (dim_1_sq_bars * bar_signs).sum()
#             loss = L0+L1
#             #print(pred.shape, beita, L0, L1)
#             loss_all += loss
#         return loss_all/len(preds)
#     def __call__(self, preds, beita_lv, beita_myo):
#         pred_lv = preds[:,1:2]
#         pred_myo = preds[:,2:3]
#         loss_lv = self.one_class_loss(pred_lv, beita_lv)
#         loss_myo = self.one_class_loss(pred_myo, beita_myo)
#         #print(loss_lv, loss_myo)
#         loss = loss_lv + loss_myo
#         return loss/2

class topo_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cubical_complex = CubicalComplex(dim=2)
        self.wasserstein = WassersteinDistance(q=1)
        
    def cal_loss(self, pers_pred, pers_gt):
        loss_all = 0
        bs = len(pers_pred)
        for i in range(bs):
            for pers1, pers2 in zip(pers_pred[i], pers_gt[i]):
                loss = self.wasserstein(pers1, pers2)
                loss_all += loss
        return loss_all
        
    def forward(self, pred, gt):
        object_pred = pred[:,1:]
        object_gt = gt[:,1:]
        pers_pred = self.cubical_complex(object_pred)
        pers_gt = self.cubical_complex(object_gt)
        L = self.cal_loss(pers_pred, pers_gt)
        return L
    
    
class PixelContrastLoss(torch.nn.Module, ABC):
    def __init__(self, args):
        super(PixelContrastLoss, self).__init__()

        self.temperature = args.temperature
        self.base_temperature = args.base_temperature

        self.ignore_label = [0]
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

        self.max_samples = args.max_samples # 所有类一共多少像素点
        self.max_views = args.max_views # 每张样本每类包含多少个像素点

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x not in self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]
            # print('this_y', this_y.shape) # [50176] 224*224
            # print([(this_y == x).nonzero().shape for x in this_classes]) #[torch.Size([35962, 1]), torch.Size([6227, 1]), torch.Size([7987, 1])]
            # print(this_classes) #[tensor(0, device='cuda:3'), tensor(1, device='cuda:3'), tensor(2, device='cuda:3')]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
        

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1
        # return the hard pixels and simple pixels in every class of every sample # considering using uncertenty to choose the samples?
        return X_, y_

    def _contrastive(self, feats_, labels_):
        # print('feats_', feats_.shape, 'labels_', labels_.shape)
        # feats_ torch.Size([6, 100, 256]) labels_ torch.Size([6])
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1) # [6,1]
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()
        # print(mask.shape) [6,6]
        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)
        #print('contrast_feature', contrast_feature.shape)
        # contrast_feature torch.Size([600, 256])
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        #print('anchor_dot_contrast', anchor_dot_contrast)
        # anchor_dot_contrast torch.Size([600, 600])
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        #print('logits_max', logits_max)
        # ogits_max torch.Size([600, 1])
        logits = anchor_dot_contrast - logits_max.detach()
        #print('logits', logits)

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        #print('mask', mask.shape)
        # torch.Size([600, 600])
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                   0)
        #print('1', mask)
        # 不计算对角线，去掉自己
        mask = mask * logits_mask
        #print('2', mask)   
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)
        #print('neg_logits', neg_logits) # torch.Size([600, 1])
        exp_logits = torch.exp(logits) #  torch.Size([600, 600])
        #print('exp_logits', exp_logits)
        #print('add', exp_logits + neg_logits)
        log_prob = logits - torch.log(exp_logits + neg_logits)
        #print('log_prob', log_prob.shape) # torch.Size([600, 600])
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        #print('1feats', feats.shape, 'labels', labels.shape, 'predict', predict.shape)
        #feats torch.Size([4, 256, 224, 224]) labels torch.Size([4, 224, 224]) predict torch.Size([4, 224, 224])
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        #print('2feats', feats.shape, 'labels', labels.shape, 'predict', predict.shape)
        #feats torch.Size([4, 50176, 256]) labels torch.Size([4, 50176]) predict torch.Size([4, 50176])2feats
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss
    
    
class PixelContrastLoss_mem(torch.nn.Module, ABC):
    def __init__(self, args):
        super(PixelContrastLoss_mem, self).__init__()

        self.temperature = args.temperature
        self.base_temperature = args.base_temperature

        self.ignore_label = [0]
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

        self.max_samples = args.max_samples
        self.max_views = args.max_views


    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x not in self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            if ii == 0: continue
            this_q = Q[ii, :cache_size, :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)

        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_, queue=queue)
        return loss
        
        
        
    
        
    



    
    
    
    
    