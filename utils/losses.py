import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
# from topologylayer.nn import LevelSetLayer2D, TopKBarcodeLengths
# from torch_topological.nn import CubicalComplex
# from torch_topological.nn import WassersteinDistance
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


    
    
        
        
        
    
        
    



    
    
    
    
    