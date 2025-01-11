import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F

from .EMA_VFI.warplayer import warp
from .EMA_VFI.refine import *



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


# class Head(nn.Module):
#     def __init__(self, in_planes, scale, c, in_else=17):
#         super(Head, self).__init__()
#         self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
#         self.scale = scale
#         self.conv = nn.Sequential(
#                                   conv(in_planes // (4*4) + in_else, c),
#                                   conv(c, c),
#                                   conv(c, 2),
#                                   )  

#     def forward(self, motion_feature, x, flow): # /16 /8 /4
#         motion_feature = self.upsample(motion_feature) #/4 /2 /1
#         print('1', motion_feature.shape)
#         if self.scale != 4:
#             print('2', x.shape, self.scale)
#             x = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
#         if flow != None:
#             if self.scale != 4:
#                 flow = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
#             print('flow', flow.shape)
#             x = torch.cat((x, flow), 1)
#         print('3', motion_feature.shape, x.shape)
#         x = self.conv(torch.cat([motion_feature, x], 1))
#         print('4', x.shape)
#         if self.scale != 4:
#             x = F.interpolate(x, scale_factor = self.scale // 4, mode="bilinear", align_corners=False)
#             flow = x * (self.scale // 4)
#         else:
#             flow = x
#         return flow


class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
                                  conv(in_planes // (2*2) + in_else, c),
                                  conv(c, c),
                                  conv(c, 2),
                                  )  

    def forward(self, motion_feature, x, flow): # /16 /8 /4
        motion_feature = self.upsample(motion_feature) #/4 /2 /1
        if self.scale != 2:
            x = F.interpolate(x, scale_factor = 2. / self.scale, mode="bilinear", align_corners=False)
        if flow != None:
            if self.scale != 2:
                flow = F.interpolate(flow, scale_factor = 2. / self.scale, mode="bilinear", align_corners=False) * 2. / self.scale
            x = torch.cat((x, flow), 1)
        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.scale != 2:
            x = F.interpolate(x, scale_factor = self.scale // 2, mode="bilinear", align_corners=False)
            flow = x * (self.scale // 2)
        else:
            flow = x
        return flow

    
class MultiScaleFlow(nn.Module):
    def __init__(self, embed_dims=[512, 256, 64], motion_dims=[32, 32, 32], num_heads=[2, 4, 4], 
                 mlp_ratios=[4, 4, 4], depths=[2, 2, 2], window_sizes=[7, 7, 7], hidden_dims=[64, 64, 64], scales=[8, 4, 2]):
    # def __init__(self, embed_dims=[512, 256], motion_dims=[32, 32], num_heads=[2, 4], 
    #              mlp_ratios=[4, 4], depths=[2, 2], window_sizes=[7, 7], hidden_dims=[64, 64], scales=[8, 4]):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = len(hidden_dims)
        self.block = nn.ModuleList([Head(motion_dims[i] * depths[i] + embed_dims[i], 
                            scales[i], hidden_dims[i],
                            6 if i==0 else 11) 
                            for i in range(self.flow_num_stage)])

    def calculate_flow(self, imgs, af=None, mf=None):
        B = imgs.size(0)//2
        img0, img1 = imgs[:B], imgs[B:]
        imgs_reverse = torch.cat([img1, img0])
        flow, mask = None, None
        # appearence_features & motion_features
        if (af is None) or (mf is None):
            af, mf = self.feature_bone(img0, img1)
        for i in range(self.flow_num_stage):
            if flow != None:
                warped_img = warp(imgs_reverse, flow)
                #warped_img = warp(imgs, flow)
                flow_  = self.block[i](
                    torch.cat([mf[-1-i], af[-1-i]],1),
                    torch.cat((imgs, imgs_reverse, warped_img), 1),
                    flow
                    )
                flow = flow + flow_
            else:
                flow = self.block[i](
                    torch.cat([mf[-1-i], af[-1-i]],1),
                    torch.cat((imgs, imgs_reverse), 1),
                    None
                    )

        return flow


    # Actually consist of 'calculate_flow' 
    def forward(self, x, logits, af, mf):
        B = x.size(0)//2
        img0, img1 = x[:B], x[B:]
        imgs_reverse = torch.cat([img1, img0])
        logits_reverse = torch.cat([logits[B:], logits[:B]])
        flow_list = []
        merged = []
        mask_list = []
        flow = None
        merged_img = []
        merged_logits = []

        for i in range(self.flow_num_stage):
            
            if flow != None:
                # 尺寸最小的先进
                flow_d = self.block[i](torch.cat([mf[i], af[i]],1), 
                                                torch.cat((x, imgs_reverse, warped_img), 1), flow)
                flow = flow + flow_d
            else:
                #print(mf[i].shape, af[i].shape, x.shape, imgs_reverse.shape)
                flow = self.block[i](torch.cat([mf[i], af[i]],1), 
                                                torch.cat((x, imgs_reverse), 1), None)
            flow_list.append(flow)
            # warped_img = warp(imgs_reverse, flow)
            # warped_logits = warp(logits_reverse, flow)
            warped_img = warp(x, flow)
            warped_logits = warp(logits, flow)
            merged_img.append(warped_img)
            merged_logits.append(warped_logits)
        return merged_img, merged_logits, flow_list
        #return warped_img, warped_logits, flow_list

