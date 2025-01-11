import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp
from .refine import *

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
                                  conv(in_planes*2 // (4*4) + in_else, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  

    def forward(self, motion_feature, x, flow): # /16 /8 /4
        #print('1', motion_feature.shape, x.shape)
        motion_feature = self.upsample(motion_feature) #/4 /2 /1
        #print('1_', motion_feature.shape, x.shape)
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
            #print('2', self.scale, x.shape)
        if flow != None:
            if self.scale != 4:
                flow = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
            x = torch.cat((x, flow), 1)
        #print('3', motion_feature.shape, x.shape)
        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = self.scale // 4, mode="bilinear", align_corners=False)
            flow = x[:, :4] * (self.scale // 4)
        else:
            flow = x[:, :4]
        mask = x[:, 4:5]
        return flow, mask

    
class MultiScaleFlow(nn.Module):
    def __init__(self, backbone, **kargs):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = len(kargs['hidden_dims'])
        self.feature_bone = backbone
        self.block = nn.ModuleList([Head( kargs['motion_dims'][-1-i] * kargs['depths'][-1-i] + kargs['embed_dims'][-1-i], 
                            kargs['scales'][-1-i], 
                            kargs['hidden_dims'][-1-i],
                            6 if i==0 else 17) 
                            for i in range(self.flow_num_stage)])
        self.unet = Unet(kargs['c'] * 2)

    def warp_features(self, xs, flow):
        y0 = []
        y1 = []
        B = xs[0].size(0) // 2
        for x in xs:
            y0.append(warp(x[:B], flow[:, 0:2]))
            y1.append(warp(x[B:], flow[:, 2:4]))
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        return y0, y1

    def calculate_flow(self, imgs, af=None, mf=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None
        # appearence_features & motion_features
        if (af is None) or (mf is None):
            af, mf = self.feature_bone(img0, img1)
        for i in range(self.flow_num_stage):
            if flow != None:
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                flow_, mask_ = self.block[i](
                    torch.cat([mf[-1-i][:B],mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                    flow
                    )
                flow = flow + flow_
                mask = mask + mask_
            else:
                flow, mask = self.block[i](
                    torch.cat([mf[-1-i][:B],mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1), 1),
                    None
                    )

        return flow, mask

    def coraseWarp_and_Refine(self, imgs, af, flow, mask):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        mask_ = torch.sigmoid(mask)
        merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
        pred = torch.clamp(merged + res, 0, 1)
        return pred


    # Actually consist of 'calculate_flow' and 'coraseWarp_and_Refine'
    def forward(self, x_in):
        x = torch.cat((x_in[:x_in.size(0)//2],x_in[x_in.size(0)//2:]), axis=1)
        img0, img1 = x[:, :3], x[:, 3:6]
        B = x.size(0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        # appearence_features & motion_features
        af, mf = self.feature_bone(img0, img1)
        for i in range(self.flow_num_stage):
            #print(mf[-1-i][:B].shape, af[-1-i][:B].shape)
            #t = torch.full(mf[-1-i][:B].shape, timestep, dtype=torch.float).to(x.device)
            if flow != None:
                flow_d, mask_d = self.block[i]( torch.cat([mf[-1-i][:B], mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1), 
                                                torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = self.block[i](torch.cat([mf[-1-i][:B], mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1), 
                                            torch.cat((img0, img1), 1), None)
            #mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            #merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))
        
        c0, c1 = self.warp_features(af, flow)
        seg = self.unet(img0, img1, flow[:,:4], c0, c1)
        warped = torch.cat((warped_img1,warped_img0),axis=0)
        return flow_list, seg, warped
    
    def forward_4flow(self, x_in):
        x = torch.cat((x_in[:x_in.size(0)//2],x_in[x_in.size(0)//2:]), axis=1)
        img0, img1 = x[:, :3], x[:, 3:6]
        B = x.size(0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        # appearence_features & motion_features
        af, mf = self.feature_bone(img0, img1)
        for i in range(self.flow_num_stage):
            #print(mf[-1-i][:B].shape, af[-1-i][:B].shape)
            #t = torch.full(mf[-1-i][:B].shape, timestep, dtype=torch.float).to(x.device)
            if flow != None:
                flow_d, mask_d = self.block[i]( torch.cat([mf[-1-i][:B], mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1), 
                                                torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = self.block[i](torch.cat([mf[-1-i][:B], mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1), 
                                            torch.cat((img0, img1), 1), None)
            #mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            #merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))
        warped = torch.cat((warped_img1,warped_img0),axis=0)
        return flow_list, warped