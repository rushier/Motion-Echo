import torch
import torchvision
import torch.nn as nn
#from torchvision.models.video import swin_transformer
# import sys
# sys.path.append('/home/pai/lib/python3.9/site-packages/torchvision/models/video/')
# import swin_transformer

from .Swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class SSLHead(nn.Module):
    def __init__(self, args, upsample="vae", dim=768):
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )
        self.decoder_dims = [dim//8, dim//4, dim//2, dim]
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, args.in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, args.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.decoder1 = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
                
            self.decoder2 = nn.Sequential(
                nn.Conv3d(dim, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            
            self.decoder3 = nn.Sequential(
                nn.Conv3d(dim // 2, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            
            self.decoder4 = nn.Sequential(
                nn.Conv3d(dim // 4, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            
            self.decoder5 = nn.Sequential(
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, args.in_channels, kernel_size=1, stride=1))

    def forward(self, x):
        x0_out, x1_out, x2_out, x3_out, x4_out= self.swinViT(x.contiguous())
        #print(x0_out.shape, x1_out.shape, x2_out.shape, x3_out.shape, x4_out.shape)
        _, c, h, w, d = x4_out.shape
        #print(x_out.shape)
        x4_pool = self.pool(x4_out)
        x4_pool = x4_pool.view((x4_pool.shape[0],-1))
        # x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        # print('1', x4_reshape.shape)
        # x4_reshape = x4_reshape.transpose(1, 2)
        # print('2', x4_reshape.shape)
        #x_rot = self.rotation_pre()
        x_rot = self.rotation_head(x4_pool)
       
        #x_rec = x_out.flatten(start_dim=2, end_dim=4)
        #x_rec = x_rec.view(-1, c, h, w, d)
        x4_up = self.decoder1(x4_out)
        x4_cat = torch.cat((x4_up, x3_out), dim=1)
        #print(x4_cat.shape)
        x3_up = self.decoder2(x4_cat)
        x3_cat = torch.cat((x3_up, x2_out), dim=1)
        #print(x3_cat.shape)
        x2_up = self.decoder3(x3_cat)
        x2_cat = torch.cat((x2_up, x1_out), dim=1)
        #print(x2_cat.shape)
        x1_up = self.decoder4(x2_cat)
        #x1_cat = torch.cat((x1_up, x0_out), dim=1)
        #print(x1_cat.shape)
        x_rec = self.decoder5(x1_up)
        #x_rec = self.conv(x4_out)
        return x_rot, x_rec




# class SSLHead(nn.Module):
#     def __init__(self, args, upsample="vae", dim=768):
#         super(SSLHead, self).__init__()
#         patch_size = ensure_tuple_rep(2, args.spatial_dims)
#         window_size = ensure_tuple_rep(7, args.spatial_dims)
#         self.swinViT = SwinViT(
#             in_chans=args.in_channels,
#             embed_dim=args.feature_size,
#             window_size=window_size,
#             patch_size=patch_size,
#             depths=[2, 2, 2, 2],
#             num_heads=[3, 6, 12, 24],
#             mlp_ratio=4.0,
#             qkv_bias=True,
#             drop_rate=0.0,
#             attn_drop_rate=0.0,
#             drop_path_rate=args.dropout_path_rate,
#             norm_layer=torch.nn.LayerNorm,
#             use_checkpoint=args.use_checkpoint,
#             spatial_dims=args.spatial_dims,
#         )
#         self.pool3d = nn.MaxPool3d((1,4,4), stride=1)  # max pooling
#         self.order_pre = nn.Identity()
#         self.order_head = nn.Sequential(nn.BatchNorm1d(dim),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(dim,1))

#         if upsample == "large_kernel_deconv":
#             self.conv = nn.ConvTranspose3d(dim, args.in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
#         elif upsample == "deconv":
#             self.conv = nn.Sequential(
#                 nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
#                 nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
#                 nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
#                 nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
#                 nn.ConvTranspose3d(dim // 16, args.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
#             )
#         elif upsample == "vae":
#             self.conv = nn.Sequential(
#                 nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
#                 nn.InstanceNorm3d(dim // 2),
#                 nn.LeakyReLU(),
#                 nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
#                 nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
#                 nn.InstanceNorm3d(dim // 4),
#                 nn.LeakyReLU(),
#                 nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
#                 nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
#                 nn.InstanceNorm3d(dim // 8),
#                 nn.LeakyReLU(),
#                 nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
#                 nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
#                 nn.InstanceNorm3d(dim // 16),
#                 nn.LeakyReLU(),
#                 nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
#                 nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
#                 nn.InstanceNorm3d(dim // 16),
#                 nn.LeakyReLU(),
#                 nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
#                 nn.Conv3d(dim // 16, args.in_channels, kernel_size=1, stride=1),
#             )

#     def forward(self, x):
#         x_out = self.swinViT(x.contiguous())[4]
#         _, c, h, w, d = x_out.shape
#         x_rec = self.order_pre(x_out)
#         x_order = self.pool3d(x_out)
#         x_order = x_order.view((-1,c))
#         x_order = self.order_head(x_order)
#         print(x_order.shape)
#         #x_rec = 0
#         # x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
#         # x4_reshape = x4_reshape.transpose(1, 2)
#         # x_order = self.order_pre(x4_reshape[:, 0])
#         # x_order = self.order_head(x_order)
#         # x_rec = x_out.flatten(start_dim=2, end_dim=4)
#         # x_rec = x_rec.view(-1, c, h, w, d)
        
#         x_rec = self.conv(x_rec)
#         return x_order, x_rec
    
class load_Swin(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(load_Swin, self).__init__()
        self.model = torchvision.models.video.__dict__['swin3d_t'](pretrained=pretrained, num_classes=num_classes)
        self.order_head = nn.Sequential(nn.BatchNorm1d(num_classes),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(num_classes,1))
                   
    def forward(self, video):
        pred = self.model(video)
        pred = self.order_head(pred)
        return pred, None
    
if __name__ == '__main__':
    inputs = torch.rand((4,1,32,128,128))