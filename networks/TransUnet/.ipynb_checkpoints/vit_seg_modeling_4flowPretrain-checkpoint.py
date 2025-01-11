from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
import copy
import logging
import math

from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .vit_seg_modeling import *
from .. import channel_attn
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from ..projection import ProjectionHead 
from ..EMA_VFI import feature_extractor
from ..flow_estimation import *


class VisionTransformer_cst_temporal_flow(nn.Module):
    def __init__(self, args, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer_cst_temporal_flow, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
#         self.decoder = DecoderCup(config)
#         self.segmentation_head = SegmentationHead(
#             in_channels=config['decoder_channels'][-1],
#             out_channels=config['n_classes'],
#             kernel_size=3,
#         )
        
#         if self.args.classify:
#             self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
#             self.classify_head = nn.Linear(512, 4)

#         self.proj_head = ProjectionHead(dim_in=16) 
        self.config = config
        self.attn = feature_extractor.my_attention_TransUnet_add()
        self.flow_estimation = MultiScaleFlow()
        
        self.register_buffer("segment_queue", torch.randn(num_classes, args.memory_size, args.proj_dim))
        self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

        self.register_buffer("pixel_queue", torch.randn(num_classes, args.memory_size, args.proj_dim))
        self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
        self.register_buffer("pixel_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

    def forward(self, imgs):
        if imgs.size()[1] == 1:
            imgs = imgs.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(imgs)  # (B, n_patch, hidden)
        # appearance_attn, motion_attn = self.attn(features[:2])
        # appearance_attn.extend(features[2:])
        appearance_attn, motion_attn = self.attn(features[:3])
        appearance_attn.extend(features[3:])
        warped_imgs, warped_logits, flow_list = self.flow_estimation(imgs, imgs, appearance_attn, motion_attn)
        # q = self.proj_head(x)
        # if self.args.classify:
        #     x_cls = self.avg_pool(features[0]).view(features[0].size(0), -1)
        #     chamber_pred = self.classify_head(x_cls)
        # else:
        #     chamber_pred = torch.randn(x.shape[0],4).to(logits.device)
        return flow_list, warped_imgs

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
                        
                        
                        
class VisionTransformer_cst_temporal_flow_woResidual(nn.Module):
    def __init__(self, args, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer_cst_temporal_flow_woResidual, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
#         self.decoder = DecoderCup(config)
#         self.segmentation_head = SegmentationHead(
#             in_channels=config['decoder_channels'][-1],
#             out_channels=config['n_classes'],
#             kernel_size=3,
#         )
        
#         if self.args.classify:
#             self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
#             self.classify_head = nn.Linear(512, 4)

        # self.proj_head = ProjectionHead(dim_in=16) 
        self.config = config
        self.attn = feature_extractor.my_attention_TransUnet()
        self.flow_estimation = MultiScaleFlow()
        
        self.register_buffer("segment_queue", torch.randn(num_classes, args.memory_size, args.proj_dim))
        self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

        self.register_buffer("pixel_queue", torch.randn(num_classes, args.memory_size, args.proj_dim))
        self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
        self.register_buffer("pixel_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

    def forward(self, imgs):
        if imgs.size()[1] == 1:
            imgs = imgs.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(imgs)  # (B, n_patch, hidden)
        appearance_attn, motion_attn = self.attn(features[:3])
        appearance_attn.extend(features[3:])
        # x = self.decoder(x, appearance_attn)
        # logits = self.segmentation_head(x)
        warped_imgs, warped_logits, flow_list = self.flow_estimation(imgs, imgs, appearance_attn[:3], motion_attn)
        # q = self.proj_head(x)
        # if self.args.classify:
        #     x_cls = self.avg_pool(features[0]).view(features[0].size(0), -1)
        #     chamber_pred = self.classify_head(x_cls)
        # else:
        #     chamber_pred = torch.randn(x.shape[0],4).to(logits.device)
        return flow_list, warped_imgs

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)