import torch
import TransUnet.vit_seg_modeling
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 4
config_vit.n_skip = 3
config_vit.patches.grid = (int(224 / 16), int(224 / 16))

class params:
    def __init__(self):
        self.memory_size = 5000
        self.proj_dim = 256

args = params()


inputs = torch.randn(8,3,224,224)
net = TransUnet.vit_seg_modeling.VisionTransformer_cst_temporal_flow(args, config_vit, img_size=224, num_classes=config_vit.n_classes)

res = net(inputs)