import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class InterFrameAttention(nn.Module):
    def __init__(self, dim, motion_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cor_embed = nn.Linear(2, motion_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.motion_proj = nn.Linear(motion_dim, motion_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2, cor, H, W, mask=None):
        B, N, C = x1.shape
        B, N, C_c = cor.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        cor_embed_ = self.cor_embed(cor)
        cor_embed = cor_embed_.reshape(B, N, self.num_heads, self.motion_dim // self.num_heads).permute(0, 2, 1, 3)
        k, v = kv[0], kv[1]    
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0] # mask: nW, N, N
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        c_reverse = (attn @ cor_embed).transpose(1, 2).reshape(B, N, -1)
        motion = self.motion_proj(c_reverse-cor_embed_)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, motion


class MotionFormerBlock(nn.Module):
    def __init__(self, dim, motion_dim, num_heads, window_size=0, shift_size=0, mlp_ratio=4., bidirectional=True, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()
        self.window_size = window_size
        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)
        self.shift_size = shift_size
        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)
        self.bidirectional = bidirectional
        self.norm1 = norm_layer(dim)
        self.attn = InterFrameAttention(
            dim,
            motion_dim, 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, cor, H, W, B):
        x = x.view(2*B, H, W, -1)
        x_pad, mask = pad_if_needed(x, x.size(), self.window_size)
        cor_pad, _ = pad_if_needed(cor, cor.size(), self.window_size)

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            cor_pad = torch.roll(cor_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            
            if hasattr(self, 'HW') and self.HW.item() == H_p * W_p: 
                shift_mask = self.attn_mask
            else:
                shift_mask = torch.zeros((1, H_p, W_p, 1))  # 1 H W 1
                h_slices = (slice(0, -self.window_size[0]),
                            slice(-self.window_size[0], -self.shift_size[0]),
                            slice(-self.shift_size[0], None))
                w_slices = (slice(0, -self.window_size[1]),
                            slice(-self.window_size[1], -self.shift_size[1]),
                            slice(-self.shift_size[1], None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        shift_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(shift_mask, self.window_size).squeeze(-1)  
                shift_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                shift_mask = shift_mask.masked_fill(shift_mask != 0, 
                                float(-100.0)).masked_fill(shift_mask == 0, 
                                float(0.0))
                                
                if mask is not None:
                    shift_mask = shift_mask.masked_fill(mask != 0, 
                                float(-100.0))
                self.register_buffer("attn_mask", shift_mask)
                self.register_buffer("HW", torch.Tensor([H_p*W_p]))
        else: 
            shift_mask = mask
        
        if shift_mask is not None:
            shift_mask = shift_mask.to(x_pad.device)
            

        _, Hw, Ww, C = x_pad.shape
        x_win = window_partition(x_pad, self.window_size)
        cor_win = window_partition(cor_pad, self.window_size)

        nwB = x_win.shape[0]
        x_norm = self.norm1(x_win)

        x_reverse = torch.cat([x_norm[nwB//2:], x_norm[:nwB//2]])
        x_appearence, x_motion = self.attn(x_norm, x_reverse, cor_win, H, W, shift_mask)
        x_norm = x_norm + self.drop_path(x_appearence)

        x_back = x_norm
        x_back_win = window_reverse(x_back, self.window_size, Hw, Ww)
        x_motion = window_reverse(x_motion, self.window_size, Hw, Ww)
        
        if self.shift_size[0] or self.shift_size[1]:
            x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x_motion = torch.roll(x_motion, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, x.size(), self.window_size).view(2*B, H * W, -1)
        x_motion = depad_if_needed(x_motion, cor.size(), self.window_size).view(2*B, H * W, -1)
            
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x, x_motion
    

            
        
        