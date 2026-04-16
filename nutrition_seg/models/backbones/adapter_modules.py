import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath

_logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2

class FFN(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, in_features, hidden_features, out_features, num_layers, activation = nn.GELU, affine_func=nn.Linear, drop=0.
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_features] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([in_features] + h, h + [out_features])
        )
        self.drop = drop
        self.dropout = nn.Dropout(drop)

        self.activation = activation()

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            if self.drop > 0:
                x = self.dropout(self.activation(layer(x))) if i < self.num_layers - 1 else layer(x)
            else:
                x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return self.dropout(x) if self.drop > 0 else x
    
class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

class SpraseFeatAttn(nn.Module):
    def __init__(self, dim, num_heads, drop=0., drop_path=0., norm_layer = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        assert dim % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.head_dim = dim // num_heads
        self.num_heads = num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.out_proj = nn.Linear(dim, dim)

        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        
        self.dropout_prob = drop
        self.dropout = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self._reset_parameters()

    def forward(self, query, attn_mask = None, num_plate_emd = None):
        """ query:[plate_emd, c2, c3, c4] 
            shape: (bs, n, dim)"""
        plate_emd, c = query[:, :num_plate_emd, :], query[:, num_plate_emd:, :]
        q = self.norm_q(plate_emd)
        c = self.norm_kv(c)
        
        bs, q_length, dim = q.shape
        _, k_length, _ = c.shape
        q = self.q_proj(q).view(bs, q_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(c).view(bs, k_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(c).view(bs, k_length, self.num_heads, self.head_dim).transpose(1, 2)

        scaling = q.size(-1) ** -0.5
        attn_weight = (q @ k.transpose(-2, -1)) * scaling
        
        if attn_mask is not None:
            attn_weight += attn_mask

        attn_weight = torch.softmax(attn_weight, dim=-1)
        
        if self.training and self.dropout_prob > 0:
            attn_weight = self.dropout(attn_weight)
            
        q = attn_weight @ v  # (bs, num_heads, q_length, head_dim)

        q = q.transpose(1, 2).contiguous().view(bs, q_length, -1)
        q = self.out_proj(q)
        # q = self.dropout(q)
        # return plate_emd + self.drop_path(q)
        return q

    def _reset_parameters(self, ):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
            nn.init.constant_(self.out_proj.bias, 0)
                
"""
对于plate emd的Extractor增加masked self attention
"""
class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, with_plateffn=True, cffn_ratio=0.25, ffn_ratio = 2., num_layers = 2, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False, num_plate_emd=None):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.sprase_feat_attn = SpraseFeatAttn(dim, num_heads)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_ffn = with_plateffn
        self.with_cp = with_cp
        self.num_plate_emd = num_plate_emd

        if with_plateffn:
            self.plateffn = FFN(in_features=dim, hidden_features=int(dim * ffn_ratio), out_features=dim, num_layers=num_layers)
            self.plateffn_norm = norm_layer(dim)
            self.sprase_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            
        if with_cffn:
            self.cffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.cffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

            

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W, attn_mask = None):

        def _inner_forward(query, feat):
            # query: (bs, 1+21n, d)
            plate_emd, c = query[:, :self.num_plate_emd, :], query[:, self.num_plate_emd:, :]

            attn = self.attn(self.query_norm(c), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            c = c + attn
            if self.with_cffn:
                c = c + self.drop_path(self.cffn(self.cffn_norm(c), H, W))
            query = torch.cat([plate_emd, c], dim=1)

            plate_emd = self.sprase_feat_attn(query, attn_mask, self.num_plate_emd)

            if self.with_ffn:
                plate_emd = plate_emd + self.sprase_drop_path(self.plateffn(self.plateffn_norm(plate_emd)))
            query = torch.cat([plate_emd, c], dim=1)

            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query

  
"""
适用于RGB和depth map的Injector
"""
class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, 
                 attn_module: nn.Module = MSDeformAttn):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = attn_module(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            num_pixels = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()
            
            if feat.shape[1] != num_pixels:
                feat = feat[:, -num_pixels:, :]

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query

class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False, with_depth=None, num_plate_emd=None):
        super().__init__()

        self.injector_rgb = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        if with_depth:
            self.injector_d = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
            
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp, num_plate_emd=num_plate_emd)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp, num_plate_emd=num_plate_emd)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

        self.with_depth = with_depth

    def forward(self, x, c, d, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector_rgb(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        if self.with_depth:
            x = self.injector_d(query=x, reference_points=deform_inputs1[0],
                          feat=d, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c


# class InteractionBlockWithCls(nn.Module):
#     def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                  drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
#                  deform_ratio=1.0, extra_extractor=False, with_cp=False):
#         super().__init__()

#         self.injector_rgb = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
#                                  n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
#                                  with_cp=with_cp)
#         self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
#                                    norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
#                                    cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
#         if extra_extractor:
#             self.extra_extractors = nn.Sequential(*[
#                 Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
#                           with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
#                           drop=drop, drop_path=drop_path, with_cp=with_cp)
#                 for _ in range(2)
#             ])
#         else:
#             self.extra_extractors = None

#     def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W):
#         x = self.injector_rgb(query=x, reference_points=deform_inputs1[0],
#                           feat=c, spatial_shapes=deform_inputs1[1],
#                           level_start_index=deform_inputs1[2])
#         x = torch.cat((cls, x), dim=1)
#         for idx, blk in enumerate(blocks):
#             x = blk(x, H, W)
#         cls, x = x[:, :1, ], x[:, 1:, ]
#         c = self.extractor(query=c, reference_points=deform_inputs2[0],
#                            feat=x, spatial_shapes=deform_inputs2[1],
#                            level_start_index=deform_inputs2[2], H=H, W=W)
#         if self.extra_extractors is not None:
#             for extractor in self.extra_extractors:
#                 c = extractor(query=c, reference_points=deform_inputs2[0],
#                               feat=x, spatial_shapes=deform_inputs2[1],
#                               level_start_index=deform_inputs2[2], H=H, W=W)
#         return x, c, cls


class SpatialPriorModule(nn.Module):
    def __init__(self, in_channels = 3, kernel_size = 3, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(in_channels, inplanes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1) // 2, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs
    

