import os
import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS, POSITIONAL_ENCODING
)
from mmcv.utils import build_from_cfg
from projects.mmdet3d_plugin.models.attention import gen_sineembed_for_position

class PointsEncoder(nn.Module):
    def __init__(self, feat_channel, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_mlp = nn.Sequential(
            nn.Linear(feat_channel, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.second_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.encoder_channel),
        )

    def forward(self, x, mask=None):
        '''
            x: [BN, M-1, 6], polygon_featreus
            mask: [BN, M-1], valida_mask
        '''
        BS, M, C = x.shape
        device = x.device   

        x_valid = self.first_mlp(x[mask]) # [BN, C]
        x_features = torch.zeros(BS, M, self.encoder_channel, device=device) # [BN, M, C]
        x_features[mask] = x_valid

        pooled_feature = x_features.max(dim=1)[0] # [BN, C]
        x_features = torch.cat(
            [x_features, pooled_feature.unsqueeze(1).repeat(1, M, 1)], dim=-1
        )

        x_features_valid = self.second_mlp(x_features[mask])
        res = torch.zeros(BS, M, self.encoder_channel, device=device)  # [BN, M, C]
        res[mask] = x_features_valid
        res = res.max(dim=1)[0] # [BN, C]
        return res

@PLUGIN_LAYERS.register_module()
class MapEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel,
        n_laneline_type,
        map_dim,
        map_pos,
        pos_norm,
        pc_range,
        fourier_embed_tag=False,
        fourier_embed_cfg=None,
    ) -> None:
        super().__init__()
        self.map_dim = map_dim
        self.polygon_encoder = PointsEncoder(polygon_channel, map_dim)
        self.laneline_type_emb = nn.Embedding(n_laneline_type, map_dim)
        self.map_pos = map_pos
        self.pos_norm = pos_norm
        self.fourier_embed_tag = fourier_embed_tag
        self.fourier_embed_cfg = fourier_embed_cfg
        if self.map_pos is True:
            if self.fourier_embed_tag:
                self.map_pos_embed = build_from_cfg(fourier_embed_cfg, POSITIONAL_ENCODING)
            else:
                self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
                self.map_pos_embed = nn.Sequential(
                        nn.Linear(self.map_dim, self.map_dim * 2),
                        nn.ReLU(),
                        nn.Linear(self.map_dim * 2, self.map_dim),
                    )
            

    def forward(self, map_target, map_label, map_mask):
        '''
        Args:
            map_target: [B, N, M, 2]
            map_label: [B, N]
            map_mask: [B, N]
        Return:
            map_feats: [B, N, C]
            x_pos: [B, N, C]
            map_mask: [B, N]
        '''
        valid_mask = map_mask 
        n_pt = map_target.shape[2]
        polygon_center = map_target.clone()[:,:,n_pt//2,:] # [B, N, 2]

        point_position = map_target.clone()   # point_position
        point_vector = point_position[:,:,1:,:] - point_position[:,:,:-1,:] # [B, N, M-1, 2]
        point_vector_norm = torch.norm(point_vector, dim=-1)  # [B, N, M-1, 2]
        point_vector_norm += valid_mask.unsqueeze(-1).expand_as(point_vector_norm) + 1e-6 * torch.ones_like(point_vector_norm)
        point_orientation = torch.stack([
            point_vector[:,:,:,0] / point_vector_norm,
            point_vector[:,:,:,1] / point_vector_norm,
        ], dim=-1) # [B, N, M-1, 2]

        rel_point_position = point_position[:, :, :-1, :] - polygon_center.unsqueeze(2).expand_as(point_vector)
        polygon_feature = torch.cat(
            [
                rel_point_position, # [B, N, M-1, 2]
                point_vector,  # [B, N, M-1, 2]
                point_orientation, # [B, N, M-1, 2]
            ],
            dim=-1
        )   # (bs, max_poly, n_pt, 6)

        bs, max_poly, n_pt, channel = polygon_feature.shape
        valid_mask = valid_mask.unsqueeze(-1).repeat(1, 1, n_pt).view(bs * max_poly, n_pt) # [B * N, M]
        polygon_feature = polygon_feature.reshape(bs * max_poly, n_pt, channel) # [B * N, M, C]
        
        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, max_poly, -1) # [B, N, D]
        polygon_laneline_type = map_label[:,:].long() # [B, N]
        x_laneline_type = self.laneline_type_emb(polygon_laneline_type) # [B, N, D]

        x_polygon = x_polygon + x_laneline_type 

        map_feats = torch.zeros(bs, max_poly, self.map_dim, device=x_polygon.device)  # [B, N, C]
        map_feats[map_mask] = x_polygon[map_mask]
        N_valid, _ = x_polygon[map_mask].shape
        if self.map_pos is True:
            map_pos = torch.zeros(N_valid, 3).to(x_polygon.device) # [N_Valid, 3]
            map_pos[:,:2] = polygon_center[map_mask][:, :] 
            if self.pos_norm:
                map_pos = (map_pos[:,:3] - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3]) # [N_Valid, 3]
            if self.fourier_embed_tag:
                # use the height dimension
                map_embed = self.map_pos_embed(map_pos[:, :2])
            else:
                # only care the BEV loaction
                map_embed = self.map_pos_embed(gen_sineembed_for_position(map_pos[:,:2])) # [N_Valid, C]
            
        else:
            map_embed = torch.zeros(N_valid, self.map_dim).to(x_polygon.device)

        x_pos = torch.zeros(bs, max_poly, self.map_dim, device=x_polygon.device)
        x_pos[map_mask] = map_embed

        return map_feats, x_pos, ~map_mask