import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)

from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln


@PLUGIN_LAYERS.register_module()
class DLPRefine(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        auto_regression_tag=False,
    ):
        super(DLPRefine, self).__init__()
        self.embed_dims = embed_dims # 256
        self.fut_ts = fut_ts # 12
        self.fut_mode = fut_mode # 6
        self.ego_fut_ts = ego_fut_ts # 6
        self.ego_fut_mode = ego_fut_mode # ego_fut_mode
        self.auto_regression_tag = auto_regression_tag
        if auto_regression_tag:
            ego_pred_dim = 2
            agent_pred_dim = 2
        else:
            ego_pred_dim = ego_fut_ts * 2
            agent_pred_dim = fut_ts * 2


        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, agent_pred_dim),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_pred_dim),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1) # [B, N, M]
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2) # [B, N, M1, T1, 2]
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1) # [B, 1, M]
        plan_reg = self.plan_reg_branch(plan_query).reshape(bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts, 2) # [B, 1, M1, T2, 2]
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed) # [B, 1, 10]
        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status