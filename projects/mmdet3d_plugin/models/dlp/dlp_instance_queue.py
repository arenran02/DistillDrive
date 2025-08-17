import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from projects.mmdet3d_plugin.ops import feature_maps_format
from projects.mmdet3d_plugin.core.box3d import *


@PLUGIN_LAYERS.register_module()
class DLPInstanceQueue(nn.Module):
    def __init__(
        self,
        embed_dims,
        queue_length=0,
        max_time_interval=2,
        multi_ego_anchor=False,
        multi_ego_mode=18,
    ):
        super(DLPInstanceQueue, self).__init__()
        self.embed_dims = embed_dims # 256
        self.queue_length = queue_length # 4
        self.max_time_interval = max_time_interval
        self.multi_ego_anchor = multi_ego_anchor
        if multi_ego_anchor:
            self.ego_anchor = nn.Parameter(
                torch.tensor([[0, 0.5, -1.84 + 1.56/2, np.log(4.08), np.log(1.73), np.log(1.56), np.pi / 2, 0, 0],], dtype=torch.float32),
                requires_grad=False,
            ).repeat(multi_ego_mode, 1)
        else:
            self.ego_anchor = nn.Parameter(
                torch.tensor([[0, 0.5, -1.84 + 1.56/2, np.log(4.08), np.log(1.73), np.log(1.56), np.pi / 2, 0, 0],], dtype=torch.float32),
                requires_grad=False,
            )            

        self.reset()

    def reset(self):
        self.metas = None
        self.prev_instance_id = None
        self.prev_confidence = None
        self.period = None
        self.instance_feature_queue = []
        self.anchor_queue = []
        self.prev_ego_status = None
        self.ego_period = None
        self.ego_feature_queue = []
        self.ego_anchor_queue = []

    def get(
        self,
        agent_target,
        agent_feature,
        metas,
        batch_size,
        agent_mask,
        plan_mode_query,
    ):
        if (
            self.period is not None
            and batch_size == self.period.shape[0]
        ):
            T_temp2cur = agent_target[0].new_tensor(
                np.stack(
                    [
                        x["T_global_inv"]
                        @ self.metas["img_metas"][i]["T_global"]
                        for i, x in enumerate(metas["img_metas"])
                    ]
                )
            ) 
            for i in range(len(self.anchor_queue)):
                temp_anchor = self.anchor_queue[i]
                temp_anchor = self.anchor_projection(
                    temp_anchor,
                    [T_temp2cur],
                )[0]
                self.anchor_queue[i] = temp_anchor

            for i in range(len(self.ego_anchor_queue)):
                temp_anchor = self.ego_anchor_queue[i]
                temp_anchor = self.anchor_projection(
                    temp_anchor,
                    [T_temp2cur],
                )[0]
                self.ego_anchor_queue[i] = temp_anchor
            mask = torch.zeros((agent_target.shape[0]), device=agent_target.device).bool()
            for bs in range(len(mask)):
                history_time = self.metas["img_metas"][bs]["timestamp"]
                time_interval = metas["img_metas"][bs]["timestamp"] - history_time
                mask[bs] = abs(time_interval) <= self.max_time_interval

        else:
            mask = torch.zeros((agent_target.shape[0]), device=agent_target.device).bool()
            self.reset()

        # prepare for motion and planning target
        self.prepare_motion(agent_target, agent_feature, agent_mask, metas['img_metas'])
        ego_feature, ego_anchor = self.prepare_planning(plan_mode_query, mask, batch_size)
        # temporal stacking
        temp_instance_feature = torch.stack(self.instance_feature_queue, dim=2) # [B, N, T, D]
        temp_anchor = torch.stack(self.anchor_queue, dim=2) # [B, N, T, 9]
        temp_ego_feature = torch.stack(self.ego_feature_queue, dim=2)
        temp_ego_anchor = torch.stack(self.ego_anchor_queue, dim=2)
        # period setting
        ego_period = self.ego_period.clone().repeat(1, plan_mode_query.shape[2]) # [B, M]
        period = torch.cat([self.period, ego_period], dim=1) # [B, N + M]
        # temporal feature concate
        temp_instance_feature = torch.cat([temp_instance_feature, temp_ego_feature], dim=1)
        temp_anchor = torch.cat([temp_anchor, temp_ego_anchor], dim=1)
        num_agent = temp_anchor.shape[1]
        # temporal mask
        temp_mask = torch.arange(len(self.anchor_queue), 0, -1, device=temp_anchor.device)
        temp_mask = temp_mask[None, None].repeat((batch_size, num_agent, 1))
        temp_mask = torch.gt(temp_mask, period[..., None])

        return ego_feature, ego_anchor, temp_instance_feature, temp_anchor, temp_mask

    def prepare_motion(
        self,
        agent_target,
        agent_feature,
        agent_mask,
        img_metas,
    ):
        '''keep available of memory bank'''
        instance_feature = agent_feature # [B, N1, D]
        det_anchors = agent_target # [B, N1, 9]
        if self.period == None:
            self.period = instance_feature.new_zeros(instance_feature.shape[:2]).long()
        else:
            instance_id = -torch.ones((agent_mask.shape[:2]), device=agent_mask.device) * 1e2
            for bs, instance_id_tmp in enumerate(img_metas):
                instance_id[bs,:len(instance_id_tmp['instance_id'])] = torch.tensor(instance_id_tmp['instance_id'], device=agent_mask.device)

            prev_instance_id = self.prev_instance_id
            match = instance_id[..., None] == prev_instance_id[:, None]
            for i in range(len(self.instance_feature_queue)):
                temp_feature = self.instance_feature_queue[i]
                temp_feature = (
                    match[..., None] * temp_feature[:, None]
                ).sum(dim=2)
                self.instance_feature_queue[i] = temp_feature

                temp_anchor = self.anchor_queue[i]
                temp_anchor = (
                    match[..., None] * temp_anchor[:, None]
                ).sum(dim=2)
                self.anchor_queue[i] = temp_anchor

            self.period = (
                match * self.period[:, None]
            ).sum(dim=2)

        self.instance_feature_queue.append(instance_feature.detach()) # [[B, N1, D]]
        self.anchor_queue.append(det_anchors.detach()) # [[B, N1, 9]]
        self.period += 1
        # pop the top of feature
        if len(self.instance_feature_queue) > self.queue_length:
            self.instance_feature_queue.pop(0)
            self.anchor_queue.pop(0)
        self.period = torch.clip(self.period, 0, self.queue_length) # limit to [0, self.queue_length]

    def prepare_planning(
        self,
        plan_mode_query,
        mask,
        batch_size,
    ):
        # ego instance init
        ego_feature = plan_mode_query.clone().permute(0, 2, 1, 3).squeeze(2)
        ego_anchor = torch.tile(
            self.ego_anchor[None], (batch_size, 1, 1)
        ).to(ego_feature.device) # [1, 1, 9]
        if self.prev_ego_status is not None:
            prev_ego_status = torch.where(
                mask[:, None, None],
                self.prev_ego_status,
                self.prev_ego_status.new_tensor(0),
            )
        # TODO 
            if self.multi_ego_anchor is False:
                ego_anchor[..., VY-1] = prev_ego_status[..., 6]
        if self.multi_ego_anchor is False:
            ego_anchor = ego_anchor.repeat(1, ego_feature.shape[1], 1) # [B, M, 9]


        if self.ego_period == None:
            self.ego_period = ego_feature.new_zeros((batch_size, 1)).long()
        else:
            self.ego_period = torch.where(
                mask[:, None],
                self.ego_period,
                self.ego_period.new_tensor(0),
            )

        self.ego_feature_queue.append(ego_feature.detach())
        self.ego_anchor_queue.append(ego_anchor.detach())
        self.ego_period += 1
        
        if len(self.ego_feature_queue) > self.queue_length:
            self.ego_feature_queue.pop(0)
            self.ego_anchor_queue.pop(0)
        self.ego_period = torch.clip(self.ego_period, 0, self.queue_length)

        return ego_feature, ego_anchor

    def cache_motion(self, agent_mask, metas):
        self.metas = metas
        self.prev_confidence = agent_mask.float()
        instance_id = -torch.ones((agent_mask.shape), device=agent_mask.device)
        for bs, instance_id_tmp in enumerate(metas['img_metas']):
            instance_id[bs,:len(instance_id_tmp['instance_id'])] = torch.tensor(instance_id_tmp['instance_id'], device=agent_mask.device)
        self.prev_instance_id = instance_id

    def cache_planning(self, ego_feature, ego_status):
        self.prev_ego_status = ego_status.detach()
        self.ego_feature_queue[-1] = ego_feature.detach()

    @staticmethod
    def anchor_projection(
        anchor,
        T_src2dst_list,
    ):
        '''
            For gt_bboxes_3d(x, y, z, w, l, h, yaw, vx, vy)
        '''

        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            vel = anchor[..., VX-1:] # [B, N, 2]
            vel_dim = vel.shape[-1] # 2
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            ) # T_mat [B, 1, 4, 4]

            center = anchor[..., [X, Y, Z]] # [B, N, 3]
            center = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3]
            ) # center rotation + translation # [B, N, 3]

            size = anchor[..., [W, L, H]] # [B, N, 3]
            cos_yaw = torch.cos(anchor[..., [YAW], None])
            sin_yaw = torch.sin(anchor[..., [YAW], None])
            yaw = torch.matmul(
                T_src2dst[..., :2, :2],
                torch.cat([cos_yaw, sin_yaw], dim=-2),
            ).squeeze(-1) # [B, N, 2]
            yaw_recon = torch.atan2(yaw[..., 1], yaw[..., 0]).unsqueeze(-1)

            vel = torch.matmul(
                T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]
            ).squeeze(-1) # [B, N, 3]

            dst_anchor = torch.cat([center, size, yaw_recon, vel], dim=-1)
            dst_anchors.append(dst_anchor)
        return dst_anchors