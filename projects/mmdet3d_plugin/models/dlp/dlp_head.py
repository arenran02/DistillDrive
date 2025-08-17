from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.diffusion import DistributionModule, PredictModel

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk


@HEADS.register_module()
class DLPHead(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        pred_delta=False,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        fourier_embed_tag=False,
        fourier_embed_cfg=None,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        agent_encoder=None,
        map_encoder=None,
        anchor_encoder=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        distribution_cfg=None,
        multi_modal_cfg=None,
        multi_ego_status=False,
        dqn_cfg=None,
        agent2lidar_tag=None,
        motion_clip=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        loss_vae_gen=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
    ):
        super(DLPHead, self).__init__()
        self.fut_ts = fut_ts 
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.pred_delta = pred_delta
        
        self.decouple_attn = decouple_attn
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.agent_encoder = build(agent_encoder, ATTENTION)
        self.map_encoder = build(map_encoder, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)

        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)

        if dqn_cfg is not None:
            self.dqn_model = build(dqn_cfg, PLUGIN_LAYERS)
        else:
            self.dqn_model = None

        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims
        self.fourier_embed_tag = fourier_embed_tag

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)
        
        # multi modal 
        self.multi_ego_status = multi_ego_status
        if multi_modal_cfg is not None:
            self.multi_modal = True
            self.plan_instance_dim = multi_modal_cfg.plan_instance_dim
            self.ego_feature_avp = nn.AdaptiveAvgPool1d(1)
            self.ego_pos_avp = nn.AdaptiveAvgPool1d(1)
            self.ego_anchor_tag = multi_modal_cfg.get('ego_anchor_tag', False)
        else:
            self.plan_instance_dim = 128
            self.multi_modal = False
            self.multi_modal_role = None
        
        self.agent2lidar_tag = agent2lidar_tag
        self.motion_clip = motion_clip

        # motion init
        motion_anchor = np.load(motion_anchor) # [10, 6, 12, 2] -->(K, N, T, 2)
        if self.motion_clip:
            motion_anchor = motion_anchor[..., [1, 0]]
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        if self.fourier_embed_tag:
            self.motion_anchor_encoder = build_from_cfg(fourier_embed_cfg, POSITIONAL_ENCODING)        
        else:
            self.motion_anchor_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 1),
                Linear(embed_dims, embed_dims),
            )

        # plan anchor init
        plan_anchor = np.load(plan_anchor) # [M, N, T, 2]
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        if self.fourier_embed_tag:
            self.plan_anchor_encoder = build_from_cfg(fourier_embed_cfg, POSITIONAL_ENCODING) 
        else:
            self.plan_anchor_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 1),
                Linear(embed_dims, embed_dims),
            )

        # planning instance and planning delta
        plan_anchor_delta = np.copy(plan_anchor)
        plan_anchor_delta[:, :, 1:, :] = plan_anchor_delta[:, :, 1:, :] - plan_anchor_delta[:, :, :-1, :]
        self.plan_anchor_delta = nn.Parameter(
            torch.tensor(plan_anchor_delta, dtype=torch.float32),
            requires_grad=False,
        )
        # planning encoder
        if self.fourier_embed_tag:
            # Just for instance encoder
            fourier_embed_cfg['input_dim'] = fourier_embed_cfg['input_dim'] * self.ego_fut_ts
            self.plan_instance_encoder = build_from_cfg(fourier_embed_cfg, POSITIONAL_ENCODING) 
        else:
            self.plan_instance_encoder = nn.Sequential(
                *linear_relu_ln(self.plan_instance_dim * self.ego_fut_ts, 1, 1),
                Linear(self.plan_instance_dim * self.ego_fut_ts, embed_dims),
            )

        self.num_det = num_det
        self.num_map = num_map

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def get_motion_anchor(
        self, 
        cls_ids, 
        prediction,
    ):

        motion_anchor = self.motion_anchor[cls_ids.long()] # [B, N ,6, 12, 2]
        prediction = prediction.detach()
        if self.agent2lidar_tag:
            return self._agent2lidar(motion_anchor, prediction)
        else:
            return motion_anchor

    def _agent2lidar(self, trajs, boxes):

        yaw = boxes[..., YAW]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        ) # [2, 2, B, N]

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def prepare_agent_traj(
        self,
        gt_bboxes_3d_list,
        gt_labels_3d_list,
        gt_agent_fut_trajs_list,
        gt_agent_fut_masks_list,
    ):
        '''
        Args:
            gt_bboxes_3d_list: [bbox_bs0, bbox_bs1, ...], bbox_bs0:(N, 9)
            gt_labels_3d_list: [label_bs0, label_bs1, ...], label_bs0:(N)
            gt_agent_fut_trajs_list: [traj_bs0, traj_bs1, ...], traj_bs0:(N, T, 2)
            gt_agent_fut_masks_list: [mask_bs0, traj_bs1, ...], mask_bs0:(N, T)
        Return:
            agent_target: (B, N, 9)
            agent_label: (B, N)
            agent_traj: (B, N, T, 2)
            agent_mask: (B, N, T) 
        '''

        BS = len(gt_bboxes_3d_list)
        agent_target = torch.zeros((BS, self.num_det, 9), device=gt_bboxes_3d_list[0].device)
        agent_label = torch.zeros((BS, self.num_det), device=gt_bboxes_3d_list[0].device)
        agent_traj = torch.zeros((BS, self.num_det, self.fut_ts, 2), device=gt_bboxes_3d_list[0].device)
        agent_mask = torch.zeros((BS, self.num_det, self.fut_ts), device=gt_bboxes_3d_list[0].device, dtype=bool)
        for bs in range(BS):
            agent_target[bs, :len(gt_bboxes_3d_list[bs]), :] = gt_bboxes_3d_list[bs]
            agent_label[bs, :len(gt_bboxes_3d_list[bs])] = gt_labels_3d_list[bs]
            agent_mask_tmp = gt_agent_fut_masks_list[bs].bool()[:, :self.fut_ts]
            agent_mask[bs, :len(gt_bboxes_3d_list[bs]), :] = agent_mask_tmp
            if gt_bboxes_3d_list[bs].shape[0] != 0:
                agent_traj[bs, :len(gt_bboxes_3d_list[bs]), :][agent_mask_tmp] = gt_agent_fut_trajs_list[bs][:, :self.fut_ts, :][agent_mask_tmp].float()

        return agent_target, agent_label, agent_traj, agent_mask

    def prepare_map(
        self,
        gt_map_pts_list,
        gt_map_labels_list,
    ):
        '''
        Args:
            gt_map_pts_list: [map_bs0, map_bs1, ...], map_bs0:(N, P, M, 9), N(nums of lane), P(permute 38, default use 0), M(points of lane)
            gt_map_labels_list: [map_label_bs0, map_label_bs1, ...], map_label_bs0:(N)
        Return:
            map_target: (B, N, M, 2)
            map_label: (B, N)
            map_mask: (B, N)
        '''

        BS = len(gt_map_pts_list)
        # map_num = max([(len(gt_map_pts_pre)) for gt_map_pts_pre in gt_map_pts_list])
        map_target = torch.zeros((BS, self.num_map, gt_map_pts_list[0].shape[2], 2), device=gt_map_pts_list[0].device)
        map_label = torch.zeros((BS, self.num_map), device=gt_map_pts_list[0].device)
        map_mask = torch.zeros((BS, self.num_map), device=gt_map_pts_list[0].device, dtype=bool)
        for bs in range(BS):
            # 0 is for permute dimension
            map_target[bs, :len(gt_map_pts_list[bs]), :] = gt_map_pts_list[bs][:, 0, :, :]
            map_label[bs, :len(gt_map_pts_list[bs])] = gt_map_labels_list[bs][:]
            map_mask[bs, :len(gt_map_pts_list[bs])] = True
        return map_target, map_label, map_mask


    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self, 
        metas,
    ):   

        # agent encoder
        agent_target, agent_label, agent_traj, agent_mask = self.prepare_agent_traj(
            metas['gt_bboxes_3d'], metas['gt_labels_3d'], metas['gt_agent_fut_trajs'], metas['gt_agent_fut_masks']
        )
        agent_feature, agent_pos_embed, agent_padding_mask = self.agent_encoder(agent_target, agent_traj, agent_mask)
        # map encoder
        map_target, map_label, map_mask = self.prepare_map(metas['gt_map_pts'], metas['gt_map_labels'])
        map_feature, map_pos, map_padding_mask = self.map_encoder(map_target, map_label, map_mask)
        # =========== det/map feature/anchor ===========
        instance_feature = agent_feature # [B, N1, D]
        anchor_embed = agent_pos_embed # [B, N1, D]
        det_anchors = agent_target # [B, N1, 11]
        det_label = agent_label # [B, N1]

        map_instance_feature = map_feature # [B, N2, D]
        map_embed = map_pos # [B, N2, D]
        map_anchors = map_target # [B, N2, 20, 2]

        bs, num_anchor, dim = instance_feature.shape
        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(agent_label, det_anchors) # [B, N, M, T, 2]
        plan_anchor = torch.tile(
            self.plan_anchor[None], (bs, 1, 1, 1, 1) 
        ) # [B, 3, M, T, 2]
        plan_anchor_delta = torch.tile(
            self.plan_anchor_delta[None], (bs, 1, 1, 1, 1)
        ) # [B, 3, M, T, 2]
        
        if self.fourier_embed_tag:
            motion_mode_pos = self.motion_anchor_encoder(motion_anchor[..., -1, :])# [B, N, M, D]
            plan_mode_pos = self.plan_anchor_encoder(plan_anchor[..., -1, :]).flatten(1, 2).unsqueeze(1) # [B, 1, C * M, D]
            plan_mode_query = self.plan_instance_encoder(plan_anchor_delta.flatten(-2)).flatten(1, 2).unsqueeze(1)
        else:
            motion_pos = gen_sineembed_for_position(motion_anchor[..., -1, :]) # endpoint [B, N, M, D]
            motion_mode_pos = self.motion_anchor_encoder(motion_pos) # [B, N, M, D]
            plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :]) # [B, C, M, D]
            plan_mode_pos = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1) # [B, 1, C * M, D]
            plan_instance_pos = gen_sineembed_for_position(plan_anchor_delta, 128) # [B, C, M, T, F]
            plan_mode_query = self.plan_instance_encoder(plan_instance_pos.flatten(-2)).flatten(1, 2).unsqueeze(1) # [B, 1, 18, D]

        # =========== get ego/temporal feature/anchor ===========
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            agent_target,
            agent_feature,
            metas,
            bs,
            agent_mask,
            plan_mode_query,
        )
        
        #  process temporal feature 
        ego_anchor_embed = self.anchor_encoder(ego_anchor) # [B, M, C]
        temp_anchor_embed = self.anchor_encoder(temp_anchor)  # [B, N+M, C]
        temp_instance_feature = temp_instance_feature.flatten(0, 1) # [(N+M) * B, T, D]
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1) # [(N+M) * B, T, D]
        temp_mask = temp_mask.flatten(0, 1) # [(N+M) * B, T]
        # =========== cat instance and ego ===========
        instance_feature = torch.cat([instance_feature, ego_feature], dim=1) # [B, N+1, D]
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1) # [B, N+1, D]
        ego_padding_mask = torch.zeros((bs, plan_mode_query.shape[2]), dtype=bool, device=anchor_embed.device) # [B, C * M]
        instance_padding_mask = torch.cat([agent_padding_mask, ego_padding_mask], dim=1)  # [B, N + C * M]
        ego_instance_num = self.ego_fut_mode * 3
        ego_feature_memory = ego_feature.clone()

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        planning_memory_feature = []
        planning_feature = []
        for i, op in enumerate(self.operation_order):
            if op == "temp_gnn":
                # temporal
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1), # [(N+M) * B, 1, D]
                    temp_instance_feature, # [(N+M) * B, T, D] temporal feature
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1), # [N+M, 1, D]
                    key_pos=temp_anchor_embed, # [(N+M) * B, T, D]
                    key_padding_mask=temp_mask, # [(N+M) * B, T]
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + ego_instance_num, dim) #[B, N + 1, D]
            elif op == "gnn":
                # key anget
                instance_feature = self.graph_model(
                    i,
                    instance_feature,  # [B, N+1, D]
                    query_pos=anchor_embed, # [B, N+1, D]
                    key_padding_mask=instance_padding_mask,
                )
            elif op == "cross_gnn":
                # direction attention
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature, # [B, M, D]
                    query_pos=anchor_embed,
                    key_pos=map_embed, # [B, M, D]
                    key_padding_mask=map_padding_mask,
                ) # [B, N1 + ME, D]
            elif op == "norm" or op == "ffn":
                # normarization and linear
                instance_feature = self.layers[i](instance_feature)
                if op == "ffn":
                    planning_memory_feature.append(instance_feature[:, num_anchor:])
            elif op == "refine":
                motion_query = motion_mode_pos + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2) # [B, N, M, D]
                plan_query = plan_mode_pos + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(1) # [B, 1, C * M, D] 
                # Only get one Status, Must to commperss in MultiModal
                if self.multi_ego_status:
                    ego_feature = instance_feature[:, num_anchor:]
                    ego_anchor_embed = anchor_embed[:, num_anchor:]
                else:
                    ego_feature = self.ego_feature_avp(instance_feature[:, num_anchor:].permute(0, 2, 1)).permute(0, 2, 1)
                    ego_anchor_embed = self.ego_pos_avp(anchor_embed[:, num_anchor:].permute(0, 2, 1)).permute(0, 2, 1)
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,
                ) = self.layers[i](
                    motion_query,
                    plan_query,
                    ego_feature,
                    ego_anchor_embed,
                )
                
                if self.pred_delta:
                    plan_delta = plan_anchor_delta.clone().flatten(1, 2).unsqueeze(1) # [B, 1, ME, T2, 2]
                    plan_reg = plan_reg + plan_delta
                motion_classification.append(motion_cls) # [B, 900, M]
                motion_prediction.append(motion_reg) # [B, 900, M1, T1, 2]
                planning_classification.append(plan_cls) # [B, 1, C]
                planning_prediction.append(plan_reg) # [B, 1, M2, T2, 2]
                planning_status.append(plan_status) # [B, 1, 10]
                planning_feature.append(plan_query) # [B, 1, C * M, D]
        self.instance_queue.cache_motion(agent_mask.any(dim=-1), metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            'mask':~instance_padding_mask[:,:self.num_det],
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "feature": planning_feature,
            "encoder_feature": ego_feature_memory,
            "decoder_feature": planning_memory_feature,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output


    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        agnet_mask = model_outs['mask']
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                agnet_mask,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                status_pred,
            ) = self.planning_sampler.sample(
                cls,
                reg,
                status,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)

            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )

            status_loss = self.plan_loss_status(status_pred.squeeze(1), data['ego_status'])
            # reinforce learning loss
            if self.dqn_model is not None:
                planning_loss_rl = self.dqn_model(data['ego_status'], status_pred.squeeze(1), reg_target, reg_pred, data['gt_ego_fut_cmd'])
            else:
                planning_loss_rl = torch.tensor(0.0, device=status_loss.device)

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                    f"planning_loss_rl_{decoder_idx}": planning_loss_rl,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            data['gt_bboxes_3d'],
            data['gt_labels_3d'],
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            data['gt_bboxes_3d'],
            motion_output,
            planning_output, 
            data,
        )

        return motion_result, planning_result