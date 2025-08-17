from typing import Optional

import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.detection3d.decoder import *
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners


@BBOX_CODERS.register_module()
class AgentDecoder(SparseBox3DDecoder):
    def __init__(self):
        super(AgentDecoder, self).__init__()

    def decode(
        self,
        gt_target,
        gt_label,
        motion_output,
    ):
        
        bs = len(gt_target)
        output = []
        bbox = gt_target[0]
        label = gt_label[0]
        traj = motion_output["prediction"][0][0]
        traj_cls = motion_output["classification"][0][0].sigmoid()
        traj_mask = motion_output['mask'].squeeze(0)
        agent_traj = traj[traj_mask] # [N1, M, T, 2]
        agent_cls = traj_cls[traj_mask] # [N1, M]
        agent_bbox = bbox[traj_mask[:bbox.shape[0]]] # [N1, 9]
        agnet_label = label[traj_mask[:bbox.shape[0]]]
        agent_traj = agent_traj.cumsum(dim=-2) + agent_bbox[:, None, None, :2]

        output.append(
            {
                "boxes_3d": agent_bbox.cpu(),
                "labels_3d": agnet_label.cpu(),
                "scores_3d": torch.ones(agnet_label.shape).float(),
                "gt_bboxes_3d": agent_bbox.cpu(),  # N, 9
                "gt_labels_3d": agnet_label.cpu(), # N
                "trajs_3d": agent_traj.cpu(), # N, M, T, 2
                "trajs_score": agent_cls.cpu() # N, M
            }
        )
        
        return output


@BBOX_CODERS.register_module()
class EgoDecoder(object):
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        use_rescore=False,
    ):
        super(EgoDecoder, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.use_rescore = use_rescore
    
    def decode(
        self, 
        gt_target,
        motion_output,
        planning_output, 
        data,
    ):
        classification = planning_output['classification'][-1]
        prediction = planning_output['prediction'][-1]
        bs = classification.shape[0]
        classification = classification.reshape(bs, 3, self.ego_fut_mode)
        prediction = prediction.reshape(bs, 3, self.ego_fut_mode, self.ego_fut_ts, 2).cumsum(dim=-2)
        classification, final_planning = self.select(gt_target, motion_output, classification, prediction, data)
        output = []

        for i, (cls, pred) in enumerate(zip(classification, prediction)):
            output.append(
                {
                    "planning_score": cls.sigmoid().cpu(),
                    "planning": pred.cpu(),
                    "final_planning": final_planning[i].cpu(),
                }
            )

        return output

    def select(
        self,
        gt_target,
        motion_output,
        plan_cls,
        plan_reg,
        data,
    ):
        # target
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1]
        motion_mask = motion_output['mask']
        
        # cmd select
        bs = motion_cls.shape[0]
        bs_indices = torch.arange(bs, device=motion_cls.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        plan_cls_full = plan_cls.detach().clone()
        plan_cls = plan_cls[bs_indices, cmd]
        plan_reg = plan_reg[bs_indices, cmd]

        # rescore
        if self.use_rescore:
            plan_cls = self.rescore(
                plan_cls,
                plan_reg, 
                motion_cls,
                motion_reg, 
                motion_mask,
                gt_target[0],
            )
        plan_cls_full[bs_indices, cmd] = plan_cls
        mode_idx = plan_cls.argmax(dim=-1)
        final_planning = plan_reg[bs_indices, mode_idx]
        return plan_cls_full, final_planning

    def rescore(
        self, 
        plan_cls,
        plan_reg, 
        motion_cls,
        motion_reg, 
        motion_mask,
        gt_target,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):
        
        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat
        
        def get_yaw(traj, start_yaw=np.pi/2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            # for static object, estimated future yaw would be unstable
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(
                mask.unsqueeze(-1),
                start_yaw,
                yaw,
            )
            return yaw.unsqueeze(-1)
        
        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = gt_target.new_zeros(bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[motion_mask] # [N, M, T, 2]
        motion_cls = motion_cls[motion_mask] # [N, M]
        gt_target = gt_target.unsqueeze(0)[motion_mask[:,:gt_target.shape[0]]]
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2) # [N, M, T, 2]
        motion_reg = cat_with_zero(motion_reg) + gt_target[:, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, self.ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)
        
        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,)) # [B, 1, T+1, 7]
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = gt_target[..., None, None, [W, L, H]]
        box_yaw = gt_target[:, YAW]
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]
        motion_box = motion_box.unsqueeze(0)
        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        if num_anchor == 0 or self.ego_fut_mode == 1:
            return plan_cls
        
        else:
            ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
            motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)
            ego_box[0] += offset * torch.cos(ego_box[6])
            ego_box[1] += offset * torch.sin(ego_box[6])
            col = check_collision(ego_box, motion_box) # [B * N * M1 * M2 * T]
            col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4) # [B, M1, N, M2, T]
            col = col.flatten(2, -1).any(dim=-1) # [B, M1]
            all_col = col.all(dim=-1) # [1]
            col[all_col] = False # for case that all modes collide, no need to rescore
            score_offset = col.float() * -999
            plan_cls = plan_cls + score_offset
            return plan_cls


def check_collision(boxes1, boxes2):
    '''
        A rough check for collision detection: 
            check if any corner point of boxes1 is inside boxes2 and vice versa.
        
        boxes1: tensor with shape [N, 7], [x, y, z, w, l, h, yaw]
        boxes2: tensor with shape [N, 7]
    '''
    col_1 = corners_in_box(boxes1.clone(), boxes2.clone())
    col_2 = corners_in_box(boxes2.clone(), boxes1.clone())
    collision = torch.logical_or(col_1, col_2)

    return collision

def corners_in_box(boxes1, boxes2):
    if  boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return False

    boxes1_yaw = boxes1[:, 6].clone()
    boxes1_loc = boxes1[:, :3].clone()
    cos_yaw = torch.cos(-boxes1_yaw)
    sin_yaw = torch.sin(-boxes1_yaw)
    rot_mat_T = torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw]),
            torch.stack([-sin_yaw, cos_yaw]),
        ]
    )
    # translate and rotate boxes
    boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
    boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
    boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw

    boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
    boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
    boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw

    corners_box2 = box3d_to_corners(boxes2)[:, [0, 3, 7, 4], :2]
    corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device)
    H = boxes1[:, [3]]
    W = boxes1[:, [4]]

    collision = torch.logical_and(
        torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0] >= -H / 2),
        torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2),
    )
    collision = collision.any(dim=-1)

    return collision