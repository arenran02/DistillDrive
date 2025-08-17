import torch
import math
from mmdet.core.bbox.builder import BBOX_SAMPLERS

__all__ = ["MotionTarget", "PlanningTarget"]


def get_cls_target(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2) # [B, 1, M ,T, 2]
    reg_target_cum = reg_target.cumsum(dim=-2) # [B, 1, T, 2]
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1) # [B, 1, M, T]
    dist = dist * reg_weight.unsqueeze(2) # [B, 1, M, T]
    dist = dist.mean(dim=-1)  # [B, 1, M]
    mode_idx = torch.argmin(dist, dim=-1) # [B, 1]
    return mode_idx

def get_cls_target_topk(
    reg_preds, 
    reg_target,
    reg_weight,
    cls_target,
    topk_num
):
    conf_tolerance = 0.05
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2) # [B, 1, M ,T, 2]
    reg_target_cum = reg_target.cumsum(dim=-2) # [B, 1, T, 2]
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1) # [B, 1, M, T]

    dist = dist * reg_weight.unsqueeze(2) # [B, 1, M, T]
    dist = dist.mean(dim=-1)  # [B, 1, M]
    _, min_idx = torch.topk(dist, k=topk_num, largest=False)
    B, _, M = min_idx.shape
    gt_conf = (torch.zeros(cls_target.shape, device=min_idx.device) + 1e-10)
    max_conf = (1 - (M-1) * conf_tolerance) if M > 1 else 1

    gt_conf[torch.arange(0, B), :, min_idx[:, 0, 0]] = max_conf

    if M > 1:
        gt_conf.scatter_(-1, min_idx[:, :, 1:], conf_tolerance)

    return gt_conf


def get_best_reg(
    reg_preds, 
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
    best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
    return best_reg


@BBOX_SAMPLERS.register_module()
class AgentTarget():
    def __init__(
        self,
    ):
        super(AgentTarget, self).__init__()

    def sample(
        self,
        reg_pred,
        agent_mask,
        gt_reg_target,
        gt_reg_mask,
    ):
        bs, num_anchor, mode, ts, d = reg_pred.shape
        reg_target = reg_pred.new_zeros((bs, num_anchor, ts, d))
        reg_weight = reg_pred.new_zeros((bs, num_anchor, ts))
        num_pos = reg_pred.new_tensor([0])
        for i in range(bs):
            valid_agent_num = sum(agent_mask[i])
            # some unavailabel target in ground truth, so we need fillter
            gt_valid_mask = agent_mask[i][: gt_reg_target[i].shape[0]]
            if valid_agent_num == 0:
                continue

            reg_target[i][agent_mask[i]] = gt_reg_target[i][gt_valid_mask][:, :ts, :]
            reg_weight[i][agent_mask[i]] = gt_reg_mask[i][gt_valid_mask][:, :ts]
            num_pos += valid_agent_num

        cls_target = get_cls_target(reg_pred, reg_target, reg_weight)
        cls_weight = reg_weight.any(dim=-1)
        best_reg = get_best_reg(reg_pred, reg_target, reg_weight)

        return cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos


@BBOX_SAMPLERS.register_module()
class EgoTarget():
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        gt_result='label',
        multi_ego_status=False,
    ):
        super(EgoTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.topk_num = math.ceil(ego_fut_mode / 2) if math.ceil(ego_fut_mode / 2) <= 6 else 6
        self.gt_result = gt_result
        self.multi_ego_status = multi_ego_status

    def sample(
        self,
        cls_pred,
        reg_pred,
        status_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1) # [B, 1, T, 2]
        gt_reg_mask = gt_reg_mask.unsqueeze(1) # [B, 1, T]

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)

        cls_pred = cls_pred.reshape(bs, 3, 1, self.ego_fut_mode) # [B, C, 1, M]
        reg_pred = reg_pred.reshape(bs, 3, 1, self.ego_fut_mode, self.ego_fut_ts, 2) # [B, C, 1, M, T, 2]
        cls_pred = cls_pred[bs_indices, cmd] # [B, 1, M]
        reg_pred = reg_pred[bs_indices, cmd] # [B, 1, M, T, 2]
        if self.gt_result == 'label':
            cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask) # [B, 1]
            mode_idx = cls_target.clone().squeeze()
        elif self.gt_result == 'distribution':
            cls_target = get_cls_target_topk(reg_pred, gt_reg_target, gt_reg_mask, cls_pred, self.topk_num) # [B, 1, M]
            mode_idx = torch.argmax(cls_target, dim=2).squeeze()
        # prepare ground truth for ego status
        if self.multi_ego_status:
            status_pred = status_pred.reshape(bs, 3, 1, self.ego_fut_mode, 10)[bs_indices, cmd, :, mode_idx, :]
        else:
            status_pred = status_pred
        cls_weight = gt_reg_mask.any(dim=-1) # [B, 1]
        best_reg = get_best_reg(reg_pred, gt_reg_target, gt_reg_mask)

        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask, status_pred