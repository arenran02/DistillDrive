import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pickle
from mmdet.models import LOSSES

@LOSSES.register_module()
class KLLoss(nn.Module):
    """
        kl-loss for the prediction of confidence
    """
    def __init__(self, loss_type='KLLoss', loss_weight=1):
        super(KLLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_weight = loss_weight

    def forward(self, cls_pred, cls_target, weight):
        '''
            cls_pred: [B, M]
            cls_target: [B, M]
            weight: [B]
        '''
        batch_mask = weight.bool()
        assert cls_pred.shape == cls_target.shape, "When use KL Loss, must keep the same dimension"
        valid_batch_size = sum(batch_mask).item()
        cls_pred = nn.Softmax(dim=1)(cls_pred)
        kl_divergence = F.kl_div(cls_pred.log(), cls_target, reduction='none')
        cls_loss = torch.sum(kl_divergence[batch_mask]) / (valid_batch_size if valid_batch_size > 1 else 1) * self.loss_weight
        return cls_loss