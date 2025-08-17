from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head


@HEADS.register_module()
class LightDLPHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        motion_plan_head = dict,
        init_cfg=None,
        **kwargs,
    ):
        super(LightDLPHead, self).__init__(init_cfg)
        self.task_config = task_config
        if self.task_config['with_motion_plan']:
            self.planning_head = build_head(motion_plan_head)

    def init_weights(self):
        if self.task_config['with_motion_plan']:
            self.planning_head.init_weights()

    def forward(
        self,
        metas: dict,
    ):
        if self.task_config['with_motion_plan']:
            motion_output, planning_output = self.planning_head(
                metas,
            )
        else:
            motion_output, planning_output = None, None

        return motion_output, planning_output

    def loss(self, model_outs, data):
        motion_output, planning_output = model_outs
        losses = dict()
        if self.task_config['with_motion_plan']:
            loss_motion = self.planning_head.loss(
                motion_output, 
                planning_output, 
                data, 
            )
            losses.update(loss_motion)
        return losses

    def post_process(self, model_outs, data):
        motion_output, planning_output = model_outs
        if self.task_config['with_motion_plan']:
            motion_result, planning_result = self.planning_head.post_process(
                motion_output, 
                planning_output,
                data,
            )
        batch_size = len(motion_result)
        results = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config['with_motion_plan']:
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])

        return results