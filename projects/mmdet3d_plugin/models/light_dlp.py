from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_head,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["LightDLP"]


@DETECTORS.register_module()
class LightDLP(BaseDetector):
    def __init__(
        self,
        head,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_deformable_func=False,
    ):
        super(LightDLP, self).__init__(init_cfg=init_cfg)
        self.head = build_head(head)
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func

    
    @force_fp32(apply_to=("img",))
    def forward(self, **data):
        if self.training:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self, **data):
        model_outs = self.head(data)
        output = self.head.loss(model_outs, data)
        return output

    def forward_test(self, **data):
        return self.simple_test(**data)

    def simple_test(self, **data):
        model_outs = self.head(data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        return None