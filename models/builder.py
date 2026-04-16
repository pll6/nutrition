# Copyright (c) OpenMMLab. All rights reserved.
import warnings  # noqa: F401,F403

from mmcv.utils import Registry

MASK_ASSIGNERS = Registry('mask_assigner')

def build_assigner(cfg):
    """Build Assigner."""
    return MASK_ASSIGNERS.build(cfg)
