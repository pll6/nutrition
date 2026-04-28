# Copyright (c) OpenMMLab. All rights reserved.
import warnings  # noqa: F401,F403

from mmcv.utils import Registry

MASK_ASSIGNERS = Registry('mask_assigner')
MATCH_COST = Registry('match_cost')
TEXT_ENCODERS = Registry('text_encoder')

def build_assigner(cfg):
    """Build Assigner."""
    return MASK_ASSIGNERS.build(cfg)

def build_match_cost(cfg):
    """Build Match Cost."""
    return MATCH_COST.build(cfg)

def build_text_encoder(cfg):
    """Build Text Encoder."""
    return TEXT_ENCODERS.build(cfg)