# Copyright (c) OpenMMLab. All rights reserved.
from .maskdecoder import MLPMaskDecoder
from .regression_head import MultiScaleNutritionHead

__all__ = [
    'MLPMaskDecoder', 'MultiScaleNutritionHead'
]
