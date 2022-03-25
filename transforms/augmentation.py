import math
from typing import List, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from monai.transforms import Compose, OneOf
from monai.transforms import RandGaussianNoiseD, RandAdjustContrastD, RandGaussianSmoothD
from monai.transforms import RandFlipD, Rand3DElasticD, RandRotateD, RandZoomD, RandAffineD

def intensity_transform(hparams, loaded_keys) -> Tuple[Compose, List[str]]:
    assert len(loaded_keys) > 0
    image_key = loaded_keys[0]

    return OneOf([
            RandGaussianNoiseD(keys=image_key),
            RandAdjustContrastD(
                keys=image_key,
                gamma=1.5
            ),
            RandGaussianSmoothD(
                # gaussian smoothing is also applied to mask
                keys=loaded_keys,
                sigma_x=(0.25, 0.75),
                sigma_y=(0.25, 0.75),
                sigma_z=(0.25, 0.75)
            ),
    ]), loaded_keys

def robustness_transform(hparams, loaded_keys) -> Tuple[Compose, List[str]]:
    keys_to_transform = loaded_keys
    return Compose([
            RandAffineD(
                keys=keys_to_transform,
                translate_range=(((-20, 10), 5, 5)), # permits keypoint misplacing as well as minor centering deviations
                padding_mode="zeros",
                prob=1 # elevated probability
            ),
            RandRotateD(
                keys=keys_to_transform,
                range_x=math.radians(7), # spine roll, patient is not lying perfectly flat on their back
                range_y=math.radians(5), # spine yaw, patient position is slightly skewed
                range_z=math.radians(20), # spine pitch, strong deviation permitted
                prob=1 # elevated probability
            ),
    ]), loaded_keys

def spatial_transform(hparams, loaded_keys, mode: Literal['simple', 'default'] = "default") -> Tuple[Compose, List[str]]:
    keys_to_transform = loaded_keys
    transforms = [
            RandFlipD(
                keys=keys_to_transform,
                spatial_axis=2 # only flip the vertebrae
            ),
            RandRotateD(
                keys=keys_to_transform,
                range_x=math.radians(7), # spine roll, patient is not lying perfectly flat on their back
                range_y=math.radians(5), # spine yaw, patient position is slightly skewed
                range_z=math.radians(2), # spine pitch, patient's spine has a slight slope
                                         # (may simulate kyphosis but resulting images are questionable
                                         #  for larger rotation angles)
                prob=0.1 if not "robustness" in hparams.transforms else 0 # prevent attempts to re-rotate
            ),
            RandZoomD(
                keys=keys_to_transform,
                mode='nearest',#['area', 'nearest'],
                min_zoom=0.9, max_zoom=1.1,
            ),
    ]

    if mode == 'simple':
        transforms.append( 
            Rand3DElasticD(
                sigma_range=(1, 2),
                magnitude_range=(2, 8),
                keys=keys_to_transform
            )
        )

    return Compose(transforms), loaded_keys