from pathlib import Path
from typing import Callable, List, Optional, Tuple

from monai.transforms import Compose

from transforms.base import get_image_loading_transform, get_apply_crop_transform, get_stacking_transform
from transforms.mask import get_mask_transform
from transforms.augmentation import *
from transforms.backbone import *


def _build_transforms_composition(hparams, transform_getters: List[Callable], *initial_args) -> Tuple[Compose, List[str]]:
    """
    Builds a transforms composition from the given functions, which take the hparams and loaded keys as arguments, and
    produce a Compose containing the desired transforms. The initialization function receives the provided initial arguments.
    """
    transforms = []
    keys = []

    for i in range(0, len(transform_getters)):
        if len(keys) == 0:
            assert i == 0, f"Function {transform_getters[i]} did not yield any loaded keys."
            # initialize
            transform, keys = transform_getters[0](hparams, *initial_args)
        else:
            transform, keys = transform_getters[i](hparams, keys)
        transforms.append(transform)

    return Compose(transforms), keys

def _get_config_transform_by_name(transform_name: str) -> Callable:
    if transform_name == "intensity":
        return intensity_transform
    elif transform_name.startswith("spatial3d"):
        if "simple" in transform_name:
            return lambda hparams, loaded_keys: spatial_transform(hparams, loaded_keys, mode='simple')
        else:
            return lambda hparams, loaded_keys: spatial_transform(hparams, loaded_keys, mode='default')
    elif transform_name == "modelsgenesis":
        return models_genesis_transform
    elif transform_name == "robustness":
        return robustness_transform
    else:
        raise ValueError(f"Unknown transform: {transform_name}")

def get_training_transforms(hparams, image_dir: Path, mask_dir: Optional[Path] = None) -> Compose:
    transforms_base = [get_image_loading_transform, get_mask_transform]

    # robustness has to run early as we may need to operate on the whole volume for affine transformation and padding, 
    # which must occur prior to any cropping or normalization
    if "robustness" in hparams.transforms: transforms_base.append(_get_config_transform_by_name("robustness"))

    transforms_base.extend([get_apply_crop_transform])

    # preprocessing transforms must be run first
    preprocessing_transforms = ["modelsgenesis"]
    config_transforms = [_get_config_transform_by_name(transform_name) for transform_name in hparams.transforms if transform_name in preprocessing_transforms]
    
    # then append the rest minus the robustness transform that is run earlier
    exclusion_criterion = lambda transform_name: transform_name in preprocessing_transforms or transform_name == "robustness"
    config_transforms.extend([_get_config_transform_by_name(transform_name) for transform_name in hparams.transforms if not exclusion_criterion])

    # the stacking transform must not occur before config transforms are run to avoid any interference
    return _build_transforms_composition(hparams, transforms_base + config_transforms + [get_stacking_transform], image_dir, mask_dir)[0]

def get_base_transforms(hparams, image_dir: Path, mask_dir: Optional[Path] = None) -> Compose:
    transforms_base = [get_image_loading_transform, get_mask_transform, get_apply_crop_transform]

    # apply preprocessing transforms
    preprocessing_transforms = ["modelsgenesis"]
    config_transforms = [_get_config_transform_by_name(transform_name) for transform_name in hparams.transforms if transform_name in preprocessing_transforms]

    return _build_transforms_composition(hparams, transforms_base + config_transforms + [get_stacking_transform], image_dir, mask_dir)[0]