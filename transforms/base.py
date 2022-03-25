import torch
import numpy as np

from pathlib import Path
from typing import List, Optional, Tuple
from monai.transforms import Compose, LoadImageD, LambdaD, AddChannelD, CenterSpatialCropD

class DuplicateKeyD:
  """
  Duplicates a key in the given data dictionary.
  """
  def __init__(self, from_key: str, to_key: str):
    self.from_key = from_key
    self.to_key = to_key

  def __call__(self, data):
    d = dict(data)
    return {**d, self.to_key: d[self.from_key]}


class MergeKeysD:
  """
  Stacks images from multiple keys into a single one.
  """
  def __init__(self, keys: List[str], to_key: str):
    self.keys = keys
    self.to_key = to_key

  def __call__(self, data):
    d = dict(data)
    
    if len(self.keys) == 1 and self.keys[0] == self.to_key:
        # nothing to stack or move
        return d

    arrays_to_stack = [d.pop(key) for key in self.keys]

    if isinstance(arrays_to_stack[0], torch.Tensor):
        d[self.to_key] = torch.cat(arrays_to_stack, dim=0)
    else:
        d[self.to_key] = np.concatenate(arrays_to_stack, axis=0)

    return d


def get_image_loading_transform(hparams, image_dir: Path, mask_dir: Optional[Path] = None) -> Tuple[Compose, List[str]]:
    """
    Loads an image and, depending on the configuration, its corresponding mask.
    """

    if hparams.mask == 'none':
        return Compose([
            LambdaD(keys='image', func=lambda p: image_dir / p),
            LoadImageD(keys='image'),
        ]), ['image']

    else:
        assert mask_dir is not None
        return Compose([
            DuplicateKeyD(from_key='image', to_key='mask'),
            # load image and mask
            LambdaD(keys='image', func=lambda p: image_dir / p),
            LoadImageD(keys='image'),
            LambdaD(keys='mask', func=lambda p: mask_dir / p),
            LoadImageD(keys='mask'),
        ]), ['image', 'mask']

def get_apply_crop_transform(hparams, loaded_keys: List[str]) -> Tuple[Compose, List[str]]:
    """
    Applies a crop to the loaded keys to achieve the desired size, if appropriate for the given configuration.
    """

    if hparams.mask == 'crop':
        return Compose([]), loaded_keys
    else:
        return Compose([
            AddChannelD(keys=loaded_keys),
            CenterSpatialCropD(keys=loaded_keys, roi_size=[hparams.input_size] * hparams.input_dim),
        ]), loaded_keys

def get_stacking_transform(hparams, loaded_keys: List[str]) -> Tuple[Compose, List[str]]:
    """
    Stacks multiple loaded keys (i.e. image and mask) as channels into the single 'image' key. If a single key
    is loaded, do nothing.
    """

    if len(loaded_keys) > 1:
        return Compose([
            MergeKeysD(loaded_keys, 'image')
        ]), ['image']
    else:
        return Compose([]), loaded_keys