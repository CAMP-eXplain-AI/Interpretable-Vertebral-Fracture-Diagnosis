import numpy as np
from typing import List, Tuple
from monai.transforms import Compose, AddChannelD, MaskIntensityD, DeleteItemsD, CropForegroundD, ResizeD

class SelectMaskByLevelD:
  """
  Selects a mask segment from a mask image based on a given level index. May also
  be applied to a single channel.
  """
  def __init__(self, mask_key: str, level_idx_key: str):
    self.mask_key = mask_key
    self.level_idx_key = level_idx_key

  def __call__(self, data):
    d = dict(data)
    mask = np.zeros_like(d[self.mask_key])
    mask[d[self.mask_key] == d[self.level_idx_key]] = 1
    d[self.mask_key] = mask
    return d

def get_mask_transform(hparams, loaded_keys: List[str], level_idx_key='level_idx') -> Tuple[Compose, List[str]]:
    """
    Depending on the configuration values for 'MASK', the transform returned by this method does one of the following:
        - nothing ('none')
        - applies the mask of the critical vertebra to the image ('apply')
        - applies the mask of all visible vertebrae to the image ('apply_all')
        - loads the mask into the 'mask' key s.t. it will later be stacked with the image ('channel')
        - crop the image to the critical vertebra and upsample it ('crop')
    """

    if hparams.mask == 'none':
        return Compose([]), loaded_keys

    assert len(loaded_keys) == 2
    image_key, mask_key = loaded_keys

    if hparams.mask == 'apply':
        return Compose([
            # only select relevant vertebra
            SelectMaskByLevelD(mask_key=mask_key, level_idx_key=level_idx_key),
            # apply mask
            MaskIntensityD(keys=image_key, mask_key=mask_key),
            # once the mask is applied, release it
            DeleteItemsD(keys=mask_key),
        ]), [image_key]

    elif hparams.mask == 'apply_all':
        return Compose([
            # keeps all vertebra in the mask
            # apply mask
            MaskIntensityD(keys=image_key, mask_key=mask_key),
            # once the mask is applied, release it
            DeleteItemsD(keys=mask_key),
        ]), [image_key]

    elif hparams.mask == 'channel':
        return Compose([
            SelectMaskByLevelD(mask_key=mask_key, level_idx_key=level_idx_key),
        ]), loaded_keys

    elif hparams.mask == 'crop':
        # TODO CropForegroundD ignores one spatial dimension, thus not truly cropping
        return Compose([
            SelectMaskByLevelD(mask_key=mask_key, level_idx_key=level_idx_key),
            CropForegroundD(keys=image_key, source_key=mask_key, margin=2),
            DeleteItemsD(keys=mask_key),
            AddChannelD(keys=image_key),
            ResizeD(keys=image_key, spatial_size=[hparams.input_size] * hparams.input_dim, mode='trilinear'),
        ]), [image_key]