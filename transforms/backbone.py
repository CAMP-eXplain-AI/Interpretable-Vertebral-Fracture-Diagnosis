from typing import List, Tuple

from monai.transforms import Compose, CenterSpatialCropD, ResizeD, NormalizeIntensityD, ScaleIntensityRangeD, adaptor
from monai.transforms.utils_pytorch_numpy_unification import clip


class HUClipTransform:
    """Clips image to the desired HU range."""
    def __init__(self, range: Tuple[int, int]):
        self.range = range

    def __call__(self, image):
        return clip(image, *self.range)

def models_genesis_transform(hparams, loaded_keys) -> Tuple[Compose, List[str]]:
    """
    Transforms image to match the normalization used in Models Genesis pre-training:
        - Clip HU to [-1000, 1000]
        - Scale to [0, 1]

    See: https://github.com/MrGiovanni/ModelsGenesis/tree/master/pytorch#3-fine-tune-models-genesis-on-your-own-target-task
    """

    assert len(loaded_keys) > 0
    image_key = loaded_keys[0]

    return Compose([
            adaptor(HUClipTransform((-1000, 1000)), image_key),
            ScaleIntensityRangeD(
                keys=image_key,
                a_min=-1000,
                a_max=1000,
                b_min=0,
                b_max=1,
                clip=True
            )
        ]), loaded_keys
