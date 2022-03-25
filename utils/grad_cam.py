from typing import Callable, Iterable, Tuple
import torch
import numpy as np
import PIL.Image
import cv2
import wandb

from tqdm import tqdm
from pytorch_grad_cam import GradCAM

from utils.val_loop_hook import ValidationLoopHook

def _get_grad_cam_target(model):
    """
    Determines the appropriate GradCAM target.
    """

    # very naive check
    if hasattr(model, "features"):
        return getattr(model, "features")

    pooling = [torch.nn.AdaptiveAvgPool1d, torch.nn.AvgPool1d, torch.nn.MaxPool1d, torch.nn.AdaptiveMaxPool1d,
               torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d, torch.nn.MaxPool2d, torch.nn.AdaptiveMaxPool2d,
               torch.nn.AdaptiveAvgPool3d, torch.nn.AvgPool3d, torch.nn.MaxPool3d, torch.nn.AdaptiveMaxPool3d]
    convolutions = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d]

    # reverse search starting from the final module
    inverted_modules = list(model.modules())[::-1]
    for i, module in enumerate(inverted_modules):
        if any([isinstance(module, po) for po in pooling]):
            # if a pooling layer was hit, pick the module directly before it
            return inverted_modules[i+1]
        elif any([isinstance(module, co) for co in convolutions]):
            # if a convolution was hit (but no pooling layer), pick that one instead
            return module
        elif isinstance(module, torch.nn.Sequential):
            # if a sequential module is hit, explore it
            for child in list(module.children())[::-1]:
                sequential_result = _get_grad_cam_target(child)
                if sequential_result is not None:
                    return sequential_result

def _show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """ This function overlays the cam mask on the image as an heatmap.
        By default the heatmap is in BGR format.
        :param img: The base image in RGB or BGR format.
        :param mask: The cam mask.
        :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        :param colormap: The OpenCV colormap to be used.
        :returns: The default image with the cam overlay.
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        normalize = lambda x: (x - np.min(x))/np.ptp(x)

        cam = 0.6 * heatmap + normalize(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

def _strip_image_from_grid_row(row, gap=5, bg=255):
    strip = torch.full(
            (row.shape[0] * (row.shape[3] + gap) - gap,
             row.shape[1] * (row.shape[3] + gap) - gap,
             row.shape[4]), bg, dtype=row.dtype)
    for i in range(0, row.shape[0] * row.shape[1]):
        strip[(i // row.shape[1]) * (row.shape[2] + gap) : ((i // row.shape[1])+1) * (row.shape[2] + gap) - gap,
              (i % row.shape[1]) * (row.shape[3] + gap) : ((i % row.shape[1])+1) * (row.shape[3] + gap) - gap,
              :] = row[i // row.shape[1]][i % row.shape[1]]
    return PIL.Image.fromarray(strip.numpy())


class GradCAMBuilder(ValidationLoopHook):
    def __init__(self, image_shape: Iterable[int], target_category: int = None, num_images: int = 5):
        self.image_shape = image_shape
        self.target_category = target_category
        self.num_images = num_images

        self.targets = torch.zeros(self.num_images)
        self.activations = torch.zeros(self.num_images)
        self.images = torch.zeros(torch.Size([self.num_images]) + torch.Size(self.image_shape))

    def process(self, batch, target_batch, logits_batch, prediction_batch):
        image_batch = batch["image"]

        with torch.no_grad():
            if self.target_category is not None:
                local_activations = logits_batch[:, self.target_category]
            else:
                local_activations = torch.amax(logits_batch, dim=-1)

        # filter samples where the prediction lines up with the target
        target_match = (prediction_batch == target_batch)

        # filter public dataset samples
        public = torch.tensor(["verse" in id for id in batch["verse_id"]]).type_as(target_match)

        mask = target_match & public

        if torch.max(mask) == False:
            # no samples match criteria in this batch, skip
            return

        # identify better activations and replace them accordingly
        local_top_idx = torch.argsort(local_activations, descending=True)
        # filter samples
        local_top_idx = local_top_idx[mask[local_top_idx]]
        current_idx = 0

        while current_idx < self.num_images and local_activations[local_top_idx[current_idx]] > torch.min(self.activations):
            # next item in local batch matches criteria and has a higher activation than one in the global batch, replace it
            idx_to_replace = torch.argsort(self.activations)[0]

            self.activations[idx_to_replace] = local_activations[local_top_idx[current_idx]]
            self.images[idx_to_replace] = image_batch[local_top_idx[current_idx]]
            self.targets[idx_to_replace] = target_batch[local_top_idx[current_idx]]

            current_idx += 1

    def trigger(self, module):
        model = module.backbone
        module.eval()

        # determine the Grad-CAM target module/layer
        grad_cam_target = _get_grad_cam_target(model)

        cam = GradCAM(model, [grad_cam_target], use_cuda=torch.cuda.is_available())

        # determine final order such that the highest activations are placed on top
        sorted_idx = torch.argsort(self.activations, descending=True)

        self.activations = self.activations[sorted_idx]
        self.images = self.images[sorted_idx]
        self.targets = self.targets[sorted_idx]

        # if a polyaxon experiment crashes here, remove the GradCAMBuilder instance from the
        # model.validation_hooks list
        grad_cams = cam(input_tensor=self.images, target_category=self.target_category)

        module.train()

        if len(self.images.shape) == 5:
            # 3D, visualize slices
            ld_res = grad_cams.shape[-1]
            img_res = self.images.shape[-1]
            img_slices = torch.linspace(int(img_res/ld_res/2), img_res-int(img_res/ld_res/2), ld_res, dtype=torch.long)
            
            # Show all images slices in a larger combined image
            grad_cams_image = _strip_image_from_grid_row(
                torch.stack([
                    torch.stack([
                        torch.tensor(
                            _show_cam_on_image((self.images[i, 0, ..., img_slices[s]]).unsqueeze(-1).repeat(1, 1, 3).numpy(), grad_cams[i, ..., s], use_rgb=True)
                        )
                    for s in range(grad_cams.shape[-1])])
                for i in range(self.num_images if self.num_images < grad_cams.shape[0] else grad_cams.shape[0])])
            )

        elif len(self.images.shape) == 4:
            # 2D
            grad_cams_image = _strip_image_from_grid_row(
                torch.stack([
                    torch.stack([
                        torch.tensor(
                            _show_cam_on_image((self.images[i, 0, ...]).unsqueeze(-1).repeat(1, 1, 3).numpy(), grad_cams[i, ...], use_rgb=True)
                        )
                    ])
                for i in range(self.num_images if self.num_images < grad_cams.shape[0] else grad_cams.shape[0])])
            )

        else:
            raise RuntimeError("Attempting to build Grad-CAMs for data that is neither 2D nor 3D")

        module.logger.experiment.log({
            "val/grad_cam": wandb.Image(grad_cams_image)
        })

    def reset(self):
        self.targets = torch.zeros(self.num_images)
        self.activations = torch.zeros(self.num_images)
        self.images = torch.zeros(torch.Size([self.num_images]) + torch.Size(self.image_shape))