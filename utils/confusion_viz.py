from threading import local
import torch
import wandb
import numpy as np
import PIL.Image
from typing import Iterable

from utils.val_loop_hook import ValidationLoopHook

def _strip_image_from_grid_row(row, gap=5, bg=255):
    strip = torch.full(
            (row.shape[0] * (row.shape[3] + gap) - gap,
            row.shape[1] * (row.shape[3] + gap) - gap), bg, dtype=row.dtype)
    for i in range(0, row.shape[0] * row.shape[1]):
        strip[(i // row.shape[1]) * (row.shape[2] + gap) : ((i // row.shape[1])+1) * (row.shape[2] + gap) - gap,
            (i % row.shape[1]) * (row.shape[3] + gap) : ((i % row.shape[1])+1) * (row.shape[3] + gap) - gap] = row[i // row.shape[1]][i % row.shape[1]]
    return PIL.Image.fromarray(strip.numpy())

class ConfusionVisualizer(ValidationLoopHook):
    def __init__(self, image_shape: Iterable[int], num_classes: int, num_images: int = 5, num_slices: int = 8):
        self.image_shape = image_shape
        self.num_images = num_images
        self.num_classes = num_classes
        self.num_slices = num_slices

        self.activations = -99 * torch.ones(self.num_classes, self.num_images)
        self.images = torch.zeros(torch.Size([self.num_classes, self.num_images]) + torch.Size(self.image_shape))

    def process(self, batch, target_batch, logits_batch, prediction_batch):
        image_batch = batch["image"]

        with torch.no_grad():
            local_activations = torch.amax(logits_batch, dim=-1)

        # filter samples where the prediction does not line up with the target
        confused_samples = (prediction_batch != target_batch)

        # filter public dataset samples
        public = torch.tensor(["verse" in id for id in batch["verse_id"]]).type_as(confused_samples)

        mask = confused_samples & public

        for current_idx in torch.nonzero(mask).squeeze(1):
            target_class = target_batch[current_idx]
            # next item in local batch has a higher activation than the previous confusions for this class, replace it
            if local_activations[current_idx] > torch.min(self.activations[target_class]):
                idx_to_replace = torch.argsort(self.activations[target_class])[0]
                self.activations[target_class, idx_to_replace] = local_activations[current_idx]
                self.images[target_class, idx_to_replace] = image_batch[current_idx].cpu()

    def trigger(self, module):
        for class_idx in range(self.num_classes):
            # determine final order such that the highest activations are placed on top
            sorted_idx = torch.argsort(self.activations[class_idx], descending=True)

            self.images[class_idx] = self.images[class_idx, sorted_idx]

            normalize = lambda x: (x - np.min(x))/np.ptp(x)

            if len(self.images.shape) == 6:
                # 3D, visualize slices
                img_res = self.images[class_idx].shape[-1]
                img_slices = torch.linspace(0, img_res-1, self.num_slices+2, dtype=torch.long)[1:-1]
                
                # Show all images slices in a larger combined image
                top_confusing_samples = _strip_image_from_grid_row(
                    torch.stack([
                        torch.stack([
                            torch.tensor(
                                np.uint8(255 * normalize((self.images[class_idx, i, 0, ..., img_slices[s]]).numpy()))
                            )
                        for s in range(self.num_slices)])
                    for i in range(self.num_images if self.num_images < self.images[class_idx].shape[0] else self.images[class_idx].shape[0])])
                )

            elif len(self.images.shape) == 5:
                # 2D
                top_confusing_samples = _strip_image_from_grid_row(
                    torch.stack([
                        torch.stack([
                            torch.tensor(
                                np.uint8(255 * normalize((self.images[class_idx, i, 0, ...]).numpy()))
                            )
                        ])
                    for i in range(self.num_images if self.num_images < self.images[class_idx].shape[0] else self.images[class_idx].shape[0])])
                )

            else:
                raise RuntimeError("Unknown image shape found for confusion visualization")

            module.logger.experiment.log({
                # class_idx represents the ground truth, i.e. these were samples to be classified as class_idx
                # but they were predicted to belong to a different class
                f"val/top_confusing_of_class_{class_idx}": wandb.Image(top_confusing_samples)
            })

    def reset(self):
        self.activations = -99 * torch.ones(self.num_classes, self.num_images)
        self.images = torch.zeros(torch.Size([self.num_classes, self.num_images]) + torch.Size(self.image_shape))