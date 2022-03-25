from typing import Literal
from numpy import dtype
import os
import torch
from torch.utils import data
from netdissect import nethook, imgviz, show, tally
from netdissect import setting
import torch
from netdissect import setting
import nibabel as nib
import nibabel.orientations as nio

from dataset_test import MalignantTestSet

from pytorch_grad_cam import GradCAM
from torch.utils.data import DataLoader
import cv2
import numpy as np
from tqdm import tqdm

from monai.transforms import CenterSpatialCrop

class MalignantNet3dDissectResults():
    def __init__(self, model, dataset, data_path, model_layer: str):
        """
        Args:
            path_to_data (string): Path to the data
            path_to_test_paths (string): Path to test_paths.npy
            path_to_labels (string): Path to labels.npy
        """

        model = nethook.InstrumentedModel(model)

        if torch.cuda.is_available():
            model.cuda()

        model.eval()
        self.model = model
        self.layername = model_layer
        self.model.retain_layer(self.layername)

        self.topk = None
        self.unit_images = None
        self.iou99 = None

        self.tupfn = torch.nn.Upsample(size=(64, 64, 64), mode="trilinear", align_corners=True)

        self.ds = dataset

        self.data_path = data_path

        self.sample_size = 500

        self.rq = self._get_rq_vals()
        self.iv = imgviz.ImageVisualizer(224, image_size=64, source="zc", percent_level=0.99, quantiles=self.rq)

        self.seglabels = ["background", "vertebra"]
        self.segcatlabels = [('background', 'background'), ('vertebra', 'vertebra')]

    def change_the_retained_layer(self, layername):
        self.layername = layername
        self.model.retain_layer(self.layername)
        self.topk = None
        self.unit_images = None
        self.iou99 = None
        # Restart the imviz for the new layer
        self.rq = self._get_rq_vals()
        self.iv = imgviz.ImageVisualizer(224, image_size=64, source="zc", percent_level=0.99, quantiles=self.rq)

    def _flatten_activations(self, **batch):
        image_batch = batch["image"].cuda() if torch.cuda.is_available() else batch["image"]
        _ = self.model(image_batch)
        acts = self.model.retained_layer(self.layername)
        hacts = acts
        return hacts.permute(0, 2, 3, 4, 1).contiguous().view(-1, acts.shape[1])

    def _get_rq_vals(self):
        rq = tally.tally_quantile(
            self._flatten_activations,
            dataset=self.ds,
            sample_size=self.sample_size,
            batch_size=10)
        return rq


    def _max_activations(self, data: Literal['all', 'pos', 'neg'] = 'all', **batch):
        image_batch = batch["image"].cuda() if torch.cuda.is_available() else batch["image"]
        mask = (1 if data == "all" else (batch["fx"] == (1 if data == "pos" else 0))).view(-1, 1, 1, 1, 1)
        _ = self.model(image_batch)
        acts = self.model.retained_layer(self.layername).cpu() * mask
        return acts.view(acts.shape[:2] + (-1,)).max(2)[0]

    def _mean_activations(self, data: Literal['all', 'pos', 'neg'] = 'all', **batch):
        image_batch = batch["image"].cuda() if torch.cuda.is_available() else batch["image"]
        mask = (1 if data == "all" else (batch["fx"] == (1 if data == "pos" else 0))).view(-1, 1, 1, 1, 1)
        _ = self.model(image_batch)
        acts = self.model.retained_layer(self.layername).cpu() * mask
        return acts.view(acts.shape[:2] + (-1,)).mean(2)

    def compute_topk_imgs(self, mode = 'mean', data: Literal['all', 'pos', 'neg'] = 'all'):
        if mode == 'mean':
            self.topk = tally.tally_topk(
            lambda **batch: self._mean_activations(data, **batch),
            dataset=self.ds,
            sample_size=self.sample_size,
            batch_size=10
            )
        else: # It can only be max if not mean
            self.topk = tally.tally_topk(
            lambda **batch: self._max_activations(data, **batch),
            dataset=self.ds,
            sample_size=self.sample_size,
            batch_size=10
            )

    def _compute_activations(self, batch):
        image_batch = batch["image"].cuda() if torch.cuda.is_available() else batch["image"]
        _ = self.model(image_batch)
        acts_batch = self.model.retained_layer(self.layername)
        return acts_batch

    def compute_top_unit_imgs(self, mode = 'mean', k = 5, data: Literal['all', 'pos', 'neg'] = 'all'):
        if self.topk is None:
            self.compute_topk_imgs(mode, data)
        self.unit_images = self.iv.masked_images_for_topk(
            self._compute_activations,
            self.ds,
            self.topk,
            k=k,
            num_workers=10,
            pin_memory=True)

    def compute_top_unit_nifti(self, mode = 'mean', k=5, data: Literal['all', 'pos', 'neg'] = 'all'):
        if self.topk is None:
            self.compute_topk_imgs(mode, data)
        self.unit_tensors = self.iv.image_activation_tuples_for_topk(
            self._compute_activations,
            self.ds,
            self.data_path,
            self.topk,
            k=k,
            num_workers=10,
            pin_memory=True)

    def compute_top_unit_collage(self, mode = 'mean', k=5, per_row=5, data: Literal['all', 'pos', 'neg'] = 'all'):
        if self.topk is None:
            self.compute_topk_imgs(mode, data)
        self.unit_single_images = self.iv.masked_single_images_for_topk(
            self._compute_activations,
            self.ds,
            self.topk,
            k=25,
            num_workers=10,
            per_row=5,
            pin_memory=True)
    
    def write_nifti_unit(self, unit, rank, custom_suffix=""):
        base = f'nifti{"-"+custom_suffix if len(custom_suffix) > 0 else ""}'
        os.makedirs(base, exist_ok=True)
        for (i, unit_data) in enumerate(self.unit_tensors[unit]):
            # matches the orientation used in preprocessing
            affine = nio.inv_ornt_aff(nio.axcodes2ornt(('I', 'P', 'L')), (64, 64, 64))
            nib.save(nib.Nifti1Image(unit_data[0].numpy(), affine), os.path.join(base, f'top{rank+1:02d}-{unit}-{i+1}{"-"+custom_suffix if len(custom_suffix) > 0 else ""}-ct.nii.gz'))
            nib.save(nib.Nifti1Image(unit_data[1].numpy(), affine), os.path.join(base, f'top{rank+1:02d}-{unit}-{i+1}{"-"+custom_suffix if len(custom_suffix) > 0 else ""}-acts.nii.gz'))
            nib.save(nib.Nifti1Image(unit_data[2].numpy(), affine), os.path.join(base, f'top{rank+1:02d}-{unit}-{i+1}{"-"+custom_suffix if len(custom_suffix) > 0 else ""}-ct-masked.nii.gz'))
        self.unit_images[unit].save(os.path.join(base, f'top{rank+1:02d}-{unit}-0{"-"+custom_suffix if len(custom_suffix) > 0 else ""}.png'))

    def write_collage_unit(self, unit, rank, custom_suffix=""):
        base = f'collages{"-"+custom_suffix if len(custom_suffix) > 0 else ""}'
        os.makedirs(base, exist_ok=True)
        self.unit_single_images[unit].save(os.path.join(base, f'top{rank+1:02d}-{unit}-0{"-"+custom_suffix if len(custom_suffix) > 0 else ""}-collage.png'))

    def show_seg_results(self):
        if self.unit_images is None:
            self.compute_top_unit_imgs()

        if torch.cuda.is_available():
            level_at_99 = self.rq.quantiles(0.99).cuda()[None, :, None, None, None]
        else:
            level_at_99 = self.rq.quantiles(0.99)[None, :, None, None, None]

        sample_size = 20

        def prepare_seg(seg_paths):
            seg = torch.tensor(np.array([np.load(os.path.join(self.data_path, 'seg', seg_path)) for seg_path in seg_paths]))
            seg[seg > 1] = 1
            return seg.long()

        def compute_selected_segments(**batch):
            img, seg = batch["image"], prepare_seg(batch["path"])
            # TODO this is very hack-y
            #seg = img[:, 1, ...].long()
            #     show(iv.segmentation(seg))
            image_batch = img.cuda() if torch.cuda.is_available() else img
            seg_batch = seg.cuda() if torch.cuda.is_available() else seg
            seg_batch = CenterSpatialCrop(64)(seg_batch)
            _ = self.model(image_batch)
            acts = self.model.retained_layer(self.layername)
            hacts = self.tupfn(acts)
            iacts = (hacts > level_at_99).float()  # indicator where > 0.99 percentile.
            return tally.conditional_samples(iacts, seg_batch)

        condi99 = tally.tally_conditional_mean(
            compute_selected_segments,
            dataset=self.ds,
            sample_size=sample_size, pass_with_lbl=True)

        self.iou99 = tally.iou_from_conditional_indicator_mean(condi99)
        bolded_string = "\033[1m" + self.layername + "\033[0m"

        print(bolded_string)
        iou_unit_label_99 = sorted([(
            unit, concept.item(), self.seglabels[int(concept)], bestiou.item())
            for unit, (bestiou, concept) in enumerate(zip(*self.iou99.max(0)))],
            key=lambda x: -x[-1])
        for unit, concept, label, score in iou_unit_label_99[:20]:
            show(['unit %d; iou %g; label "%s"' % (unit, score, label),
                  [self.unit_images[unit]]])

    def show_top_activating_imgs_per_units_with_seg(self, units: int, top_num: int = 1, plane: Literal["saggital", "axial", "coronal"] = "saggital"):
        if self.topk is None:
            self.compute_topk_imgs()
        top_indexes = self.topk.result()[1]
        ld_res = self.model.retained_layer(self.layername)[0].shape[-1]
        img_slices = torch.linspace(int(64/ld_res/2), 64-int(64/ld_res/2), ld_res, dtype=torch.long)

        def load_modality(path, modality: Literal["ct", "seg"], level_idx=None):
            result = torch.tensor(np.load(os.path.join(self.data_path, modality, path)))
            result = CenterSpatialCrop(64)(result[None, ...])[0, ...]
            if modality == "seg":
                result[result != level_idx] = 0
                result[result >= 1] = 1
                return result.long()
            else:
                return result.float()

        show([
            ['unit %d' % u,
             'img %d' % i,
             'slice %s' % f"{s} (ld), {img_slices[s]} (data)",
             'pred: %s' % [self.model(torch.tensor(self.ds[i]['image'][None]).cuda()) if torch.cuda.is_available() else self.model(torch.tensor(self.ds[i]['image'][None]))],
             [self.iv.masked_image(
                 load_modality(self.ds[i]['path'], 'ct')[...,img_slices[s]].repeat(3, 1, 1),
                 self.model.retained_layer(self.layername)[0][..., s],
                 u)],
                 [self.iv.image((load_modality(self.ds[i]['path'], 'ct')[..., img_slices[s]] * (load_modality(self.ds[i]['path'], 'seg', level_idx=self.ds[i]['level_idx'])[..., img_slices[s]] * 0.5 + 0.5)).repeat(3, 1, 1))],
                 [self.iv.heatmap(self.model.retained_layer(self.layername)[0][..., s], u)]
             ]
            for u in units
            for s in range(0, ld_res)
            for i in top_indexes[u, :top_num]
        ])

    def show_seg_gt(self, num_samples = 5, slices = 5):
        imgs = []
        seg = []

        def load_modality(path, modality: Literal["ct", "seg"], level_idx=None):
            result = torch.tensor(np.load(os.path.join(self.data_path, modality, path)))
            result = CenterSpatialCrop(64)(result[None, ...])[0, ...]
            if modality == "seg":
                result[result != level_idx] = 0
                result[result > 1] = 1
                return result.long()
            else:
                return result.float()

        for i in range(num_samples):
            img, lbl = self.ds[i]["image"], self.ds[i]['fx']
            img_slices = torch.linspace(int(64/slices/2), 64-int(64/slices/2), slices, dtype=torch.long)
            for s in img_slices:
                imgs.append(load_modality(self.ds[i]['path'], 'ct')[..., s].repeat(3, 1, 1))
                seg.append(load_modality(self.ds[i]["path"], "seg", level_idx=self.ds[i]['level_idx'])[..., s].long())
        show([(self.iv.image(imgs[i]), self.iv.segmentation(seg[i]),
               self.iv.segment_key_with_lbls(seg[i], self.seglabels))
              for i in range(len(seg))])

    def show_unique_concepts_graph(self, thresh = 0.04,  print_nums = False):
        if self.iou99 is None:
            self.show_seg_results()
        iou_threshold = thresh
        unit_label_99 = [
            (concept.item(), self.seglabels[concept],
             self.segcatlabels[concept], bestiou.item())
            for (bestiou, concept) in zip(*self.iou99.max(0))]
        labelcat_list = [labelcat
                         for concept, label, labelcat, iou in unit_label_99
                         if iou > iou_threshold]
        return setting.graph_conceptcatlist(labelcat_list, cats=self.seglabels, print_nums = print_nums)

class MalignantNet3dGradCam():
    def __init__(self, model, dataset):
        """
        Args:
            path_to_data (string): Path to the data
            path_to_test_paths (string): Path to test_paths.npy
            path_to_labels (string): Path to labels.npy
        """

        if torch.cuda.is_available():
            model.cuda()

        model.eval()
        self.model = model

        self.ds = dataset

    def _show_cam_on_image(self, img: np.ndarray, mask: np.ndarray, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
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

        #if np.max(img) > 1:
        #    raise Exception(
        #        "The input image should np.float32 in the range [0, 1]")

        cam = 0.6 * heatmap + img
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    def show_sample_grad_cam(self, target_layers, target_category, num_images: int = 5):
        image_batch, label_batch = next(iter(DataLoader(self.ds, batch_size=num_images)))

        cam = GradCAM(self.model, target_layers, use_cuda=torch.cuda.is_available())
        grayscale_cam = cam(input_tensor=image_batch, target_category=target_category)

        ld_res = grayscale_cam.shape[-1]
        img_res = image_batch.shape[-1]
        img_slices = torch.linspace(int(img_res/ld_res/2), img_res-int(img_res/ld_res/2), ld_res, dtype=torch.long)

        return show([imgviz.strip_image_from_grid_row(
            torch.stack([
                torch.stack([
                    torch.tensor(
                        self._show_cam_on_image((image_batch[i, 0, ..., img_slices[s]] * (image_batch[i, 1, ..., img_slices[s]] * 0.5 + 0.5)).unsqueeze(-1).repeat(1, 1, 3).numpy(), grayscale_cam[i, ..., s], use_rgb=True)
                    )
                for s in range(grayscale_cam.shape[-1])])
            for i in range(num_images)])
        )])

    def show_top_activating_grad_cam(self, target_layers, target_category, num_images: int = 5):
        dl = DataLoader(self.ds, batch_size=num_images)
        labels = []
        activations = []
        image_batches = []
        grad_cams = []

        # TODO may be implemented more effeciently by directly building num_images batch to visualize based on minimal activation to exceed
        for (input_batch, label_batch) in tqdm(dl):
            labels.append(label_batch)
            image_batches.append(input_batch)

            with torch.no_grad():
                activations.append(self.model.forward(input_batch))

            cam = GradCAM(self.model, target_layers, use_cuda=torch.cuda.is_available())
            grad_cams.append(cam(input_tensor=input_batch, target_category=target_category))

        labels = torch.hstack(labels)
        activations = torch.vstack(activations)[:, 1]
        image_batches = torch.vstack(image_batches)
        grad_cams = np.vstack(grad_cams)

        top_idx = torch.argsort(activations, descending=True)[:num_images]

        labels = labels[top_idx]
        image_batch = image_batches[top_idx]
        grayscale_cam = grad_cams[top_idx.numpy()]

        ld_res = grayscale_cam.shape[-1]
        img_res = image_batch.shape[-1]
        img_slices = torch.linspace(int(img_res/ld_res/2), img_res-int(img_res/ld_res/2), ld_res, dtype=torch.long)

        return show([imgviz.strip_image_from_grid_row(
            torch.stack([
                torch.stack([
                    torch.tensor(
                        self._show_cam_on_image((image_batch[i, 0, ..., img_slices[s]] * (image_batch[i, 1, ..., img_slices[s]] * 0.5 + 0.5)).unsqueeze(-1).repeat(1, 1, 3).numpy(), grayscale_cam[i, ..., s], use_rgb=True)
                    )
                for s in range(grayscale_cam.shape[-1])])
            for i in range(num_images)])
        )])