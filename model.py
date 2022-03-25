from typing import List
import torch
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from backbones import get_backbone
from utils.confusion_viz import ConfusionVisualizer

from utils.plots import get_confusion_matrix_figure
from utils.grad_cam import GradCAMBuilder

from loss import ordinal_regression_loss, get_breadstick_probabilities, focal_loss
from utils.val_loop_hook import ValidationLoopHook

class VerseFxClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))
        self.backbone = get_backbone(self.hparams)
        metric_args = dict(average='macro', num_classes=self.hparams.num_classes)
        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(**metric_args), 
            torchmetrics.F1(**metric_args), 
            torchmetrics.Precision(**metric_args), 
            torchmetrics.Recall(**metric_args)
        ])
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        
        self.class_weights = None

        image_shape = (1 + (self.hparams.mask =='channel'),) + (self.hparams.input_size,) * self.hparams.input_dim
        grad_cam_builder = GradCAMBuilder(image_shape, target_category=0 if self.hparams.task == 'detection' else None)

        confusion_visualizer = ConfusionVisualizer(image_shape, 2 if self.hparams.task == 'detection' else self.hparams.num_classes)

        self.validation_hooks: List[ValidationLoopHook] = [grad_cam_builder, confusion_visualizer]
        
    def get_class_weights(self, dm: pl.LightningDataModule):
        targets = []
        for batch in tqdm(dm.train_dataloader(), desc="Determining class weights"):
            targets.append(self.batch_to_targets(batch))
        targets = torch.cat(targets)
        classes, counts = torch.unique(targets, return_counts=True)
    
        return (1 / counts) * torch.sum(counts) / classes.shape[0]
    
    def on_pretrain_routine_start(self):
        # FIXME This is slightly inefficient if multiple GPUs are used as this routine
        # is called once per device. There might be a better hook available.
        super().on_pretrain_routine_start()
        
        if self.hparams.weighted_loss:
            self.class_weights = self.get_class_weights(self.trainer.datamodule).to(self.device)
            
            if self.hparams.loss == 'binary_cross_entropy':
                # Only keep the positive class weight
                self.class_weights = self.class_weights[-1]

    def forward(self, x):
        return self.backbone(x)

    def loss(self, logits, targets):
        if self.hparams.loss == 'cross_entropy':
            return F.cross_entropy(logits, targets, weight=self.class_weights)
        elif self.hparams.loss == 'binary_cross_entropy':
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets.float(),
                                                      pos_weight=self.class_weights)
        elif self.hparams.loss == 'ordinal_regression':
            return ordinal_regression_loss(logits, targets, class_weights=self.class_weights)
        elif self.hparams.loss == 'focal':
            return focal_loss(logits.squeeze(-1), targets.float())
        else:
            raise ValueError

    def logits_to_predictions(self, logits):
        if self.hparams.loss == 'binary_cross_entropy' or (self.hparams.loss == 'focal' and self.hparams.task == 'detection'):
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = probs.gt(0.5).long()
        elif self.hparams.loss == 'cross_entropy' or self.hparams.loss == 'focal':
            probs = torch.softmax(logits)
            preds = probs.argmax(-1)
        elif self.hparams.loss == 'ordinal_regression':
            probs = get_breadstick_probabilities(logits)
            preds = probs.argmax(-1)
        else:
            raise ValueError
    
        return probs, preds

    def batch_to_targets(self, batch):
        if self.hparams.task == 'detection':
            return batch['fx'].long()
        elif self.hparams.task == 'grading':
            return batch['fx_grading'].long()
        elif self.hparams.task == 'simple_grading':
            targets = batch['fx_grading'].long()
            targets[torch.bitwise_or(targets==2, targets==3)] = 1
            targets[targets>3] -= 2
            return targets

    def training_step(self, batch, batch_idx):
        logits = self(batch['image'])
        targets = self.batch_to_targets(batch)
        loss = self.loss(logits, targets)
        probs, preds = self.logits_to_predictions(logits)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
        return {'loss': loss,  'probs': probs.detach(), 'preds': preds.detach(), 'targets': targets.detach()}

    def training_epoch_end(self, outputs):
        outputs = {k: torch.cat([d[k] for d in outputs]) for k in outputs[0] if k != 'loss'}
        metrics = self.train_metrics(outputs['probs'], outputs['targets'])
        self.log_dict(metrics)

        targets_flat = outputs['targets'].cpu().numpy()
        preds_flat = outputs['preds'].cpu().numpy()

        # sklearn confusion matrix
        self.logger.experiment.log({
            "train/confusion_matrix": get_confusion_matrix_figure(
                targets_flat,
                preds_flat,
                title="Training Confusion Matrix"
            )
        })
        plt.close('all')

    def validation_step(self, batch, batch_idx):
        logits = self(batch['image'])
        targets = self.batch_to_targets(batch)
        loss = self.loss(logits, targets)
        probs, preds = self.logits_to_predictions(logits)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
        for val_hook in self.validation_hooks:
            val_hook.process(batch, targets, logits, preds)
        metrics = self.val_metrics(probs, targets)
        self.log_dict(metrics)
        return {'loss': loss, 'probs': probs.detach(), 'preds': preds.detach(), 'targets': targets.detach()}

    def validation_epoch_end(self, outputs):
        outputs = {k: torch.cat([d[k] for d in outputs]) for k in outputs[0] if k != 'loss'}
        metrics = self.val_metrics(outputs['probs'], outputs['targets'])
        self.log_dict(metrics)

        targets_flat = outputs['targets'].cpu().numpy()
        preds_flat = outputs['preds'].cpu().numpy()

        # sklearn confusion matrix
        self.logger.experiment.log({
            "val/confusion_matrix": get_confusion_matrix_figure(
                targets_flat,
                preds_flat,
                title="Validation Confusion Matrix",
            )
        })
        plt.close('all')
        # wandb confusion matrix
        # print(targets_flat, targets_flat.squeeze(-1).shape, type(preds_flat[0]))
        self.logger.experiment.log({
            "full_fx_grading": wandb.plot.confusion_matrix(
                # probs=outputs['y_pred'],
                preds=list(preds_flat), 
                y_true=list(targets_flat),
                class_names=None,
            ),
            "epoch": self.current_epoch
        })

    def on_train_epoch_end(self):
        # Trigger all validation hooks and reset them afterwards
        for val_hook in self.validation_hooks:
            val_hook.trigger(self)
            val_hook.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer