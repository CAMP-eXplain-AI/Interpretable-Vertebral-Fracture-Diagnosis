import pandas as pd
import os
from pathlib import Path
from monai.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import Subset
import pytorch_lightning as pl
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data.dataloader import default_collate

from transforms import get_training_transforms, get_base_transforms

class VerseDataModule(pl.LightningDataModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(dict(hparams), logger=False)
    self.data_dir = Path(self.hparams.dataset_path)
    self.csv_path = self.data_dir / 'fxall_labels.csv'

    if "modelsgenesis" in hparams.transforms:
      self.image_dir = self.data_dir / 'raw'
    else:
      self.image_dir = self.data_dir / 'ct'

    if not os.path.exists(self.image_dir):
      # legacy support
      self.image_dir = self.data_dir
      self.csv_path = self.data_dir / 'slice_labels.csv'

    self.mask_dir = self.data_dir / 'seg'
    if hparams.mask != 'none' and not os.path.exists(self.mask_dir):
      raise RuntimeError("Configured to use masks, but 'seg' folder missing in dataset path")

    self.df = pd.read_csv(self.csv_path, index_col=0)

    # TODO temporary fix to check for non-existing files
    if "path" in self.df.columns:
      self.df = self.df[self.df["path"].apply(lambda p: os.path.exists(self.image_dir / p))]

    # FIXME slice_labels.csv provides the path in 'image', fxall_labels.csv in 'path'
    if "image" not in self.df.columns:
      self.df['image'] = self.df['path']

    self.transforms = {
      'training': get_training_transforms(hparams, self.image_dir, self.mask_dir),
      'validation': get_base_transforms(hparams, self.image_dir, self.mask_dir),
      'test': get_base_transforms(hparams, self.image_dir, self.mask_dir)
    }

    self.datasets = {}
    self.idxs = {}

  def setup(self, stage=None):
    # dropping samples without fracture grading
    # graded_idxs = ~self.df.fx.isna()
    # vertebrae_level_idxs = self.df.level_idx >= self.hparams.min_vertebrae_level
    # included_idxs = graded_idxs & vertebrae_level_idxs
    
    if stage == 'fit' or stage is None:
      phases = ['training', 'validation']
    else:
      phases = ['test']

    for split in phases:
      # get official verse partitions
      idxs = self.df[f'split_{self.hparams.fold}'] == split
      # idxs = np.where(included_idxs & idxs)[0]
      idxs = np.where(idxs)[0]
      self.idxs[split] = idxs
      self.datasets[split] = Dataset(
        self.df.iloc[idxs].to_dict('records'), 
        transform=self.transforms[split]
      )

  def get_label(self, data):
    train_df = self.df.iloc[self.idxs['training']]
    grading = train_df.fx_grading
    if self.hparams.task == 'detection':
        return train_df.fx
    elif self.hparams.task == 'grading':
        return grading
    elif self.hparams.task == 'simple_grading':
        if grading in [2,3]:
          return 1
        if grading>3:
          return grading-2
        else:
          return grading

  def train_dataloader(self):
    return DataLoader(
      self.datasets['training'], 
      batch_size=self.hparams.batch_size,
      sampler=ImbalancedDatasetSampler(
        num_samples=self.df.iloc[self.idxs['training']].fx.sum() * 2,
        dataset=self.datasets['training'],
        callback_get_label=self.get_label,
        ) if self.hparams.oversampling else None, 
      num_workers=2,
      shuffle=not self.hparams.oversampling
    )

  def val_dataloader(self):
    return DataLoader(
      self.datasets['validation'], 
      batch_size=self.hparams.batch_size, 
      num_workers=8,
    )

  def test_dataloader(self):
    return DataLoader(
      self.datasets['test'], 
      batch_size=self.hparams.batch_size, 
      num_workers=8,
    )
