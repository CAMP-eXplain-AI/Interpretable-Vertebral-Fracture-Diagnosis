import matplotlib
matplotlib.use('Agg')

import os
import wandb
import pytorch_lightning as pl

from data import VerseDataModule
from model import VerseFxClassifier
from utils.config import get_config

if __name__ == '__main__':
    config = get_config("config.yaml")

    USE_WANDB = 'online' if config.pop('USE_WANDB', False) else 'disabled'
    WANDB_API_KEY = config.pop('WANDB_API_KEY')
    SAVE_MODEL = config.pop('SAVE_MODEL')

    wandb.login(key=WANDB_API_KEY)

    run = wandb.init(
        project=f'fx-{config["task"]}-baseline-3d', 
        entity='ifl-diva',
        config=config,
        mode=USE_WANDB
    )

    hparams = wandb.config

    wandb_logger = pl.loggers.WandbLogger()

    model = VerseFxClassifier(hparams)
    data = VerseDataModule(hparams)

    callbacks = [pl.callbacks.EarlyStopping(monitor="val/F1", mode="max", patience=hparams.early_stopping_patience)]

    if bool(SAVE_MODEL):
        callbacks.append(pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='val/F1', mode="max",
                                                                       dirpath='saved_models',
                                                                       filename=f"{wandb.run.name}-epoch{{epoch}}-val_F1={{val/F1:.3f}}",
                                                                       auto_insert_metric_name=False))

    trainer = pl.Trainer(
        gpus=1,
        logger=wandb_logger,
        log_every_n_steps=2,
        #max_epochs=2,
        callbacks=callbacks,
        # auto_lr_find=hparams.auto_lr_find,
    )

    with run:
        trainer.fit(model, data)