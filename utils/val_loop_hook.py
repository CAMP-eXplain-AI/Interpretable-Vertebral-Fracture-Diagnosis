from abc import ABC, abstractmethod
import torch
import pytorch_lightning as pl

class ValidationLoopHook(ABC):
    @abstractmethod
    def process(self, batch: torch.Tensor, target_batch: torch.Tensor, logits_batch: torch.Tensor, prediction_batch: torch.Tensor) -> None:
        """
        Called for every validation batch to process results.
        """
        pass

    @abstractmethod
    def trigger(self, module: pl.LightningModule):
        """
        Called after the validation epoch has concluced to further interact with the module and/or log data.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Called right after build() to clean up before the next validation epoch starts.
        """
        pass