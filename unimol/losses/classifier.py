import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


@register_loss("finetune_classifier_loss")
class FinetuneclassifieroriLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)
        self.dictionary = task.dictionary
        self.eos_idx = task.dictionary.eos()
        self.bos_idx = task.dictionary.bos()
        self.indices = {v:k for k,v in self.dictionary.indices.items()}
        self.task = task

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # model.eval()
        net_output = model(
            **sample["net_input"]
        )

        bsz = sample["net_input"]['src_coord'].size(0)

        loss = net_output['loss']
        # node_dist = net_output['node_dist']
        # prop_dist = net_output['prop_dist']



        logging_output = {
                "loss": net_output['loss'].data,
                # "loss_ld": net_output["loss_ld"].data,
                # "loss_recon": net_output["loss_recon"].data,
                # "neg_log_constants": net_output["neg_log_constants"].data,
                "sample_size": 1,
                "bsz": bsz,
                # "loss_distance": net_output["loss_distance"].data if "loss_distance" in net_output else 0.0,
        }

        return loss, 1, logging_output


    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        # loss_ld = sum(log.get("loss_ld", 0) for log in logging_outputs)
        # loss_recon = sum(log.get("loss_recon", 0) for log in logging_outputs)
        # neg_log_constants = sum(log.get("neg_log_constants", 0) for log in logging_outputs)        
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # loss_distance = sum(log.get("loss_distance", 0) for log in logging_outputs)
        # acc = sum(log.get("acc", 0) for log in logging_outputs)

        
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )

        # metrics.log_scalar(
        #     "loss_ld", loss_ld / sample_size, sample_size, round=3
        # )

        # metrics.log_scalar(
        #     "loss_recon", loss_recon / sample_size, sample_size, round=3
        # )

        # metrics.log_scalar(
        #     "neg_log_constants", neg_log_constants / sample_size, sample_size, round=3
        # )

        # metrics.log_scalar(
        #     "loss_distance", loss_distance / sample_size, sample_size, round=3
        # )


    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
        
    