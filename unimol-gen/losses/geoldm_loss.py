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


@register_loss("finetune_geoldm_loss")
class FinetunegeoldmmLoss(CrossEntropyLoss):
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
        net_output = model(
            **sample["net_input"]
        )

        bsz = sample["net_input"]['src_tokens'].size(0)

        loss = net_output['loss']
        node_dist = net_output['node_dist']



        logging_output = {
                "loss": net_output['loss'].data,
                "loss_ld": net_output["loss_ld"].data,
                "loss_recon": net_output["loss_recon"].data,
                "neg_log_constants": net_output["neg_log_constants"].data,
                "sample_size": 1,
                "bsz": bsz,
                "loss_distance": net_output["loss_distance"].data if ("loss_distance" in net_output and hasattr(net_output["loss_distance"], "data")) else 0.0,
        }

        # if True:
            # model.eval()
            # # if len(self.task.args.conditioning) > 0:
            # #     model.sample_sweep_conditional(model, model.prop_dist)
            # model.sample_chain(n_tries=1)
            # self.sample_different_sizes_and_save(model, node_dist)

        # if not self.training:
        # if not self.training:
        #     print("save and sample......")
        #     model.sample_chain(n_tries=1)
        #     self.sample_different_sizes_and_save(model, node_dist)

        return loss, 1, logging_output


    def sample_different_sizes_and_save(self, model, nodes_dist, n_samples=32, epoch=0, batch_size=100):
        batch_size = min(batch_size, n_samples)
        prop_dist = None
        for counter in range(int(n_samples/batch_size)):
            nodesxsample = nodes_dist.sample(batch_size)
            one_hot, charges, x, node_mask = model.sample(
                                                    prop_dist=prop_dist,
                                                    nodesxsample=nodesxsample,
                                                )
            print(f"Generated molecule: Positions {x[:-1, :, :]}")

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_ld = sum(log.get("loss_ld", 0) for log in logging_outputs)
        loss_recon = sum(log.get("loss_recon", 0) for log in logging_outputs)
        neg_log_constants = sum(log.get("neg_log_constants", 0) for log in logging_outputs)        
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        loss_distance = sum(log.get("loss_distance", 0) for log in logging_outputs)
        # acc = sum(log.get("acc", 0) for log in logging_outputs)

        
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )

        metrics.log_scalar(
            "loss_ld", loss_ld / sample_size, sample_size, round=3
        )

        metrics.log_scalar(
            "loss_recon", loss_recon / sample_size, sample_size, round=3
        )

        metrics.log_scalar(
            "neg_log_constants", neg_log_constants / sample_size, sample_size, round=3
        )

        metrics.log_scalar(
            "loss_distance", loss_distance / sample_size, sample_size, round=3
        )


    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
        
    