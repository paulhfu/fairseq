# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import torch

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round


logger = logging.getLogger(__name__)


@dataclass
class ModelCriterionConfig(FairseqDataclass):
    loss_weights: Dict[str, float] = field(
        default_factory=dict,
        metadata={"help": "weights for the loss terms"},
    )
    log_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "additional output keys to log"},
    )
    can_sum: bool = True


@register_criterion("alpaca", dataclass=ModelCriterionConfig)
class AlpacaCriterion(FairseqCriterion):
    """
    This criterion relies on the model to supply losses.
    The losses should be a dictionary of name -> scalar returned by
    the model either by including it in the net_output dict or by
    implementing a get_losses(net_output, sample) method. The final loss is
    a scaled sum of all losses according to weights in loss_weights.
    If no weights are provided, then all losses are scaled by 1.0.

    The losses will be automatically logged. Additional keys from
    net_output dict can be logged via the log_keys parameter.
    """

    def __init__(self, task, loss_weights=None, log_keys=None, can_sum=True):
        super().__init__(task)
        self.loss_weights = loss_weights
        self.log_keys = log_keys
        self.can_sum = can_sum

    def isSampleOk(self, sample):
        for i, s in enumerate(sample):
            if s[-1].isinf().any().item() or s[-1].isnan().any().item():
                raise Exception("PPAN: Found invalid val in dataloader")

    def forward(self, model, sample, reduce=True):
        lossCfg = sample[2]
        sample = sample[:2]
        self.isSampleOk(sample)
        try:
            net_output = model(sample[0], sample[1], lossCfg=lossCfg)
        except Exception as e:
            print(e)
            raise e

        loss = net_output["losses"]["main"]

        sample_size = net_output["sample_size"]
        rndInd = torch.randint(0, sample_size, (3,))

        logging_output = {
            "loss": loss.data,
            "ntokens": sample[0].shape[1],
            "sample_size": sample[0].shape[0],
            "_input0": sample[0][rndInd[0], :-1, :2].reshape((-1,)).cpu(),
            "_future0": lossCfg[0][rndInd[0], ..., :2].reshape((-1,)).cpu(),
            "_tgts": net_output["_tgt"].cpu(),
            "_preds": net_output["_pred"].cpu()
        }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        metrics.log_scalar("sample_size", sample_size)

        builtin_keys = {
            "loss",
            "ntokens",
            "nsentences",
            "sample_size",
            "_world_size",
        }

        world_size = utils.item(
            sum(log.get("_world_size", 0) for log in logging_outputs)
        )

        for k in logging_outputs[0]:
            if k not in builtin_keys and not k.startswith("_"):
                val = sum(log.get(k, 0) for log in logging_outputs)
                if k.startswith("loss_"):
                    metrics.log_scalar(k, val / sample_size, sample_size, round=3)
                else:
                    metrics.log_scalar(k, val / world_size, round=3)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        total = sum(log.get("count", 0) for log in logging_outputs)

        if total > 0:
            metrics.log_scalar("_correct", correct)
            metrics.log_scalar("_total", total)

            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(
                    meters["_correct"].sum / meters["_total"].sum, 5
                )
                if meters["_total"].sum > 0
                else float("nan"),
            )

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return self.can_sum
