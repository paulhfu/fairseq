# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from omegaconf import II, MISSING

import numpy as np

from fairseq import utils
from fairseq.data import (
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    RightPaddingMaskDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from examples.alpaca.data.alpaca_dataset import AlpacaDataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from fairseq.tasks.language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES

logger = logging.getLogger(__name__)


@dataclass
class AlpacaConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    nSymbols: int = field(
        default=16,
        metadata={"help": "num symbols in input"}
    )
    klineSz: int = field(
        default=8,
        metadata={"help": "num elements in candle defn"}
    )
    batchSize: int = field(
        default=128,
        metadata={"help": "batchSize"}
    )
    futureLenMins: int = field(
        default=30,
        metadata={"help": "shorten training set for debugging"},
    )
    ulMargin: float = field(
        default=5.0,
        metadata={"help": "upper margin for limit order in the base symbol"},
    )
    llMargin: float = field(
        default=1.0,
        metadata={"help": "lower stop loss margin for limit order in the base symbol"},
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    mask_size: int = field(
        default=1,
        metadata={"help": "probability of replacing a token with mask"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    seed: int = II("common.seed")

    include_target_tokens: bool = field(
        default=False,
        metadata={
            "help": "include target tokens in model input. this is used for data2vec"
        },
    )
    include_index: bool = field(
        default=True,
        metadata={"help": "include index in model input. this is used for data2vec"},
    )
    skip_masking: bool = field(
        default=False,
        metadata={"help": "skip masking at dataset"},
    )
    d2v2_multi: bool = field(
        default=False,
        metadata={"help": "prepare dataset for data2vec_multi"},
    )


@register_task("alpaca", dataclass=AlpacaConfig)
class AlpacaTask(FairseqTask):

    cfg: AlpacaConfig

    def __init__(self, cfg: AlpacaConfig):
        super().__init__(cfg)

    @classmethod
    def setup_task(cls, cfg: AlpacaConfig, **kwargs):
        return cls(cfg)

    def _load_dataset_split(self, split, epoch, combine):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = AlpacaDataset(split_path, self.cfg.batchSize, self.cfg.tokens_per_sample, self.cfg.futureLenMins, self.cfg.nSymbols, self.cfg.klineSz)
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        return dataset

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        try:
            dataset = self._load_dataset_split(split, epoch, combine)
            self.datasets[split] = dataset
        except ...:
            raise Exception("MaskedAlpacaTask: Could not load Dataset")

    def begin_epoch(self, epoch, model):
        model.set_epoch(epoch)

    def max_positions(self):
        return self.cfg.tokens_per_sample
