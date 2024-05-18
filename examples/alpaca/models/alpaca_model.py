# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
)
from fairseq.models.roberta.model import RobertaLMHead, RobertaClassificationHead
from examples.alpaca.models.transformer_trades_encoder import TransformerTradesEncoder, TradesEncoderConfig

from fairseq.modules.transformer_sentence_encoder import init_bert_params

logger = logging.getLogger(__name__)


@dataclass
class AlpacaModelConfig(FairseqDataclass):
    max_positions: int = II("task.tokens_per_sample")

    head_layers: int = 1

    transformer: TradesEncoderConfig = TradesEncoderConfig()

    load_checkpoint_heads: bool = field(
        default=False,
        metadata={"help": "(re-)register and load heads when loading checkpoints"},
    )

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    tokenSize: int = field(
        default=1028, metadata={"help": "sz of input tokens"}
    )
    dsetFeatureDim: int = field(
        default=8, metadata={"help": "sz if data feature dim"}
    )
    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("alpaca", dataclass=AlpacaModelConfig)
class AlpacaModel(FairseqEncoderModel):
    def __init__(self, cfg: AlpacaModelConfig, encoder):
        super().__init__(encoder)
        self.cfg = cfg

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        encoder = AlpacaEncoder(cfg, task.cfg.data)
        return cls(cfg, encoder)

    def forward(
        self,
        src_tokens,
        target_tokens=None,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        res = self.encoder(
            src_tokens, target_tokens, features_only, return_all_hiddens, **kwargs
        )

        if isinstance(res, tuple):
            x, extra = res
        else:
            return res

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=self.cfg.transformer.encoder.embed_dim,
            inner_dim=inner_dim or self.cfg.transformer.encoder.embed_dim,
            num_classes=num_classes,
            activation_fn="tanh",
            pooler_dropout=0,
        )

    @property
    def supported_targets(self):
        return {"self"}

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

            if self.encoder.regression_head is not None:
                if ".lm_head." in k:
                    new_k = k.replace(".lm_head.", ".regression_head.")
                    state_dict[new_k] = state_dict[k]
                    del state_dict[k]
            else:
                if ".regression_head." in k:
                    del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            or self.classification_heads is None
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if self.cfg.load_checkpoint_heads:
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if (
            hasattr(self, "classification_heads")
            and self.classification_heads is not None
            and len(self.classification_heads) > 0
        ):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

            for k in list(state_dict.keys()):
                if k.startswith(prefix + "encoder.lm_head.") or k.startswith(
                    prefix + "encoder.emb_head."
                ):
                    del state_dict[k]

            self.encoder.lm_head = None

        if self.encoder.target_model is None:
            for k in list(state_dict.keys()):
                if k.startswith(prefix + "encoder.target_model."):
                    del state_dict[k]

        if (self.encoder.ema is None) and (prefix + "encoder._ema" in state_dict):
            del state_dict[prefix + "encoder._ema"]

    def remove_pretraining_modules(self, last_layer=None):
        self.encoder.lm_head = None
        self.encoder.regression_head = None
        self.encoder.ema = None
        self.classification_heads = None

        if last_layer is not None:
            self.encoder.sentence_encoder.layers = nn.ModuleList(
                l
                for i, l in enumerate(self.encoder.sentence_encoder.layers)
                if i <= last_layer
            )
            self.encoder.sentence_encoder.layer_norm = None


class AlpacaEncoder(FairseqEncoder):
    def __init__(self, cfg: AlpacaModelConfig, task_data):
        super().__init__(None)

        self.cfg = cfg
        self.sequence_encoder = self.build_encoder(cfg)
        self.mask_token = torch.ones(cfg.tokenSize, dtype=torch.float)
        self.ema = None
        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_scale = cfg.loss_scale

        assert self.cfg.head_layers >= 1

        embed_dim = cfg.transformer.encoder.embed_dim
        curr_dim = embed_dim
        projs = []
        for i in range(self.cfg.head_layers - 1):
            next_dim = embed_dim * 2 if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim

        projs.append(nn.Linear(curr_dim, cfg.transformer.dsetFeatureDim))
        self.regression_head = nn.Sequential(*projs)

        self.num_updates = 0

    def build_encoder(self, cfg):
        encoder = TransformerTradesEncoder(cfg.transformer, return_fc=True)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params
        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        src_tokens,
        target_tokens=None,
        features_only=False,
        return_all_hiddens=False,
        lossCfg=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """

        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )

        if features_only:
            return x, extra

        assert target_tokens is not None
        y = target_tokens.squeeze(1).reshape((x.shape[0], -1, 6))

        x = self.regression_head(x[:, -1])
        x = x.reshape((x.shape[0], -1, 6))

        # categorical weighted ce loss with focal loss weighting sheme
        x = x[..., :3]
        y = y[..., :3]
        x = torch.softmax(x, -1)

        sz = x.size(0)
        x, y = x[:, lossCfg[2]], y[:, lossCfg[2]]

        gamma = 2
        alpha = lossCfg[1]
        p_t = y * x
        cce = y * x.log()
        loss = - (alpha * ((1 - p_t) ** gamma) * cce).sum(-1)
        loss = loss.mean()

        result = {
            "losses": {
                "main": loss
            },
            "_pred": x.argmax(-1) + 1,
            "_tgt": y.argmax(-1) + 1,
            "sample_size": sz,
        }

        result["logs"] = {}
        return result

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        encoder_out = self.sequence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {
            "inner_states": inner_states,
            "encoder_embedding": encoder_out["encoder_embedding"][0],
        }

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.cfg.max_positions
