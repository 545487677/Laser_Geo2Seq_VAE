# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .transformer_encoder_with_tri_pair import TransformerEncoderWithTriPair
from typing import Dict, Any, List

BACKBONE = {
    'transformer': TransformerEncoderWithPair,
    'transformer_tri': TransformerEncoderWithTriPair,
}

logger = logging.getLogger(__name__)


@register_model("unimol-laser")
class UniMolLASERModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="transformer",
            choices=BACKBONE.keys(),
            help="backbone of unimol model",
        )
        parser.add_argument(
            "--kernel",
            type=str,
            default="gaussian",
            help="kernel type for distance map",
        )
        parser.add_argument(
            "--kernel-size",
            type=int,
            default=64,
            help="kernel size for distance map",
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self._num_updates = None
        self.encoder = BACKBONE[args.backbone](
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
        )

        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(args.kernel_size, args.encoder_attention_heads, args.activation_fn)
        if args.kernel == 'gaussian':
            self.gbf = GaussianLayer(args.kernel_size, n_edge_type)
        elif args.kernel == 'numerical':
            self.gbf = NumericalEmbed(args.kernel_size, n_edge_type)
        elif args.kernel == 'oled':
            self.gbf = OLEDEmbed(args.kernel_size, n_edge_type)

        self.classification_heads = nn.ModuleDict()
        self.node_classification_heads = nn.ModuleDict()
        self.dist_heads = nn.ModuleDict()
        self.coord_heads = nn.ModuleDict()
        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        **kwargs
    ):
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(x=dist, edge_type=et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        outputs = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        return outputs

    def register_distance_head(
            self, name, num_classes=None, opt_coord=True, **kwargs
    ):
        """Register a dist head."""
        if not opt_coord:
            self.dist_heads[name] = DistanceHead(self.args.encoder_attention_heads, num_classes, self.args.activation_fn)
        else:
            self.dist_heads[name] = DistanceWithCoordHead(self.args.encoder_attention_heads, 1, self.args.activation_fn)

    def register_coord_head(
            self, name, **kwargs
    ):
        """Register a coord head."""
        self.coord_heads[name] = SE3CoordHead(self.args.encoder_attention_heads, self.args.activation_fn)
        
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
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def register_node_classification_head(
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
        self.classification_heads[name] = NodeClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )    

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class NodeClassificationHead(nn.Module):
    """Head for node-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        out_dim,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x, **unused):
        bsz, seq_len, seq_len, _ = x.size()
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len, -1)
        x = (x + x.transpose(1, 2)) * 0.5
        return x

class SE3CoordHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.pair2coord_proj = NonLinearHead(heads, 1, activation_fn)

    def forward(self, src_coord, delta_encoder_pair_rep, padding_mask=None):
        if padding_mask is not None:
            atom_num = (torch.sum(1-padding_mask.type_as(src_coord), dim=1) - 1).view(-1, 1, 1, 1)
        else:
            atom_num = src_coord.shape[1] - 1
        delta_pos = src_coord.unsqueeze(1) - src_coord.unsqueeze(2)
        attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
        coord_update = delta_pos / atom_num * attn_probs
        coord_update = torch.sum(coord_update, dim=2)
        encoder_coord = src_coord + coord_update
        return encoder_coord

class DistanceWithCoordHead(nn.Module):
    def __init__(
        self,
        heads,
        out_dim,
        activation_fn,
        iter=2,
        early_stopping=5,
        dist_th=4.5,
        **unused,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.iter = iter
        self.early_stopping = early_stopping
        self.dist_th = dist_th
    
    def scoring_function(self, opt_coord, dist_predict, dist_mask):
        dist = torch.norm(opt_coord.unsqueeze(-2) - opt_coord.unsqueeze(-3), dim=-1) # bs, mol_sz, mol_sz
        # print(dist_mask.float().mean())
        # dist_mask = (dist < self.dist_th) & dist_mask
        # print(dist_mask.float().mean())
        dist_score = ((dist[dist_mask] - dist_predict[dist_mask])**2).mean()
        loss = dist_score
        return loss

    def forward(self, x, src_coord=None, dist_mask=None):
        opt_coord = torch.ones_like(src_coord).type_as(src_coord) * src_coord
        ### predict pair distance
        bsz, seq_len, seq_len, _ = x.size()
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len, -1)
        x = (x + x.transpose(1, 2)) * 0.5
        x = x.squeeze(-1)
        
        #### pair distance to coord
        if src_coord is not None:
            opt_coord.requires_grad = True
            optimizer = torch.optim.LBFGS([opt_coord], lr=1.0)
            bst_loss, times = 10000.0, 0
            # import pdb; pdb.set_trace()
            for i in range(self.iter):
                def closure():
                    optimizer.zero_grad()
                    loss = self.scoring_function(opt_coord, x, dist_mask)
                    loss.backward(retain_graph=True)
                    return loss
                loss = optimizer.step(closure)
                # print("iter: {}, loss: {}".format(i, loss.item()))
                if loss.item() < bst_loss:
                    bst_loss = loss.item()
                    times = 0 
                else:
                    times += 1
                    if times > self.early_stopping:
                        break
            return x, opt_coord.detach()
        return x, src_coord

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type, **params):  # x: (atoms, hidden),  edge_type: atoms, atoms
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class NumericalEmbed(nn.Module):
    def __init__(self, K=128, edge_types=1024, activation_fn='gelu'):
        super().__init__()
        self.K = K 
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        self.w_edge = nn.Embedding(edge_types, K)

        self.proj = NonLinearHead(1, K, activation_fn, hidden=2*K)
        self.ln = LayerNorm(K)

        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
        nn.init.kaiming_normal_(self.w_edge.weight)


    def forward(self, x, edge_type, **params):    # edge_type, atoms
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        w_edge = self.w_edge(edge_type).type_as(x)
        edge_emb = w_edge * torch.sigmoid(mul * x.unsqueeze(-1) + bias)
        
        edge_proj = x.unsqueeze(-1).type_as(self.mul.weight)
        edge_proj = self.proj(edge_proj)
        edge_proj = self.ln(edge_proj)

        h = edge_proj + edge_emb
        h = h.type_as(self.mul.weight)
        return h
    
class OLEDEmbed(nn.Module):
    def __init__(self, K=128, edge_types=1024, activation_fn='gelu'):
        super().__init__()
        self.K = K 
        self.proj_dist = NonLinearHead(1, K, activation_fn, hidden=K)
        self.ln_dist = LayerNorm(K)
        self.proj_charge = NonLinearHead(2, K, activation_fn, hidden=K)
        self.ln_chare = LayerNorm(K)
        self.proj_et = nn.Embedding(edge_types, K)

        self.proj = NonLinearHead(3*K, K, activation_fn, hidden=K)

    def forward(self, x, edge_type, pair_charge, **params):    # pair_dist, edge_type, atoms_pair_charge
        dist = x.unsqueeze(-1).type_as(self.proj_dist.linear1.weight)
        dist = self.proj_dist(dist)
        dist = self.ln_dist(dist)

        charge = pair_charge.type_as(self.proj_charge.linear1.weight)
        charge = self.proj_charge(charge)
        charge = self.ln_chare(charge)

        et = self.proj_et(edge_type)

        h = torch.cat([dist, charge, et], dim=-1)
        h = self.proj(h)

        return h

@register_model_architecture("unimol", "unimol-laser")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 1024)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.masked_charge_loss = getattr(args, "masked_charge_loss", -1.0)
    args.energy_loss = getattr(args, "energy_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    args.backbone = getattr(args, "backbone", "transformer")
    args.kernel = getattr(args, "kernel", 'gaussian')
    args.kernel_size = getattr(args, "kernel_size", 128)

@register_model_architecture("unimol", "unimol_laser_small")
def unimol_laser_architecture(args):
    base_architecture(args)
    args.encoder_layers = 8
    args.kernel_size = 128