# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm
from .encoder import UniMolV1Model, UniMolOLEDModel, UniMolOPVModel
from .decoder import TransformerDecoder, GRUDecoder
from ..utils import inference_greedy, inference_beam, inference_top_p, inference_top_k

logger = logging.getLogger(__name__)

ENCODER_REGISTER = {
    'unimolv1': UniMolV1Model,
    'unimol-oled': UniMolOLEDModel,
    'unimol-opv': UniMolOPVModel,
}    

DECODER_REGISTER = {
    'TFM': TransformerDecoder,
    'GRU': GRUDecoder,
}

INFERENCE_REGISTER = {
    'GREEDY': inference_greedy,
    'BEAM':   inference_beam,
    'TOPP':   inference_top_p,
    'TOPK':   inference_top_k,
}

@register_model("cvae")
class Cvae(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--encoder",
            type=str,
            metavar='D',
            choices=ENCODER_REGISTER.keys(),
        )        
        parser.add_argument(
            "--decoder",
            type=str,
            metavar="D",
            choices=DECODER_REGISTER.keys(),
        )
        parser.add_argument(
            "--infer_method",
            type=str,
            default="GREEDY",
            choices=["GREEDY", "TOPP", "TOPK", "BEAM"],
        )

    def __init__(self, args, dictionary, smi_dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.teacher_forcing = True if self.args.decoder == 'TFM' else False
        self.padding_idx = dictionary.pad()
        self.smi_padding_idx = smi_dictionary.pad()
        
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self.latent_dim = args.encoder_embed_dim
        self._num_updates = None
        self.encoder = ENCODER_REGISTER[args.encoder](self.args, dictionary)
        decoder = DECODER_REGISTER[args.decoder](
            decoder_layers=args.decoder_layers,
            embed_dim=args.embed_dim,
            ffn_embed_dim=args.ffn_embed_dim,
            attention_heads=args.attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            rel_pos=args.rel_pos,
            rel_pos_bins=args.rel_pos_bins,
            max_rel_pos=args.max_rel_pos,
            post_ln=args.post_ln,
            auto_regressive=args.auto_regressive,
        )
        self.mol_cvae_model = Molecule_CVAE(
            latent_dim = self.latent_dim, 
            condition_dim = 1,
            smi_dictionary=smi_dictionary,
            decoder=decoder,
            teacher_forcing=self.teacher_forcing,
            infer_method=self.args.infer_method,
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.smi_dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        condition_info,
        tgt_tokens,
        **kwargs
    ):
        smi_seq_len = tgt_tokens.size(1)
        encoder_outputs = self.encoder(src_tokens=src_tokens, 
                                        src_distance=src_distance, 
                                        src_coord=src_coord,
                                        src_edge_type=src_edge_type,
                                        **kwargs)
        encoder_rep = encoder_outputs[0][:, 0, :]
        condition_info = condition_info.unsqueeze(1)
        output, mean, logvar = self.mol_cvae_model(encoder_rep, condition_info, smi_seq_len, tgt_tokens)
        return output, mean, logvar     

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

    

class Molecule_CVAE(nn.Module):
    def __init__(self, latent_dim, condition_dim, smi_dictionary, decoder=None, teacher_forcing=None, infer_method=None, **kwargs):
        super(Molecule_CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.vocab_size = len(smi_dictionary)
        self.decoder = decoder
        self.dec_embeded_tokens = nn.Embedding(
            len(smi_dictionary), latent_dim, smi_dictionary.pad()
        )
        self.linear_1 = nn.Linear(latent_dim + condition_dim, latent_dim)
        self.linear_2 = nn.Linear(latent_dim + condition_dim, latent_dim)
        self.linear_3 = nn.Linear(latent_dim + condition_dim, latent_dim)
        self.teacher_forcing = teacher_forcing
        self.infer_method = infer_method
        self.linear_4 = nn.Linear(latent_dim, self.vocab_size)

    def encode(self, x, condition_info):
        condition_info = condition_info.type_as(x)
        x = torch.cat((x, condition_info), dim=-1)    
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        if self.training:
            epsilon = 1e-2 * torch.randn_like(z_logvar)
            return torch.exp(0.5 * z_logvar) * epsilon + z_mean
        else:
            return z_mean

    def decode(self, z=None, condition_info=None, max_len=256, dec_in=None):
        condition_info = condition_info.type_as(z)
        z = torch.cat((z, condition_info), dim=-1)
        z = self.linear_3(z.to(self.linear_3.weight.dtype))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, max_len, 1)
        if self.teacher_forcing:
            dec_in_emb = self.dec_embeded_tokens(dec_in)
            output = self.decoder(dec_in_emb, z)
        else:
            output = self.decoder(z)
        y = self.linear_4(output)
        return y

    def forward(self, x, condition_info, max_len, dec_in=None):
        z_mean, z_logvar = self.encode(x, condition_info)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z, condition_info, max_len, dec_in), z_mean, z_logvar
    
    def inference(self, mean=None, logvar=None, condition_info=None, max_len=None, dec_in=None, eos_idx=None):
        mean = mean.to(self.dec_embeded_tokens.weight)
        logvar = logvar.to(self.dec_embeded_tokens.weight)
        if dec_in is not None:
            dec_in = dec_in.to(self.dec_embeded_tokens.weight).long()
        eos_token = torch.tensor([eos_idx] * mean.size(0)).unsqueeze(1).to(self.dec_embeded_tokens.weight)
        epsilon = 1e-2 * torch.randn_like(logvar)
        z = torch.exp(0.5 * logvar) * epsilon + mean
        output_seq = []
        for _ in range(max_len):
            output = self.decode(z, condition_info, max_len, dec_in)
            current_token = INFERENCE_REGISTER[self.infer_method](output)
            # current_token = torch.argmax(output[:,-1,:], dim=1, keepdim=True)
            if torch.equal(current_token, eos_token):
                break
            output_seq.append(current_token)
            if dec_in is not None:
                dec_in = torch.cat([dec_in, current_token], dim=1)
        return torch.cat(output_seq, dim=1)

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
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
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

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-4
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


@register_model_architecture("cvae", "cvae")
def base_architecture(args):
    # encoder
    args.encoder = getattr(args, "encoder", "unimolv1")
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)

    # decoder
    args.decoder = getattr(args, "decoder", "TFM")
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.embed_dim = getattr(args, "embed_dim", 512)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 2048)
    args.attention_heads = getattr(args, "attention_heads", 64)
    args.rel_pos = getattr(args, "rel_pos", True)
    args.rel_pos_bins = getattr(args, "rel_pos_bins", 32)
    args.max_rel_pos = getattr(args, "max_rel_pos", 256)
    args.auto_regressive = getattr(args, "auto_regressive", True)

@register_model_architecture("cvae", "cvae-oled")
def base_architecture(args):
    # encoder
    args.encoder_layers = getattr(args, "encoder_layers", 8)
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
    args.backbone = getattr(args, "backbone", "transformer")
    args.kernel = getattr(args, "kernel", 'oled')
    args.kernel_size = getattr(args, "kernel_size", 64)

    # decoder
    args.decoder = getattr(args, "decoder", "TFM")
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.embed_dim = getattr(args, "embed_dim", 512)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 2048)
    args.attention_heads = getattr(args, "attention_heads", 64)
    args.rel_pos = getattr(args, "rel_pos", True)
    args.rel_pos_bins = getattr(args, "rel_pos_bins", 32)
    args.max_rel_pos = getattr(args, "max_rel_pos", 256)
    args.auto_regressive = getattr(args, "auto_regressive", True)


@register_model_architecture("cvae", "cvae-opv")
def base_architecture(args):
    # encoder
    args.encoder_layers = getattr(args, "encoder_layers", 8)
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
    args.backbone = getattr(args, "backbone", "transformer")
    args.kernel = getattr(args, "kernel", 'gaussian')
    args.kernel_size = getattr(args, "kernel_size", 64)

    # decoder
    args.decoder = getattr(args, "decoder", "TFM")
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.embed_dim = getattr(args, "embed_dim", 512)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 2048)
    args.attention_heads = getattr(args, "attention_heads", 64)
    args.rel_pos = getattr(args, "rel_pos", True)
    args.rel_pos_bins = getattr(args, "rel_pos_bins", 32)
    args.max_rel_pos = getattr(args, "max_rel_pos", 256)
    args.auto_regressive = getattr(args, "auto_regressive", True)