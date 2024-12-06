# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm
from .encoder import UniMolV1Model, UniMolOLEDModel, UniMolOPVModel, EGNN, UniMolQM9Model
from .encoder.unimol_qm9 import SE3CoordHead, SE3CoordClofHead
from .decoder import TransformerDecoder, GRUDecoder, MLPDecoder
from ..utils import inference_greedy, inference_beam, inference_top_p, inference_top_k
from tqdm import tqdm

logger = logging.getLogger(__name__)

ENCODER_REGISTER = {
    'unimolv1': UniMolV1Model,
    'unimol-qm9': UniMolQM9Model,
    'unimol-oled': UniMolOLEDModel,
    'unimol-opv': UniMolOPVModel,
}    

DECODER_REGISTER = {
    'TFM': TransformerDecoder,
    'GRU': GRUDecoder,
    'SE3': SE3CoordHead,
    'MLP': MLPDecoder,
    'EGNN': EGNN,
    'Clof': SE3CoordClofHead,
}

INFERENCE_REGISTER = {
    'GREEDY': inference_greedy,
    'BEAM':   inference_beam,
    'TOPP':   inference_top_p,
    'TOPK':   inference_top_k,
}

def cosine_beta_schedule(timesteps, s=0.008, **kwargs):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges

def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

def get_edges_with_mask(n_nodes, mask):
    rows, cols = [], []
    valid_nodes = torch.arange(n_nodes)[mask].long()
    for i in valid_nodes:
        for j in valid_nodes:
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges

def get_edges_batch_with_mask(n_nodes, batch_size, mask):
    all_rows, all_cols = [], []
    for b in range(batch_size):
        edges = get_edges_with_mask(n_nodes, mask[b])
        rows, cols = edges
        all_rows.extend(rows + b * n_nodes)
        all_cols.extend(cols + b * n_nodes)
    
    edge_attr = torch.ones(len(all_rows), 1)
    edges = [torch.LongTensor(all_rows), torch.LongTensor(all_cols)]
    return edges, edge_attr

def create_special_mask(src_tokens, special_idx_tensor):
    return torch.isin(src_tokens, special_idx_tensor.to(src_tokens))

def extract(coef, t):
    out = coef[t]
    return out.unsqueeze(-1)

def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=-1)

def compute_edge_type(x, num_type):
    src_edge_type = []
    for i in range(len(x)):
        node_input = x[i].clone()
        offset = node_input.view(-1, 1) * num_type + node_input.view(1, -1)
        src_edge_type.append(offset)
    return torch.stack(src_edge_type)
        
def compute_distance_matrix(x):
    # x is of shape (batch_size, N, 3)
    # Expand dims to enable broadcasting subtraction
    pairwise_diff = x.unsqueeze(1) - x.unsqueeze(2)  # shape becomes (batch_size, N, N, 3)

    # Compute pairwise distance
    pairwise_dist = torch.norm(pairwise_diff, dim=-1)  # Compute 2-norm, shape becomes (batch_size, N, N)

    return pairwise_dist

def center_pos(pos, mode='none'):
    # COM
    if mode == 'none':
        offset = torch.zeros_like(pos[0]) 
    elif mode == 'pos':
        non_zero_mask = (pos.sum(dim=-1) != 0).float().unsqueeze(-1)
        non_zero_count = non_zero_mask.sum(dim=1, keepdim=True)
        offset = (pos * non_zero_mask).sum(dim=1, keepdim=True) / (non_zero_count + 1e-8)
        pos = pos - offset        
    else:
        raise NotImplementedError
    return pos, offset

def index_to_log_onehot(x,num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = ((gumbel_noise + logits).argmax(dim=-1))
    return sample_index

def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas

def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

class VarianceSchedule(nn.Module):
    def __init__(self, schedule_name="linear_beta_schedule", beta_start=None, beta_end=None):
        super(VarianceSchedule, self).__init__()

        self.schedule_name = schedule_name

        beta_schedule_dict = {'linear_beta_schedule': linear_beta_schedule,
                              'cosine_beta_schedule': cosine_beta_schedule,
                              'quadratic_beta_schedule': quadratic_beta_schedule,
                              'sigmoid_beta_schedule': sigmoid_beta_schedule}

        if schedule_name in beta_schedule_dict:
            self.selected_schedule = beta_schedule_dict[schedule_name]
        else:
            raise ValueError('Function not found in dictionary')

        if beta_end and beta_start is None and schedule_name != "cosine_beta_schedule":
            self.beta_start = 0.0001
            self.beta_end = 0.02
        else:
            self.beta_start = beta_start
            self.beta_end = beta_end

    def forward(self, timesteps): # 1000
        # print(self.beta_start)
        return self.selected_schedule(timesteps=timesteps) if self.schedule_name == "cosine_beta_schedule" \
            else self.selected_schedule(timesteps=timesteps, beta_start=self.beta_start, beta_end=self.beta_end)


@register_model("ddpm")
class DDPM(BaseUnicoreModel):
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
        ),
        parser.add_argument(
            "--model_mean_type",
              type=str, 
              default="noise", 
              choices=["noise", "C0"]
        )
        parser.add_argument(
            "--beta_schedule",
            type=str,
            default="sigmoid"
        )
        parser.add_argument(
            "--beta_start",
            type=float,
            default=1.e-7, #1.e-7
        )
        parser.add_argument(
            "--beta_end",
            type=float,
            default=0.02,
        )
        parser.add_argument(
            "--v_beta_schedule",
            type=str,
            default="cosine"
        )
        parser.add_argument(
            "--v_beta_s",
            type=float,
            default=0.01
        )
        parser.add_argument(
            "--num_diffusion_timesteps",
            type=int,
            default=1000, #1000
        )
        parser.add_argument(
            "--loss_v_weight",
            type=float,
            default=100. #100 200
        )
        parser.add_argument(
            "--loss_pos_weight",
            type=float,
            default=1. #1 50
        )
        parser.add_argument(
            "--sample_time_method",
            type=str,
            default="symmetric",
            choices=["importance", "symmetric", "simple"]
        )
        parser.add_argument(
            "--time_emb_dim",
            type=int,
            default=0
        )
        parser.add_argument(
            "--time_emb_mode",
            type=str,
            default="simple"
        )
        parser.add_argument(
            "--center_pos_mode",
            type=str,
            default="pos"
        )
        parser.add_argument(
            "--CoM",
            type=bool,
            default=True # True
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.indices = {k:v for k,v in self.dictionary.indices.items()}
        self.special_idx = [self.indices[i] for i in dictionary.specials]
        self.__init__diffusion(args)
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self.v_inference = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim * 2),
            ShiftedSoftplus(),
            nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim),
            nn.Linear(args.encoder_embed_dim, len(dictionary)),
        )
        self.latent_dim = args.encoder_embed_dim
        self.num_classes = len(dictionary)
        self._num_updates = None
        self.encoder = ENCODER_REGISTER[args.encoder](self.args, dictionary)
        self.se3_coord = DECODER_REGISTER[args.decoder](self.args.encoder_attention_heads, self.args.activation_fn)
        # self.se3_coord = NonLinearHead(3, 3, self.args.activation_fn, args.encoder_embed_dim)
        # self.egnn = DECODER_REGISTER[args.decoder](self.args, dictionary)
        self.method_dict = {
            'SE3': self.compute_pred_pos_se3,
            'MLP': self.compute_pred_pos_mlp,
            'EGNN': self.compute_pred_pos_egnn,
            'Clof': self.compute_pred_pos_clof,
        }
        self.mlp = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim * 2),
            ShiftedSoftplus(),
            nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim),
            nn.Linear(args.encoder_embed_dim, 3),
        )


    def __init__diffusion(self, args):
        self.model_mean_type = args.model_mean_type
        self.loss_pos_weight = args.loss_pos_weight
        self.loss_v_weight = args.loss_v_weight
        self.sample_time_method = args.sample_time_method  # ['importance', 'symmetric']
        self.CoM = args.CoM
        self.num_timesteps = args.num_diffusion_timesteps


        if args.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(args.num_diffusion_timesteps, args.pos_beta_s) ** 2
            # print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=args.beta_schedule,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                num_diffusion_timesteps=args.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        self.betas = to_torch_const(betas)
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        # variance_schedule_func = VarianceSchedule(schedule_name=args.beta_schedule, beta_start=args.beta_start, beta_end=args.beta_end)
        # self.betas = variance_schedule_func(self.num_timesteps).numpy()
        # alphas = 1. - self.betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # atom type diffusion schedule in log space
        if args.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, args.v_beta_s)
        else:
            raise NotImplementedError
        
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))
        self.center_pos_mode = args.center_pos_mode
        self._edges_dict = {}
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

    def sample_time(self, batch_size, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(batch_size, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(batch_size // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:batch_size]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt
        
        elif method == 'simple':
            time_step = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt
        else:
            raise ValueError

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float().unsqueeze(-1)
        # if 0 in t:
        #     print('T=0', t)
        loss_v = torch.mean(mask * decoder_nll_v + (1 - mask) * kl_v, dim=-1)
        return loss_v
    
   # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t.unsqueeze(-1),
            log_1_min_alpha_t.unsqueeze(-1) - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t.unsqueeze(-1),
            log_1_min_cumprod_alpha.unsqueeze(-1) - np.log(self.num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t):
        log_qvt_v0 = self.q_v_pred(log_v0, t) # torch.argmax(log_qvt_v0) == src_tokens[0]
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    def _predict_x0_from_eps(self, xt, eps, t):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t).unsqueeze(-1) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t).unsqueeze(-1) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t).unsqueeze(-1) * x0 + \
                         extract(self.posterior_mean_ct_coef, t).unsqueeze(-1) * xt
        return pos_model_mean

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t):
        # q(vt-1 | vt, v0) = q(vt | vt-1, v0) * q(vt-1 | v0) / q(vt | v0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

    def compute_pred_pos_se3(self, pos_perturbed, encoder_outputs, **kwargs):
        if 'padding_mask' not in kwargs:
            padding_mask = None
        else:
            padding_mask = kwargs['padding_mask']
        return self.se3_coord(pos_perturbed, encoder_outputs[2], padding_mask)

    def compute_pred_pos_mlp(self, pos_perturbed, encoder_outputs, **kwargs):
        return self.se3_coord(pos_perturbed.type_as(encoder_outputs[0]))

    def compute_pred_pos_clof(self, pos_perturbed, encoder_outputs, **kwargs):
        if 'padding_mask' not in kwargs:
            padding_mask = None
        else:
            padding_mask = kwargs['padding_mask']
        return self.se3_coord(pos_perturbed, encoder_outputs[2], padding_mask)

    def compute_pred_pos_egnn(self, pos_perturbed, encoder_outputs, **kwargs):
        pos_perturbed = pos_perturbed.type_as(encoder_outputs[0])
        batch_size, n_nodes = pos_perturbed.size(0), pos_perturbed.size(1)
        edges = self.get_adj_matrix(n_nodes, batch_size, pos_perturbed.device)
        edges = [x.to(pos_perturbed.device) for x in edges]
        if self.edge_mask is not None:
            return self.se3_coord(
                encoder_outputs[0].view(batch_size * n_nodes, -1).clone() * (~self.egnn_node_mask),
                pos_perturbed.view(batch_size * n_nodes, -1).clone() * (~self.egnn_node_mask),
                edges,
                edge_mask=(~self.edge_mask),
                node_mask=(~self.egnn_node_mask),
            )[1].reshape(batch_size, n_nodes, -1)
        else:
            return self.se3_coord(
                encoder_outputs[0].view(batch_size*n_nodes, -1).clone(), 
                pos_perturbed.view(batch_size*n_nodes, -1).clone(), 
                edges, 
            )[1].reshape(batch_size, n_nodes, -1)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_coord,
        time_step=None,
        **kwargs
    ):  
        
        batch_size = src_tokens.size(0)
        n_nodes = src_tokens.size(1)

        # special mask
        special_idx_tensor = torch.tensor(self.special_idx)
        mask = create_special_mask(src_tokens, special_idx_tensor)

        # edge mask
        edge_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask.to(edge_mask)
        self.egnn_node_mask = mask.view(batch_size*n_nodes, 1)
        self.edge_mask = edge_mask.view(batch_size*n_nodes*n_nodes, 1)
        
        if self.CoM:
            src_coord, offset = center_pos(
                src_coord*(~mask).unsqueeze(-1).float(), mode=self.center_pos_mode
            )
        
        # 1. sample noise
        if time_step is None:
            time_step, _ = self.sample_time(batch_size, src_coord.device, self.sample_time_method)

        # 2. add noise to coord and v
        a = self.alphas_cumprod.index_select(0, time_step)
        a_pos = a.reshape(batch_size, *((1,) * (len(src_coord.shape) - 1)))

        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        pos_noise = torch.randn_like(src_coord) # use randn_like rather than zero_like
        pos_perturbed = a_pos.sqrt() * src_coord + (1. - a_pos).sqrt() * pos_noise

        # Vt = a * V0 + (1-a) / K
        log_v0 = index_to_log_onehot(src_tokens, self.embed_tokens.weight.shape[0]) # torch.argmax(log_v0, -1) == src_tokens
        v_perturbed, log_vt = self.q_v_sample(log_v0, time_step)

        # init src_distance / src_edge_type
        src_distance = compute_distance_matrix(pos_perturbed)
        src_edge_type = compute_edge_type(v_perturbed, self.num_classes)

        encoder_outputs = self.encoder(
                src_tokens=v_perturbed,
                src_distance=src_distance,
                src_coord=pos_perturbed,  # not used
                src_edge_type=src_edge_type,
                padding_mask=mask,
                **kwargs
        )

        pred_v = self.v_inference(encoder_outputs[0])
        

        # pred_pos = self.mlp(encoder_outputs[0])
        pred_pos = self.method_dict[self.args.decoder](pos_perturbed, encoder_outputs, **{'padding_mask': mask})



        pred_pos_noise = pred_pos - pos_perturbed

        # pos
        if self.model_mean_type == 'C0':
            pos_model_mean = self.q_pos_posterior(
                x0=pred_pos, xt=pos_perturbed, t=time_step)
        elif self.model_mean_type == 'noise':
            pos0_from_e = self._predict_x0_from_eps(
                xt=a_pos, eps=pred_pos_noise, t=time_step)
            pos_model_mean = self.q_pos_posterior(
                x0=pos0_from_e, xt=a_pos, t=time_step)
        else:
            raise ValueError
        
        # atom pos loss
        if self.model_mean_type == 'C0':
            target, pred = src_coord, pred_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise        
        else:
            raise ValueError
        
        # mask specials tokens
        mask_loss = (~mask).unsqueeze(-1).float()
        pred_masked = pred * mask_loss
        target_masked = target * mask_loss
        loss_pos = F.mse_loss(pred_masked, target_masked, reduction='sum') / (~mask).float().sum()


        # atom type loss
        log_v_recon = F.log_softmax(pred_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_v_recon, log_vt, time_step)
        log_v_true_prob = self.q_v_posterior(log_v0, log_vt, time_step)
        v_next = log_sample_categorical(log_v_model_prob)
        
        kl_v = self.compute_v_Lt(
            log_v_model_prob=log_v_model_prob * mask_loss, 
            log_v0=log_v0 * mask_loss,
            log_v_true_prob=log_v_true_prob * mask_loss, 
            t=time_step,
        )

        loss_v = kl_v.mean()

        loss = loss_pos * self.loss_pos_weight + loss_v * self.loss_v_weight

        pos_log_variance = extract(self.posterior_logvar, time_step)

        # no noise when t == 0
        nonzero_mask = (1 - (time_step == 0).float()).unsqueeze(-1).unsqueeze(-1)

        pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance.unsqueeze(-1)).exp() * torch.randn_like(src_coord)

        if self.CoM:
            pos_next = pos_next + offset.to(pos_next) 

        return {
            'loss_pos': loss_pos * self.loss_pos_weight,  # used for backpropagation
            'loss_v': loss_v * self.loss_v_weight,  # used for backpropagation
            'loss': loss,  # used for backpropagation
            'x0': src_coord.detach(),  # detached from the computation graph
            'pred_pos': pred_pos.detach(),  # detached from the computation graph
            'pred_v': pred_v.detach(),  # detached from the computation graph
            'pred_pos_noise': (pred_pos - pos_perturbed).detach(),  # detached from the computation graph
            'v_recon': torch.argmax(pred_v, dim=-1).detach(),  # detached from the computation graph
            'pos_next': pos_next.detach(), # detached from the computation graph
            'v_next': v_next.detach(), # detached from the computation graph
        }

    def inference(
            self, 
            bsz, 
            max_len,
            **kwargs
        ):

        device = self.embed_tokens.weight.device
        
        self.egnn_node_mask = None
        self.edge_mask = None
        
        ## init pos
        init_pos = torch.randn((bsz, max_len, 3))

        ## init atom type
        uniform_logits = torch.zeros(bsz, max_len, self.num_classes)
        init_v = log_sample_categorical(uniform_logits)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        pos, v = init_pos.to(device), init_v.to(device)

        time_seq = list(reversed(range(0, self.num_timesteps)))

        variance_list= [] 
        variance_gt_list = []
        sqrt_one_minus_alpha_cum = []
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(bsz,), fill_value=i, dtype=torch.long, device=device)

            if self.CoM:
                pos, offset = center_pos(pos, mode=self.center_pos_mode)        

            # init src_distance / src_edge_type
            src_distance = compute_distance_matrix(pos)
            src_edge_type = compute_edge_type(v, self.num_classes)

            encoder_outputs = self.encoder(src_tokens=v,
                                           src_distance=src_distance,
                                           src_coord=pos, # not used
                                           src_edge_type=src_edge_type,
                                           **kwargs)
            
            pred_v = self.v_inference(encoder_outputs[0])

            pred_pos = self.method_dict[self.args.decoder](pos, encoder_outputs)

            # Compute posterior mean and variance
            if self.model_mean_type == 'C0':
                pos0_from_e = pred_pos
                v0_from_e = pred_v
            elif self.model_mean_type == 'noise':
                pred_pos_noise = pred_pos - pos
                pos0_from_e = self._predict_x0_from_eps(xt=pos, eps=pred_pos_noise, t=t)
                v0_from_e = pred_v
            else:
                raise ValueError
            

            # origin
            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=pos, t=t)

            ########################
            # test gt pos need to rm
            # pos_model_mean = self.q_pos_posterior(x0=kwargs['gt_pos'], xt=pos, t=t)
            ########################

            pos_log_variance = extract(self.posterior_logvar, t)

            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float()).unsqueeze(-1).unsqueeze(-1)

            pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance.unsqueeze(-1)).exp() * torch.randn_like(pos)
            
            if self.CoM:
                ori_pos = pos + offset.to(pos)            
                pos = pos_next + offset.to(pos)
            else:
                ori_pos = pos 
                pos = pos_next

            log_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_v = index_to_log_onehot(v, self.num_classes)
            log_model_prob = self.q_v_posterior(log_v_recon, log_v, t)
            v_next = log_sample_categorical(log_model_prob)

            ##################
            ### need to rm ### #### test gt_v
            # log_v0 = index_to_log_onehot(kwargs['gt_v'], self.embed_tokens.weight.shape[0])
            # log_model_prob = self.q_v_posterior(log_v0, log_v, t)
            # v_next = log_sample_categorical(log_model_prob)         

            ###################

            v0_pred_traj.append(log_v_recon.clone().cpu())
            vt_pred_traj.append(log_model_prob.clone().cpu())
            v = v_next



            pos_traj.append(ori_pos.clone().cpu())
            v_traj.append(v_next.clone().cpu())

            variance_list.append(pos_next - extract(self.sqrt_alphas_cumprod, t).unsqueeze(-1) * pos0_from_e)

            ############### rm 
            # variance_gt_list.append(pos_next - extract(self.sqrt_alphas_cumprod, t).unsqueeze(-1) * kwargs['gt_pos'])
            ###############
            sqrt_one_minus_alpha_cum.append(extract(self.sqrt_one_minus_alphas_cumprod, t).unsqueeze(-1))


        return {
            'pos': pos,
            'v': v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
            'variance_list': variance_list,
            'variance_gt_list': variance_gt_list,
            'sqrt_one_minus_alpha_cum': sqrt_one_minus_alpha_cum,
        }


    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

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


@register_model_architecture("ddpm", "ddpm")
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

@register_model_architecture("ddpm", "ddpm-qm9")
def base_architecture(args):
    # encoder
    args.encoder_layers = getattr(args, "encoder_layers", 8) # 8
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
    args.kernel = getattr(args, "kernel", 'gaussian') # numerical
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

@register_model_architecture("ddpm", "ddpm-oled")
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


@register_model_architecture("ddpm", "ddpm-opv")
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