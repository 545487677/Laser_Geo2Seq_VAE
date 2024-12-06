# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import logging
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm
from .encoder import UniMolV1Model, UniMolOLEDModel, UniMolOPVModel,  UniMolQM9Model
from .encoder.unimol_oled import SE3CoordHead
from .decoder import TransformerDecoder, GRUDecoder, MLPDecoder
from ..utils import inference_greedy, inference_beam, inference_top_p, inference_top_k,save_xyz_file, check_stabilityori, save_xyz_fileori

from tqdm import tqdm
from torch.distributions.categorical import Categorical

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
}

INFERENCE_REGISTER = {
    'GREEDY': inference_greedy,
    'BEAM':   inference_beam,
    'TOPP':   inference_top_p,
    'TOPK':   inference_top_k,
}

HISTOGRAM = {
    # 'qm9': {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
    #             15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
    #             8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 1},

    'qm9':{
        'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
        'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
        'n_nodes':{17: 18330, 16: 17827, 15: 17388, 14: 14262, 19: 13187, 18: 12597, 13: 10636, 12: 7093, 21: 6359, 20: 4482, 11: 4250, 10: 2332, 23: 1923, 9: 1146, 22: 712, 8: 526, 25: 356, 7: 193, 6: 69, 24: 59, 27: 35, 5: 21, 4: 12, 3: 5, 2: 4, 1: 2}
    },
    'qm9_geoldm':{
            'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'n_nodes': {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
                15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
                8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 1},
    'max_n_nodes': 29,
    'atom_types': {1: 635559, 2: 101476, 0: 923537, 3: 140202, 4: 2323},
    'distances': [903054, 307308, 111994, 57474, 40384, 29170, 47152, 414344, 2202212, 573726,
                  1490786, 2970978, 756818, 969276, 489242, 1265402, 4587994, 3187130, 2454868, 2647422,
                  2098884,
                  2001974, 1625206, 1754172, 1620830, 1710042, 2133746, 1852492, 1415318, 1421064, 1223156,
                  1322256,
                  1380656, 1239244, 1084358, 981076, 896904, 762008, 659298, 604676, 523580, 437464, 413974,
                  352372,
                  291886, 271948, 231328, 188484, 160026, 136322, 117850, 103546, 87192, 76562, 61840,
                  49666, 43100,
                  33876, 26686, 22402, 18358, 15518, 13600, 12128, 9480, 7458, 5088, 4726, 3696, 3362, 3396,
                  2484,
                  1988, 1490, 984, 734, 600, 456, 482, 378, 362, 168, 124, 94, 88, 52, 44, 40, 18, 16, 8, 6,
                  2,
                  0, 0, 0, 0,
                  0,
                  0, 0],
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'with_h': True,
    'all_species': [1, 6, 7, 8, 9],
    },

    'oled_geoldm': 
    {   
        'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'Se': 5, 'B': 6},
        'atom_decoder': ['H', 'C', 'N', 'O', 'S', 'Se', 'B'],
        'n_nodes':{110: 131370, 109: 112706, 108: 98004, 111: 87531, 98: 69419, 107: 58851, 114: 58626, 102: 50776, 99: 49013, 106: 48299, 112: 48053, 80: 47673, 97: 46866, 100: 41622, 96: 39004, 81: 38857, 79: 36009, 84: 35692, 82: 33545, 78: 32931, 94: 29512, 104: 28507, 77: 26954, 68: 26922, 95: 25821, 86: 24818, 76: 24266, 72: 21899, 83: 20599, 105: 20334, 69: 20116, 67: 18512, 92: 18206, 74: 17949, 70: 17785, 101: 17542, 93: 17496, 75: 17108, 88: 15112, 66: 14743, 90: 14691, 103: 13021, 87: 12802, 65: 12172, 85: 11820, 113: 11530, 71: 10794, 56: 10523, 73: 10459, 58: 8498, 57: 7869, 52: 7585, 64: 7419, 60: 7004, 91: 6893, 54: 6574, 89: 6457, 55: 6358, 48: 6213, 46: 6109, 63: 5371, 53: 5327, 50: 5207, 44: 5185, 45: 5055, 51: 4460, 49: 4223, 62: 4036, 47: 3962, 59: 3647, 38: 3587, 43: 3174, 40: 3088, 36: 2455, 37: 2354, 42: 2315, 39: 1871, 116: 1733, 61: 1702, 41: 1446, 35: 1401, 34: 639, 118: 496, 33: 199, 32: 156, 120: 129, 26: 105, 27: 98, 28: 78, 29: 58, 25: 37, 31: 37, 30: 34, 122: 32, 117: 27, 24: 13, 23: 11, 121: 8, 123: 2},
        'max_n_nodes': 123,
        'all_species': [1, 6, 7, 8, 16, 34, 5],
    },
   'oled_geoldm1': 
    {   
        'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'Se': 5, 'B': 6},
        'atom_decoder': ['H', 'C', 'N', 'O', 'S', 'Se', 'B'],
        'n_nodes':{110: 131370, 109: 112706, 108: 98004, 111: 87531, 98: 69419, 107: 58851, 114: 58626, 102: 50776, 99: 49013, 106: 48299, 112: 48053, 80: 47673, 97: 46866, 100: 41622, 96: 39004, 81: 38857, 79: 36009, 84: 35692, 82: 33545, 78: 32931, 94: 29512, 104: 28507, 77: 26954, 68: 26922, 95: 25821, 86: 24818, 76: 24266, 72: 21899, 83: 20599, 105: 20334, 69: 20116, 67: 18512, 92: 18206, 74: 17949, 70: 17785, 101: 17542, 93: 17496, 75: 17108, 88: 15112, 66: 14743, 90: 14691, 103: 13021, 87: 12802, 65: 12172, 85: 11820, 113: 11530, 71: 10794, 56: 10523, 73: 10459, 58: 8498, 57: 7869, 52: 7585, 64: 7419, 60: 7004, 91: 6893, 54: 6574, 89: 6457, 55: 6358, 48: 6213, 46: 6109, 63: 5371, 53: 5327, 50: 5207, 44: 5185, 45: 5055, 51: 4460, 49: 4223, 62: 4036, 47: 3962, 59: 3647, 38: 3587, 43: 3174, 40: 3088, 36: 2455, 37: 2354, 42: 2315, 39: 1871, 116: 1733, 61: 1702, 41: 1446, 35: 1401, 34: 639, 118: 496, 33: 199, 32: 156, 120: 129, 26: 105, 27: 98, 28: 78, 29: 58, 25: 37, 31: 37, 30: 34, 122: 32, 117: 27, 24: 13, 23: 11, 121: 8, 123: 2},
        'max_n_nodes': 123,
        'all_species': [1, 6, 7, 8, 16, 34, 5],
    },
   'opv_geoldm': 
    {   
        'atom_encoder': {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'Se': 34, 'B': 5, 'F': 9, 'Si': 14},
        'atom_decoder': ['H', 'C', 'N', 'O', 'S', 'Se', 'B','F', 'Si'],
        'n_nodes':{132: 38825, 136: 38807, 134: 38656, 138: 38463, 130: 38239, 128: 38015, 140: 37889, 142: 37606, 126: 37232, 144: 36843, 146: 36441, 124: 36334, 148: 35907, 122: 35439, 150: 35064, 120: 33950, 152: 33876, 154: 33558, 118: 32818, 156: 32658, 158: 31713, 116: 31204, 160: 30321, 114: 29508, 162: 29166, 164: 28463, 112: 27816, 166: 27299, 168: 25856, 110: 25793, 170: 24519, 108: 23903, 172: 23362, 174: 22134, 106: 21746, 176: 20959, 178: 20007, 104: 19537, 180: 18565, 182: 17637, 102: 17426, 184: 16597, 186: 15536, 100: 15163, 188: 14592, 190: 13404, 98: 13049, 192: 12594, 194: 11600, 96: 11031, 196: 10885, 198: 9904, 94: 9089, 200: 9026, 202: 8426, 204: 7840, 92: 7304, 206: 6779, 208: 6350, 90: 5784, 210: 5772, 212: 5390, 214: 4730, 88: 4406, 141: 4268, 137: 4265, 139: 4240, 135: 4232, 216: 4184, 145: 4176, 131: 4130, 147: 4116, 133: 4112, 143: 4096, 129: 4067, 149: 4055, 151: 3994, 155: 3927, 127: 3902, 153: 3876, 157: 3846, 218: 3797, 125: 3770, 159: 3656, 161: 3598, 163: 3567, 123: 3552, 220: 3466, 165: 3395, 167: 3341, 121: 3321, 86: 3286, 169: 3168, 171: 3168, 173: 3082, 222: 3065, 119: 3057, 224: 2897, 175: 2888, 177: 2752, 117: 2716, 179: 2583, 181: 2575, 226: 2517, 115: 2410, 84: 2371, 228: 2304, 183: 2299, 185: 2136, 113: 2107, 187: 2043, 230: 1923, 191: 1855, 189: 1796, 111: 1758, 232: 1665, 193: 1628, 82: 1625, 234: 1571, 195: 1500, 109: 1444, 236: 1422, 197: 1374, 199: 1306, 201: 1294, 238: 1161, 107: 1151, 240: 1144, 80: 1096, 203: 1044, 207: 992, 242: 988, 205: 974, 244: 930, 105: 866, 211: 808, 209: 793, 246: 741, 78: 692, 213: 688, 103: 656, 248: 656, 215: 609, 217: 606, 250: 590, 219: 549, 252: 516, 101: 460, 221: 457, 254: 450, 223: 424, 76: 408, 225: 373, 256: 372, 258: 370, 227: 359, 229: 314, 260: 310, 99: 304, 262: 291, 231: 258, 233: 258, 74: 232, 235: 230, 264: 213, 266: 208, 237: 205, 97: 196, 268: 192, 239: 189, 270: 185, 241: 160, 272: 155, 243: 133, 245: 125, 72: 124, 274: 119, 95: 116, 276: 108, 278: 106, 249: 103, 247: 91, 251: 88, 280: 81, 253: 74, 282: 68, 257: 66, 70: 60, 93: 60, 284: 57, 259: 55, 286: 52, 255: 51, 261: 51, 288: 50, 290: 39, 267: 37, 292: 32, 277: 30, 298: 29, 68: 28, 91: 28, 263: 28, 294: 27, 296: 26, 265: 25, 269: 24, 275: 16, 271: 15, 302: 15, 300: 14, 273: 13, 279: 13, 304: 13, 285: 12, 306: 12, 281: 11, 283: 10, 287: 9, 295: 9, 66: 8, 89: 8, 314: 8, 289: 7, 301: 7, 311: 7, 291: 6, 293: 6, 308: 6, 310: 6, 316: 6, 318: 5, 303: 4, 320: 4, 324: 3, 305: 2, 312: 2, 326: 2, 297: 1, 299: 1, 313: 1, 315: 1, 317: 1, 321: 1, 322: 1, 328: 1, 329: 1, 338: 1},
        'max_n_nodes': 123,
        'all_species': [1, 6, 7, 8, 16, 34, 5, 9, 14],
    }    
}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False




def create_special_mask(src_tokens, special_idx_tensor):
    return torch.isin(src_tokens, special_idx_tensor.to(src_tokens))

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    row = [int(i) for i in row]
    col = [int(i) for i in col]
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
            [[np.cos(theta), -np.sin(theta), 0.],
            [np.sin(theta), np.cos(theta), 0.],
            [0., 0., 1.]]
        ).float()
    Qx = torch.tensor(
            [[1., 0., 0.],
            [0., np.cos(theta), -np.sin(theta)],
            [0., np.sin(theta), np.cos(theta)]]
        ).float()
    Qy = torch.tensor(
            [[np.cos(theta), 0., np.sin(theta)],
            [0., 1., 0.],
            [-np.sin(theta), 0., np.cos(theta)]]
        ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]

# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5
    )


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
            (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask
        )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d



def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask): #(64, 23, 3)
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers) if n_layers > 0 else float(coords_range)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, 
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', 
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, 
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf,  act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, 
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.context_node_nf = context_node_nf
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, xh.device)
        edges = [x.to(xh.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(xh.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)


        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

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


class EGNN_encoder_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64,
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True):
        '''
        :param in_node_nf: Number of invariant features for input nodes.'''
        super().__init__()

        include_charges = int(include_charges)
        num_classes = in_node_nf - include_charges

        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, out_node_nf=hidden_nf, 
                in_edge_nf=1, hidden_nf=hidden_nf,  act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, out_node_nf=hidden_nf + 3, 
                in_edge_nf=0, hidden_nf=hidden_nf, 
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)
        
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, out_node_nf * 2 + 1))

        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.n_dims = n_dims
        self._edges_dict = {}
        # self.condition_time = condition_time

        self.out_node_nf = out_node_nf

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):      
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, xh.device)
        edges = [x.to(xh.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(xh.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = x_final * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        h_final = self.final_mlp(h_final)
        h_final = h_final * node_mask if node_mask is not None else h_final
        h_final = h_final.view(bs, n_nodes, -1)

        vel_mean = vel
        vel_std = h_final[:, :, :1].sum(dim=1, keepdim=True).expand(-1, n_nodes, -1) # torch.Size([64, 25, 3])
        vel_std = torch.exp(0.5 * vel_std)

        h_mean = h_final[:, :, 1:1 + self.out_node_nf]
        h_std = torch.exp(0.5 * h_final[:, :, 1 + self.out_node_nf:])

        if torch.any(torch.isnan(vel_std)):
            print('Warning: detected nan in vel_std, resetting to one.')
            vel_std = torch.ones_like(vel_std)
        
        if torch.any(torch.isnan(h_std)):
            print('Warning: detected nan in h_std, resetting to one.')
            h_std = torch.ones_like(h_std)
        
        # Note: only vel_mean and h_mean are correctly masked
        # vel_std and h_std are not masked, but that's fine:

        # For calculating KL: vel_std will be squeezed to 1D
        # h_std will be masked

        # For sampling: both stds will be masked in reparameterization

        return vel_mean, vel_std, h_mean, h_std
    
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


class EGNN_decoder_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, hidden_nf=64,
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True):
        super().__init__()

        include_charges = int(include_charges)
        num_classes = out_node_nf - include_charges

        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, out_node_nf=out_node_nf, 
                in_edge_nf=1, hidden_nf=hidden_nf, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, out_node_nf=out_node_nf + 3, 
                in_edge_nf=0, hidden_nf=hidden_nf, 
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.n_dims = n_dims
        self._edges_dict = {}
        # self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, xh.device)
        edges = [x.to(xh.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(xh.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = x_final * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if node_mask is not None:
            h_final = h_final * node_mask
        h_final = h_final.view(bs, n_nodes, -1)

        return vel, h_final
    
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


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma

# class DistributionNodes:
#     def __init__(self, histogram):

#         self.n_nodes = []
#         prob = []
#         self.keys = {}
#         for i, nodes in enumerate(histogram):
#             self.n_nodes.append(nodes)
#             self.keys[nodes] = i
#             prob.append(histogram[nodes])

#         ## in case lose index 
#         for i in range(max(histogram) + 1):
#             if i not in self.keys:
#                 self.keys[i] = 1
#             if i not in self.n_nodes:
#                 self.n_nodes.append(i)


#         self.n_nodes = torch.tensor(self.n_nodes)
#         prob = np.array(prob)
#         prob = prob/np.sum(prob)

#         self.prob = torch.from_numpy(prob).float()

#         entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
#         print("Entropy of n_nodes: H[N]", entropy.item())

#         self.m = Categorical(torch.tensor(prob))

#     def sample(self, n_samples=1):
#         idx = self.m.sample((n_samples,))
#         return self.n_nodes[idx]

#     def log_prob(self, batch_n_nodes):
#         assert len(batch_n_nodes.size()) == 1

#         idcs = [self.keys[i.item()] for i in batch_n_nodes]
#         idcs = torch.tensor(idcs).to(batch_n_nodes.device)

#         log_p = torch.log(self.prob + 1e-30)

#         log_p = log_p.to(batch_n_nodes.device)

#         log_probs = log_p[idcs]

#         return log_probs

class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        # idcs = [self.keys[i.item()] for i in batch_n_nodes]
        # idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        # log_p = torch.log(self.prob + 1e-30)

        # log_p = log_p.to(batch_n_nodes.device)

        # log_probs = log_p[idcs]

        ## 以防万一
        # Use the get method of the dictionary with a default value of 1
        idcs = [self.keys.get(i.item(), 1) for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)
        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs

class DistributionProperty:
    def __init__(self, num_atoms, condition_data, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(num_atoms,
                                    condition_data[prop],
                                    self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins #min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val

class EnVariationalDiffusion(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """
    def __init__(
            self,
            dynamics: EGNN_dynamics_QM9, in_node_nf: int, n_dims: int,
            timesteps: int = 1000, parametrization='eps', noise_schedule='learned',
            noise_precision=1e-4, loss_type='vlb', norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.), include_charges=True):
        super().__init__()

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        self.include_charges = include_charges
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

        if noise_schedule != 'learned':
            self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    def phi(self, x, t, node_mask, edge_mask, context):
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):
        # Parse from z
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims+self.num_classes]
        h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)

        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error

    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)."""
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}
        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        # Discrete properties are predicted directly from z_t.
        z_h_cat = z_t[:, :, self.n_dims:-1] if self.include_charges else z_t[:, :, self.n_dims:]
        z_h_int = z_t[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)

        # Take only part over x.
        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x)

        # Compute delta indicator masks.
        h_integer = torch.round(h['integer'] * self.norm_values[2] + self.norm_biases[2]).long()
        onehot = h['categorical'] * self.norm_values[1] + self.norm_biases[1]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        log_ph_integer = torch.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon)

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps

        assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)

        # Neural net prediction.
        net_out = self.phi(z_t, t, node_mask, edge_mask, context)

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, eps)

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
            z_0 = alpha_0 * xh + sigma_0 * eps_0

            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=True)

        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        # tensor(-0.0048, device='cuda:0')
        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)
        # tensor(-0.0026, device='cuda:0')
        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )
        return zs

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in tqdm(reversed(range(0, self.T)), total=self.T):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise) # z tensor(0.0246, device='cuda:0')

        assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = remove_mean_with_mask(x, node_mask)

        return x, h

    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in tqdm(reversed(range(0, self.T)), total=self.T):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context)

            assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        assert_mean_zero_with_mask(x[:, :, :self.n_dims], node_mask)

        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {
            'log_SNR_max': log_SNR_max.item(),
            'log_SNR_min': log_SNR_min.item()}
        print(info)

        return info

class EnHierarchicalVAE(torch.nn.Module):
    """
    The E(n) Hierarchical VAE Module.
    """
    def __init__(
            self,
            encoder: EGNN_encoder_QM9,
            decoder: EGNN_decoder_QM9,
            in_node_nf: int, n_dims: int, latent_node_nf: int,
            kl_weight: float,
            norm_values=(1., 1., 1.), norm_biases=(None, 0., 0.), 
            include_charges=True):
        super().__init__()

        self.include_charges = include_charges

        self.encoder = encoder
        self.decoder = decoder

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.latent_node_nf = latent_node_nf
        self.num_classes = self.in_node_nf - self.include_charges
        self.kl_weight = kl_weight

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def compute_reconstruction_error(self, xh_rec, xh):
        """Computes reconstruction error."""

        bs, n_nodes, dims = xh.shape

        # Error on positions.
        x_rec = xh_rec[:, :, :self.n_dims]
        x = xh[:, :, :self.n_dims]
        error_x = sum_except_batch((x_rec - x) ** 2)
        
        # Error on classes.
        h_cat_rec = xh_rec[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_cat = xh[:, :, self.n_dims:self.n_dims + self.num_classes]
        h_cat_rec = h_cat_rec.reshape(bs * n_nodes, self.num_classes)
        h_cat = h_cat.reshape(bs * n_nodes, self.num_classes)
        error_h_cat = F.cross_entropy(h_cat_rec, h_cat.argmax(dim=1), reduction='none')
        error_h_cat = error_h_cat.reshape(bs, n_nodes, 1)
        error_h_cat = sum_except_batch(error_h_cat)
        # error_h_cat = sum_except_batch((h_cat_rec - h_cat) ** 2)

        # Error on charges.
        if self.include_charges:
            h_int_rec = xh_rec[:, :, -self.include_charges:]
            h_int = xh[:, :, -self.include_charges:]
            error_h_int = sum_except_batch((h_int_rec - h_int) ** 2)
        else:
            error_h_int = 0.
        
        error = error_x + error_h_cat + error_h_int

        if self.training:
            denom = (self.n_dims + self.in_node_nf) * xh.shape[1]
            error = error / denom

        return error
    
    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps
    
    def compute_loss(self, x, h, node_mask, edge_mask, context):
        """Computes an estimator for the variational lower bound."""

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)

        # Encoder output.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encode(x, h, node_mask, edge_mask, context)
        
        # KL distance.
        # KL for invariant features.
        zeros, ones = torch.zeros_like(z_h_mu), torch.ones_like(z_h_sigma)
        loss_kl_h = gaussian_KL(z_h_mu, ones, zeros, ones, node_mask)
        # KL for equivariant features.
        assert z_x_sigma.mean(dim=(1,2), keepdim=True).expand_as(z_x_sigma).allclose(z_x_sigma, atol=1e-7)
        zeros, ones = torch.zeros_like(z_x_mu), torch.ones_like(z_x_sigma.mean(dim=(1,2)))
        subspace_d = self.subspace_dimensionality(node_mask)
        loss_kl_x = gaussian_KL_for_dimension(z_x_mu, ones, zeros, ones, subspace_d)
        loss_kl = loss_kl_h + loss_kl_x

        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        z_xh = self.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        # z_xh = z_xh_mean
        assert_correctly_masked(z_xh, node_mask)
        assert_mean_zero_with_mask(z_xh[:, :, :self.n_dims], node_mask)

        # Decoder output (reconstruction).
        x_recon, h_recon = self.decoder._forward(z_xh, node_mask, edge_mask, context)
        xh_rec = torch.cat([x_recon, h_recon], dim=2)
        loss_recon = self.compute_reconstruction_error(xh_rec, xh)

        # Combining the terms
        assert loss_recon.size() == loss_kl.size()
        loss = loss_recon + self.kl_weight * loss_kl

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'loss_t': loss.squeeze(), 'rec_error': loss_recon.squeeze()}

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the ELBO if training. And if eval then always computes NLL.
        """

        loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context)

        neg_log_pxh = loss

        return neg_log_pxh

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.latent_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z
    
    def encode(self, x, h, node_mask=None, edge_mask=None, context=None):
        """Computes q(z|x)."""

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)

        assert_mean_zero_with_mask(xh[:, :, :self.n_dims], node_mask)

        # Encoder output.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encoder._forward(xh, node_mask, edge_mask, context)

        bs, _, _ = z_x_mu.size()
        sigma_0_x = torch.ones(bs, 1, 1).to(z_x_mu) * 0.0032
        sigma_0_h = torch.ones(bs, 1, self.latent_node_nf).to(z_h_mu) * 0.0032

        return z_x_mu, sigma_0_x, z_h_mu, sigma_0_h
    
    def decode(self, z_xh, node_mask=None, edge_mask=None, context=None):
        """Computes p(x|z)."""

        # Decoder output (reconstruction).
        x_recon, h_recon = self.decoder._forward(z_xh, node_mask, edge_mask, context)
        assert_mean_zero_with_mask(x_recon, node_mask)

        xh = torch.cat([x_recon, h_recon], dim=2)

        x = xh[:, :, :self.n_dims]
        assert_correctly_masked(x, node_mask)

        h_int = xh[:, :, -1:] if self.include_charges else torch.zeros(0).to(xh)
        h_cat = xh[:, :, self.n_dims:-1]  # TODO: have issue when include_charges is False
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}

        return x, h

    @torch.no_grad()
    def reconstruct(self, x, h, node_mask=None, edge_mask=None, context=None):
        pass

    def log_info(self):
        """
        Some info logging of the model.
        """
        info = None
        print(info)

        return info


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class EnLatentDiffusion(EnVariationalDiffusion):
    """
    The E(n) Latent Diffusion Module.
    """
    def __init__(self, **kwargs):
        vae = kwargs.pop('vae')
        trainable_ae = kwargs.pop('trainable_ae', False)
        super().__init__(**kwargs)

        # Create self.vae as the first stage model.
        self.trainable_ae = trainable_ae
        self.instantiate_first_stage(vae)
    
    def unnormalize_z(self, z, node_mask):
        # Overwrite the unnormalize_z function to do nothing (for sample_chain). 

        # Parse from z
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims+self.num_classes]
        h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        # x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output
    
    def log_constants_p_h_given_z0(self, h, node_mask):
        """Computes p(h|z0)."""
        batch_size = h.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_h = n_nodes * self.n_dims

        zeros = torch.zeros((h.size(0), 1), device=h.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_h * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = xh[:, :, :self.n_dims]

        # h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        # x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        # h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        # h_int = torch.round(h_int).long() * node_mask

        # Make the data structure compatible with the EnVariationalDiffusion sample() and sample_chain().
        h = {'integer': xh[:, :, self.n_dims:], 'categorical': torch.zeros(0).to(xh)}
        
        return x, h
    
    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):

        # Computes the error for the distribution N(latent | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_pxh_given_z_without_constants = -0.5 * self.compute_error(net_out, gamma_0, eps)

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_pxh_given_z_without_constants

        return log_p_xh_given_z
    
    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """

        # Encode data to latent space.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.vae.encode(x, h, node_mask, edge_mask, context) # torch.Size([64, 23, 3]) torch.Size([64, 23, 5])
        # Compute fixed sigma values.
        t_zeros = torch.zeros(size=(x.size(0), 1), device=x.device)
        gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
        sigma_0 = self.sigma(gamma_0, x)

        # Infer latent z.
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = sigma_0
        # z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        z_xh = self.vae.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        # z_xh = z_xh_mean
        z_xh = z_xh.detach()  # Always keep the encoder fixed.
        assert_correctly_masked(z_xh, node_mask)

        # Compute reconstruction loss.
        if self.trainable_ae:
            xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
            # Decoder output (reconstruction).
            x_recon, h_recon = self.vae.decoder._forward(z_xh, node_mask, edge_mask, context)
            xh_rec = torch.cat([x_recon, h_recon], dim=2)
            loss_recon = self.vae.compute_reconstruction_error(xh_rec, xh)
        else:
            loss_recon = 0

        z_x = z_xh[:, :, :self.n_dims]
        z_h = z_xh[:, :, self.n_dims:]
        assert_mean_zero_with_mask(z_x, node_mask)
        # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
        z_h = {'categorical': torch.zeros(0).to(z_h), 'integer': z_h}
        #  
        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss_ld, loss_dict = self.compute_loss(z_x, z_h, node_mask, edge_mask, context, t0_always=False) 
            # loss_ld *=  0.5
        else:
            # Less variance in the estimator, costs two forward passes.
            loss_ld, loss_dict = self.compute_loss(z_x, z_h, node_mask, edge_mask, context, t0_always=True)
            # loss_ld *=  0.5
        
        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_h_given_z0(
            torch.cat([h['categorical'], h['integer']], dim=2), node_mask) # add 0.1
        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        neg_log_pxh = loss_ld + loss_recon + neg_log_constants 

        return neg_log_pxh, loss_ld, loss_recon, neg_log_constants
    
    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        """
        Draw samples from the generative model.
        """
        z_x, z_h = super().sample(n_samples, n_nodes, node_mask, edge_mask, context, fix_noise)

        z_xh = torch.cat([z_x, z_h['categorical'], z_h['integer']], dim=2)
        assert_correctly_masked(z_xh, node_mask)
        x, h = self.vae.decode(z_xh, node_mask, edge_mask, context)

        return x, h
    
    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        chain_flat = super().sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames)

        # xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # chain[0] = xh  # Overwrite last frame with the resulting x and h.

        # chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        chain = chain_flat.view(keep_frames, n_samples, *chain_flat.size()[1:])
        chain_decoded = torch.zeros(
            size=(*chain.size()[:-1], self.vae.in_node_nf + self.vae.n_dims), device=chain.device)

        for i in range(keep_frames):
            z_xh = chain[i]
            assert_mean_zero_with_mask(z_xh[:, :, :self.n_dims], node_mask)

            x, h = self.vae.decode(z_xh, node_mask, edge_mask, context)
            xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
            chain_decoded[i] = xh
        
        chain_decoded_flat = chain_decoded.view(n_samples * keep_frames, *chain_decoded.size()[2:])

        return chain_decoded_flat

    def instantiate_first_stage(self, vae: EnHierarchicalVAE):
        if not self.trainable_ae:
            self.vae = vae.eval()
            self.vae.train = disabled_train
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            self.vae = vae.train()
            for param in self.vae.parameters():
                param.requires_grad = True


@register_model("geoldmori")
class GEOLDMori(BaseUnicoreModel):
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
            "--num_diffusion_timesteps",
            type=int,
            default=1000, #1000
        )
        parser.add_argument(
            "--augment_noise", 
            type=float, 
            default=0
        )
        parser.add_argument(
            "--include_charges", 
            type=str2bool,
            default=True,
        ),
        parser.add_argument(
            "--ode_regularization", 
            type=float, 
            default=1e-3,
        ),
        parser.add_argument(
            "--latent_dim", 
            type=int, 
            default=1,
        ),
        parser.add_argument(
            "--kl_weight", 
            type=float, 
            default=0.01,
        ),
        parser.add_argument(
            "--condition_time", 
            type=bool, 
            default=True,
        ),
        parser.add_argument(
            "--trainable_ae", 
            action="store_true",
            help="Train first stage AutoEncoder model")

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.indices = {k:v for k,v in self.dictionary.indices.items()}
        self.special_idx = [self.indices[i] for i in dictionary.specials]

        # self.tokens_list = list(set(self.dictionary.symbols) - set(self.dictionary.specials))
        # self.token_dict = {self.indices[token]: token for token in self.tokens_list if token in self.indices}
        # self.token_keys = list(self.token_dict.keys())

        # self.embed_tokens = nn.Embedding(
        #     len(dictionary), args.encoder_embed_dim, self.padding_idx
        # )
        self.latent_dim = args.latent_dim

        self.num_classes = len(HISTOGRAM[self.args.task_name]['atom_decoder'])
        self._num_updates = None


        
        if len(args.conditioning) > 0:
            print(f'Conditioning on {args.conditioning}')
            with open(os.path.join(args.data, args.task_name, "property_norms.pkl"), 'rb') as f:
                data = pickle.load(f)
                num_atoms = data['num_atoms']
                property_norms = {}
                condition_data = {}
                for condition in args.conditioning:
                    condition_data[condition] = data[condition].pop('data', None)  # retrieve 'data' if it exists
                    property_norms[condition] = data[condition]       
                context_node_nf = len(property_norms)
        else:
            context_node_nf = 0
            property_norms = None
            condition_data = None
            num_atoms = None

        self.property_norms = property_norms
        self.context_node_nf = context_node_nf

        first_stage_model, self.nodes_dist, self.prop_dist = self.__init__vae(args, condition_data, num_atoms)
        in_node_nf = self.latent_dim

        if self.prop_dist is not None:
            self.prop_dist.set_normalizer(property_norms)

        if args.condition_time:
            dynamics_in_node_nf = in_node_nf + 1 # 2 
        else:
            print('Warning: dynamics model is _not_ conditioned on time.')
            dynamics_in_node_nf = in_node_nf        


        net_dynamics = EGNN_dynamics_QM9(
            in_node_nf=dynamics_in_node_nf, 
            context_node_nf=self.context_node_nf, #2 0 
            n_dims=3, 
            hidden_nf=args.encoder_embed_dim, # 256
            act_fn=torch.nn.SiLU(), 
            n_layers=args.encoder_layers, # 9
            attention=True, 
            tanh=True, 
            mode='egnn_dynamics', 
            norm_constant=1,
            inv_sublayers=1, 
            sin_embedding=False,
            normalization_factor=1, 
            aggregation_method='sum', # 1 , sum
        )        
        self.vdm = EnLatentDiffusion(
            vae=first_stage_model,
            trainable_ae=args.trainable_ae,
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.num_diffusion_timesteps, #1000
            noise_schedule='polynomial_2', # 'polynomial_2'
            noise_precision=1e-5, #1e-5
            loss_type='l2', # l2
            norm_values=[1, 4, 10], # 1,4,10
            include_charges=self.args.include_charges, # False
            )

    def __init__vae(self, args, condition_data=None, num_atoms=None):

        in_node_nf = self.num_classes + int(self.args.include_charges)
        nodes_dist = DistributionNodes(HISTOGRAM[self.args.task_name]['n_nodes'])

        prop_dist = None
        if len(args.conditioning) > 0: 
            prop_dist = DistributionProperty(num_atoms, condition_data, args.conditioning)        
            
        encoder = EGNN_encoder_QM9(
            in_node_nf=in_node_nf,  # 6
            context_node_nf=self.context_node_nf, 
            out_node_nf=self.latent_dim,
            n_dims=3, 
            hidden_nf=args.encoder_embed_dim, # 512
            act_fn=torch.nn.SiLU(), 
            n_layers=1,
            attention=True, 
            tanh=True, 
            mode='egnn_dynamics', 
            norm_constant=1,
            inv_sublayers=1, 
            sin_embedding=False,
            normalization_factor=1, 
            aggregation_method='sum', # 1 , sum
            include_charges=args.include_charges # False
        )

        decoder = EGNN_decoder_QM9(
            in_node_nf=self.latent_dim, 
            context_node_nf=self.context_node_nf, 
            out_node_nf=in_node_nf, # 1 0 6
            n_dims=3, 
            hidden_nf=args.encoder_embed_dim, # 256
            act_fn=torch.nn.SiLU(), 
            n_layers=args.decoder_layers, #9
            attention=True, 
            tanh=True, 
            mode='egnn_dynamics', 
            norm_constant=1,
            inv_sublayers=1, 
            sin_embedding=False,
            normalization_factor=1, 
            aggregation_method='sum', # 1 , sum
            include_charges=args.include_charges # False
            )

        vae = EnHierarchicalVAE(
            encoder=encoder,
            decoder=decoder,
            in_node_nf=in_node_nf, # 6
            n_dims=3,
            latent_node_nf=self.latent_dim, # 1 
            kl_weight=args.kl_weight, #0.01
            norm_values=[1, 4, 10], # 1, 4, 10
            include_charges=args.include_charges # False
            )

        return vae, nodes_dist, prop_dist




    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def preprocess_input(self, src_coord, included_species, **kwargs):

        # atom type one hot and charges 
        charges = kwargs['charges'].to(src_coord.device, src_coord.dtype)
        included_species = torch.tensor(included_species)
        one_hot = charges.unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0).to(charges.device)
        one_hot = one_hot.to(src_coord.device)

        # atom mask 
        atom_mask = charges > 0

        # edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        # mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask.to(edge_mask.device)
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        charges = charges.unsqueeze(2)

        return src_coord, atom_mask, one_hot, edge_mask, charges

    def prepare_context(self, conditioning, positions, node_mask, property_norms, **kwargs):
        batch_size, n_nodes, _ = positions.size()
        node_mask = node_mask
        context_node_nf = 0
        context_list = []
        for key in conditioning:
            properties = kwargs[key]
            properties = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
            if len(properties.size()) == 1:
                # Global feature.
                assert properties.size() == (batch_size,)
                reshaped = properties.view(batch_size, 1, 1).repeat(1, n_nodes, 1)
                context_list.append(reshaped)
                context_node_nf += 1
            elif len(properties.size()) == 2 or len(properties.size()) == 3:
                # Node feature.
                assert properties.size()[:2] == (batch_size, n_nodes)

                context_key = properties

                # Inflate if necessary.
                if len(properties.size()) == 2:
                    context_key = context_key.unsqueeze(2)

                context_list.append(context_key)
                context_node_nf += context_key.size(2)
            else:
                raise ValueError('Invalid tensor size, more than 3 axes.')
        # Concatenate
        context = torch.cat(context_list, dim=2)
        # Mask disabled nodes!
        context = context * node_mask
        assert context.size(2) == context_node_nf
        return context

    def forward(
        self,
        # src_tokens,
        src_coord,
        time_step=None,
        **kwargs
    ):  

        self.device = src_coord.device
        all_species = HISTOGRAM[self.args.task_name]['all_species']
        x, node_mask, one_hot, edge_mask, charges = self.preprocess_input(src_coord, all_species,**kwargs)
        

        node_mask = node_mask.to(x.device, x.dtype).unsqueeze(2)
        edge_mask = edge_mask.to(x.device, x.dtype)
        one_hot = one_hot.to(x.device, x.dtype)
        charges = (charges if self.args.include_charges else torch.zeros(0)).to(x.device, x.dtype)

        check_mask_correct([x, one_hot, charges], node_mask)

        x = remove_mean_with_mask(x, node_mask) # Remove
        batch_size = src_coord.size(0)
        n_nodes = src_coord.size(1)

        if self.args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * self.args.augment_noise

        x = remove_mean_with_mask(x, node_mask).type_as(src_coord)


        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges.type_as(src_coord)}


        if len(self.args.conditioning) > 0:
            context = self.prepare_context(self.args.conditioning, x, node_mask, self.property_norms, **kwargs).to(x.device, x.dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        assert_correctly_masked(x, node_mask)

        nll, loss_ld, loss_recon, neg_log_constants = self.vdm(x, h, node_mask, edge_mask.view(batch_size, n_nodes * n_nodes), context)

        N = node_mask.squeeze(2).sum(1).long()
        log_pN = self.nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.

        loss = nll + self.args.ode_regularization * reg_term
        
        return {
            "loss": loss,
            "node_dist": self.nodes_dist,
            "prop_dist": self.prop_dist,
            "loss_ld": loss_ld.mean(),
            "loss_recon":loss_recon.mean(),
            "neg_log_constants":neg_log_constants.mean()
        }



    @torch.no_grad()
    def sample_chain(self, n_tries):
        device = self.vdm.vae.encoder.final_mlp[0].weight.device
        n_samples = 1
        if self.args.task_name == 'qm9':
            n_nodes = 19
        elif self.args.task_name == 'qm9_geoldm':
            n_nodes = 19
        elif self.args.task_name == 'oled_geoldm' or self.args.task_name == 'oled' or self.args.task_name == 'oled_geoldm1':
            n_nodes = 41
        else:
            raise ValueError()
        
        # TODO FIX: This conditioning just zeros.
        if self.context_node_nf > 0:
            context = self.prop_dist.sample(n_nodes).unsqueeze(1).unsqueeze(0)
            context = context.repeat(1, n_nodes, 1).to(device)
            #context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
        else:
            context = None

        node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
        edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)


        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = self.vdm.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100)
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()
            mol_stable = check_stabilityori(x_squeeze, atom_type, HISTOGRAM[self.args.task_name], task_name=self.args.task_name)[0]
            
            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(torch.argmax(one_hot, dim=2), num_classes=self.num_classes)
            charges = torch.round(chain[:, :, -1:]).long()
            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

        save_xyz_fileori(f'{self.args.save_dir}/outputs/{self.args.task_name}/sample_chain/chain/',
                      one_hot, charges, x, HISTOGRAM[self.args.task_name], 0, name='chain')

        return one_hot, charges, x

    @torch.no_grad()
    def sample(
        self, 
        prop_dist=None, 
        nodesxsample=torch.tensor([10]), 
        context=None,
        fix_noise=False,
    ):
        
        device = self.vdm.vae.encoder.final_mlp[0].weight.device

        if 'n_nodes' in HISTOGRAM[self.args.task_name].keys():
            max_n_nodes =  max(HISTOGRAM[self.args.task_name]['n_nodes']) # this is the maximum node_size in QM9 #29

        else: 
            max_n_nodes = self.args.max_atoms

        assert int(torch.max(nodesxsample)) <= max_n_nodes
        batch_size = len(nodesxsample) #100

        node_mask = torch.zeros(batch_size, max_n_nodes) # 100,29
        for i in range(batch_size):
            node_mask[i, 0:nodesxsample[i]] = 1

        # Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
        node_mask = node_mask.unsqueeze(2).to(device)

        # TODO FIX: This conditioning just zeros.
        if self.context_node_nf > 0:
            if context is None:
                context = prop_dist.sample_batch(nodesxsample)
            context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
        else:
            context = None

        x, h = self.vdm.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise)

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h['categorical'] # torch.Size([100, 29, 5])
        charges = h['integer'] #torch.Size([100, 29, 0])

        assert_correctly_masked(one_hot.float(), node_mask)
        if self.args.include_charges:
            assert_correctly_masked(charges.float(), node_mask)

        if hasattr(self.args, 'save_dir'):
            save_xyz_fileori(f'{self.args.save_dir}/outputs/{self.args.task_name}/sample_molecule/', one_hot, charges, x, HISTOGRAM[self.args.task_name], 0, name='molecule')

        return one_hot, charges, x, node_mask

    @torch.no_grad()
    def sample_sweep_conditional(self, prop_dist, n_nodes=19, n_frames=100, i=0):
        device = self.vdm.vae.encoder.final_mlp[0].weight.device
        nodesxsample = torch.tensor([n_nodes] * n_frames) # TODO: Change n_nodes for oled opv tasks

        context = []
        for key in prop_dist.distributions:
            min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
            mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
            min_val = (min_val - mean) / (mad)
            max_val = (max_val - mean) / (mad)
            context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
            context.append(context_row)
        context = torch.cat(context, dim=1).float().to(device)

        one_hot, charges, x, node_mask = self.sample(
                                                prop_dist=prop_dist,
                                                nodesxsample=nodesxsample, 
                                                context=context, 
                                                fix_noise=True
                                            )
        if hasattr(self.args, 'save_dir'):
            save_xyz_fileori(f'{self.args.save_dir}/outputs/{self.args.task_name}/conditional/', one_hot, charges, x, HISTOGRAM[self.args.task_name], i, name='conditional')  


        return one_hot, charges, x, node_mask

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


@register_model_architecture("geoldmori", "geoldmori")
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

@register_model_architecture("geoldmori", "geoldmori-qm9")
def base_architecture(args):
    # encoder
    args.encoder_layers = getattr(args, "encoder_layers", 9) #6 8
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256) # 512
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
    args.decoder_layers = getattr(args, "decoder_layers", 9) #12
    args.embed_dim = getattr(args, "embed_dim", 512)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 2048)
    args.attention_heads = getattr(args, "attention_heads", 64)
    args.rel_pos = getattr(args, "rel_pos", True)
    args.rel_pos_bins = getattr(args, "rel_pos_bins", 32)
    args.max_rel_pos = getattr(args, "max_rel_pos", 256)
    args.auto_regressive = getattr(args, "auto_regressive", True)

@register_model_architecture("geoldmori", "geoldmori-oled")
def base_architecture(args):
    # encoder
    args.encoder_layers = getattr(args, "encoder_layers", 4) #8 9
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
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
    args.decoder_layers = getattr(args, "decoder_layers", 4) # 9  4
    args.embed_dim = getattr(args, "embed_dim", 512)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 2048)
    args.attention_heads = getattr(args, "attention_heads", 64)
    args.rel_pos = getattr(args, "rel_pos", True)
    args.rel_pos_bins = getattr(args, "rel_pos_bins", 32)
    args.max_rel_pos = getattr(args, "max_rel_pos", 256)
    args.auto_regressive = getattr(args, "auto_regressive", True)


@register_model_architecture("geoldmori", "geoldmori-opv")
def base_architecture(args):
    # encoder
    args.encoder_layers = getattr(args, "encoder_layers", 1)
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
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.embed_dim = getattr(args, "embed_dim", 512)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 2048)
    args.attention_heads = getattr(args, "attention_heads", 64)
    args.rel_pos = getattr(args, "rel_pos", True)
    args.rel_pos_bins = getattr(args, "rel_pos_bins", 32)
    args.max_rel_pos = getattr(args, "max_rel_pos", 256)
    args.auto_regressive = getattr(args, "auto_regressive", True)