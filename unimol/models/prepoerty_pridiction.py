

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
    'classifier-qm9': UniMolQM9Model,
    'classifier-oled': UniMolOLEDModel,
    'classifier-opv': UniMolOPVModel,
}    

DECODER_REGISTER = {
    'TFM': TransformerDecoder,
    'GRU': GRUDecoder,
    'SE3': SE3CoordHead,
    'MLP': MLPDecoder,
}

# INFERENCE_REGISTER = {
#     'GREEDY': inference_greedy,
#     'BEAM':   inference_beam,
#     'TOPP':   inference_top_p,
#     'TOPK':   inference_top_k,
# }

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
    }
    
}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    
edges_dic = {}

def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(GCL_basic, self).__init__()


    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat



class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0, act_fn=nn.ReLU(), bias=True, attention=False, t_eq=False, recurrent=True):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq=t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())


        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias))

        #if recurrent:
            #self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
            #out = self.gru(out, h)
        return out


class GCL_rf(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, nf=64, edge_attr_nf=0, reg=0, act_fn=nn.LeakyReLU(0.2), clamp=False):
        super(GCL_rf, self).__init__()

        self.clamp = clamp
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(edge_attr_nf + 1, nf),
                                 act_fn,
                                 layer)
        self.reg = reg

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)
        return m_ij

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x*self.reg
        return x_out


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr


class E_GCL_vel(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """


    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention, norm_diff=norm_diff, tanh=tanh)
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)


        coord += self.coord_mlp_vel(h) * vel
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr




class GCL_rf_vel(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """
    def __init__(self,  nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0):
        super(GCL_rf_vel, self).__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, nf),
            act_fn,
            nn.Linear(nf, 1))

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        #layer.weight.uniform_(-0.1, 0.1)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, nf),
                                 act_fn,
                                 layer,
                                 nn.Tanh()) #we had to add the tanh to keep this method stable

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x += vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)



class Naive(nn.Module):
    def __init__(self, device):
        super(Naive, self).__init__()
        self.device = device
        self.linear = nn.Linear(1, 1)
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        node_mask = node_mask.view(-1, n_nodes)
        bs, n_nodes = node_mask.size()
        x = torch.zeros(bs, 1).to(self.device)
        return self.linear(x).squeeze(1)


class NumNodes(nn.Module):
    def __init__(self, device, nf=128):
        super(NumNodes, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(1, nf)
        self.linear2 = nn.Linear(nf, 1)
        self.act_fn = nn.SiLU()
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        reshaped_mask = node_mask.view(-1, n_nodes)
        nodesxmol = torch.sum(reshaped_mask, dim=1).unsqueeze(1)/29
        x = self.act_fn(self.linear1(nodesxmol))
        return self.linear2(x).squeeze(1)
    




@register_model("classifier")
class Classifier(BaseUnicoreModel):
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
        ),
        parser.add_argument(
            "--encoder",
            type=str,
            metavar='D',
            choices=ENCODER_REGISTER.keys(),
        ),
        parser.add_argument(
            "--decoder",
            type=str,
            metavar="D",
            choices=DECODER_REGISTER.keys(),
        ),
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

        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )

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
        
        self.mean, self.mad = property_norms[args.conditioning[0]]['mean'], property_norms[args.conditioning[0]]['mad']
        self.property_norms = property_norms
        self.context_node_nf = context_node_nf

        self.model = self.get_model(args)

    def get_model(self, args):
        if args.classifier[0] == 'egnn':
            args.device = self.embed_tokens.weight.device
            model = EGNN(in_node_nf=5, in_edge_nf=0, hidden_nf=args.encoder_embed_dim, device=args.device, n_layers=7,
                        coords_weight=1.0,
                        attention=1, node_attr=0)
        elif args.classifier[0] == 'naive':
            model = Naive(device=args.device)
        elif args.classifier[0] == 'numnodes':
            model = NumNodes(device=args.device)
        else:
            raise Exception("Wrong model name %s" % args.classifier[0])

        return model

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

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

        batch_size = src_coord.size(0)
        n_nodes = src_coord.size(1)
        
        atom_positions = x.view(batch_size * n_nodes, -1).to(self.device, torch.float32)
        atom_mask = node_mask.view(batch_size * n_nodes, -1).to(self.device, torch.float32)
        edge_mask = edge_mask.to(self.device, torch.float32)
        nodes = one_hot.to(self.device, torch.float32)
        nodes = nodes.view(batch_size * n_nodes, -1)


        edges = get_adj_matrix(n_nodes, batch_size, self.device)


        label = []
        for condition in self.args.conditioning:
            label.append(kwargs[condition].to(self.device, torch.float32))
        label = torch.cat(label, dim=0)

        pred = self.model(
            h0=nodes, 
            x=atom_positions, 
            edges=edges, 
            edge_attr=None, 
            node_mask=atom_mask, 
            edge_mask=edge_mask,
            n_nodes=n_nodes
        )

        loss_l1 = nn.L1Loss()

        loss = loss_l1(pred, (label - self.mean) / self.mad)
        ## TODO, INFER NEED TO self.mad * pred + self.mean

        # if self.training:
        #     loss = loss_l1(pred, (label - self.mean) / self.mad)
        # else:
        #     loss = loss_l1(self.mad * pred + self.mean, label)

        return {
            "loss": loss,
            # "node_dist": self.nodes_dist,
            # "prop_dist": self.prop_dist,
            # "loss_ld": loss_ld.mean(),
            # "loss_recon":loss_recon.mean(),
            # "neg_log_constants":neg_log_constants.mean()
        }


@register_model_architecture("classifier", "classifier")
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

@register_model_architecture("classifier", "classifier-qm9")
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

@register_model_architecture("classifier", "classifier-oled")
def base_architecture(args):
    # encoder
    args.encoder_layers = getattr(args, "encoder_layers", 9) #8 9
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
    args.decoder_layers = getattr(args, "decoder_layers", 9) # 9  4
    args.embed_dim = getattr(args, "embed_dim", 512)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 2048)
    args.attention_heads = getattr(args, "attention_heads", 64)
    args.rel_pos = getattr(args, "rel_pos", True)
    args.rel_pos_bins = getattr(args, "rel_pos_bins", 32)
    args.max_rel_pos = getattr(args, "max_rel_pos", 256)
    args.auto_regressive = getattr(args, "auto_regressive", True)


@register_model_architecture("classifier", "classifier-opv")
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