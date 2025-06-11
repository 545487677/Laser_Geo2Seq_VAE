import numpy as np
from tqdm import tqdm
import os
import time
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import joblib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_linear_schedule_with_warmup
import re
from rdkit.Chem import MolFromSmiles, MACCSkeys, AllChem
import logging
import argparse
import pathlib
import logging



# wget https://github.com/545487677/pretrain_model/raw/main/pretrain_bert.pth
PRE_TRAIN_WEIGHT_PATH = os.path.join(
    './',
    'pretrain_bert.pth'
)


ATOM_TOKEN_LIST = ['c', 'C', 'O', 'N', 'n', '[C@H]', 'F', '[C@@H]', 'S', 'Cl', '[nH]', 's', 'o', '[C@]',
                           '[C@@]', '[O-]', '[N+]', 'Br', 'P', '[n+]', 'I', '[S+]',  '[N-]', '[Si]', 'B', '[Se]', '[other_atom]']

ALL_TOKEN_LIST = ['[PAD]', '[GLO]', 'c', 'C', '(', ')', 'O', '1', '2', '=', 'N', '3', 'n', '4', '[C@H]', 'F', '[C@@H]', '-', 'S', '/', 'Cl',
                           '[nH]', 's', 'o', '5', '#', '[C@]', '[C@@]', '\\', '[O-]', '[N+]', 'Br', '6', 'P', '[n+]', '7', 'I', '[S+]', '8', '[N-]', '[Si]', 'B', '9', '[2H]', '[Se]', '[other_atom]', '[other_token]']

class SmileTokenizer(object):
    def __init__(self, **params):
        self._init_features(**params)

    def _init_features(self, **params):
       self.max_len = params.get('max_len', 200)

    def single_process(self, smile):
        smile2idx = {}
        for i,j in enumerate(ALL_TOKEN_LIST):
            smile2idx[j] = i
        tokens,tokens_idx,smile2idx = smi_tokenizer(smile,self.max_len, smile2idx)
        tokens_idx = [smile2idx[x] for x in tokens]
        return tokens_idx

    def transform(self, smiles_list):
        fps = []
        for smile in tqdm(smiles_list):
            fps.append(self.single_process(smile))
        return fps

def smi_tokenizer(smi, max_len, smile2idx):
    tokens_idx = []
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    tokens = tokens[:max_len]
    padding_list = ['[PAD]' for x in range(max_len-len(tokens))]
    tokens = ['[GLO]'] + tokens + padding_list
    for i,token in enumerate(tokens):
        if token in ATOM_TOKEN_LIST:
                tokens_idx.append(smile2idx[token])
        else:
            if token in ALL_TOKEN_LIST:
                tokens_idx.append(smile2idx[token])
            elif '[' in list(token):
                tokens[i] = '[other_atom]'
                tokens_idx.append(smile2idx['[other_atom]'])
            else:
                tokens[i] = '[other_token]'                
                tokens_idx.append(smile2idx['[other_token]'])        
    return tokens,tokens_idx,smile2idx


def get_attn_pad_mask(seq_q):
    batch_size, seq_len = seq_q.size()
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)

class Embedding(nn.Module):
    def __init__(self, vocab_size=None, d_model=None, maxlen=None):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x=None):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).to(x)
        embedding = self.tok_embed(x.long()) + self.pos_embed(pos.long())
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k=None):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e4) #这里设置为1e4 当设置为1e9 会报错value cannot be converted to type at::Half without overflow 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=None, d_k=None, d_v=None, n_heads=None):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        super(MultiHeadAttention, self).__init__()
        self.linear = nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.layernorm = nn.LayerNorm(self.d_model)
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads)

    def forward(self, Q=None, K=None, V=None, attn_mask=None):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layernorm(output + residual)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=None, d_ff=None):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs=None):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs

class Bert(nn.Module):
    def __init__(self, output_dim=1, pretrain_path=PRE_TRAIN_WEIGHT_PATH, **kwargs):
        super(Bert, self).__init__()
        self.args = base_architecture()
        self.output_dim = output_dim
        self.d_model = self.args.d_model
        self.embedding = Embedding(self.args.vocab_size, self.d_model, self.args.max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.args.d_k, self.args.d_v, self.args.n_heads, self.args.d_ff) for _ in range(self.args.n_layers)])
        self.fc = nn.Sequential(
                        nn.Dropout(self.args.dropout),
                        nn.Linear(self.d_model, self.d_model),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.d_model))
        self.classifier_global = nn.Linear(self.d_model, output_dim)
        self.classifier_atom = nn.Linear(self.d_model, self.args.atom_label_dim)
        self.load_pretrained_weights(path=pretrain_path)

    def load_pretrained_weights(self, path):
        if path is not None:
            logging.info("Loading pretrained weights from {}".format(path))
            pretrained_model = torch.load(path, map_location=lambda storage, loc: storage)
            model_dict = self.state_dict()
            unique_list = ['classifier_atom.weight', 'classifier_global.bias', 'classifier_global.weight', 'fc_global.0.bias', 'fc_atom.0.bias', 'fc_atom.0.weight', 'fc_global.0.weight', 'classifier_atom.bias']
            pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k not in unique_list}
            model_dict.update(pretrained_dict)
            self.load_state_dict(pretrained_dict, strict=False)

    def forward(self, net_input=None):
        output = self.embedding(net_input)
        attn_mask = get_attn_pad_mask(net_input)
        for layer in self.layers:
            output = layer(output, attn_mask)
        h_mol = output[:, 0]
        h_embed = self.fc(h_mol)
        logits = self.classifier_global(h_embed)
        return logits

    def batch_collate_fn(self, samples):
        x = torch.tensor([s[0] for s in samples]).float()
        label = torch.tensor([s[1] for s in samples])
        return x, label

def base_architecture():
    args = argparse.ArgumentParser()
    args.vocab_size = 47
    args.dropout = 0.1
    args.n_layers = 6
    args.d_model = 768
    args.d_k = 64
    args.d_v = 64
    args.d_ff = 3072
    args.n_heads = 12
    args.atom_label_dim = 15
    args.max_seq_len = 201
    return args

NNDATASET_REGISTER = {
    # '2D_graph': GraphDataset,
    # '2D_graph_1': GraphDataset_hignn,
}

class TorchDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

def NNDataset(data, label=None, feature_name=None, task=None):
    if feature_name in  NNDATASET_REGISTER:
        dataset = NNDATASET_REGISTER[feature_name](data, label, task)
    else:
        dataset = TorchDataset(data, label)
    return dataset

def NNDataset(data, label=None, feature_name=None, task=None):
    if feature_name in  NNDATASET_REGISTER:
        dataset = NNDATASET_REGISTER[feature_name](data, label, task)
    else:
        dataset = TorchDataset(data, label)
    return dataset

from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    roc_auc_score,
    accuracy_score,
    log_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    average_precision_score,
    recall_score,
    cohen_kappa_score,
)
from scipy.stats import (
    spearmanr,
    pearsonr
)
import logging
import copy

def cal_nan_metric(y_true, y_pred, nan_value=None, metric_func=None):
    if y_true.shape != y_pred.shape:
        raise ValueError('y_ture and y_pred must have same shape')

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()

    mask = ~np.isnan(y_true)
    if nan_value is not None:
        mask = mask & (y_true != nan_value)

    sz = y_true.shape[1]
    result= []
    for i in range(sz):
        _mask = mask[:, i]
        if not (~_mask).all():
            result.append(metric_func(y_true[:,i][_mask], y_pred[:,i][_mask]))
    return np.mean(result)

def multi_acc(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred_idx = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred_idx)


def log_loss_with_label(y_true, y_pred, labels=None):
    if labels is None:
        return log_loss(y_true, y_pred)
    else:
        return log_loss(y_true, y_pred, labels=labels)

## metric_func, is_increase, value_type
METRICS_REGISTER = {
    'regression': {
        "mae" : [mean_absolute_error, False, 'float'],
        "pearsonr": [lambda y_true, y_pred: pearsonr(y_true, y_pred)[0], True, 'float'],
        "spearmanr": [lambda y_ture, y_pred: spearmanr(y_ture, y_pred)[0], True, 'float'],
        "mse" : [mean_squared_error, False, 'float'],
        "r2" : [r2_score, True, 'float'],
    },
    'classification': {
        "auroc" : [roc_auc_score, True, 'float'],
        "auc": [roc_auc_score, True, 'float'],
        "auprc": [average_precision_score, True, 'float'],
        "log_loss" : [log_loss, False, 'float'],
        "acc": [accuracy_score, True, 'int'],
        "f1_score": [f1_score, True, 'int'],
        "mcc": [matthews_corrcoef, True, 'int'],
        "precision": [precision_score, True, 'int'],
        "recall": [recall_score,  True, 'int'],
        "cohen_kappa": [cohen_kappa_score, True, 'int'],
    },
    'multiclass':{
        "log_loss" : [log_loss_with_label, False, 'float'],
        "acc": [multi_acc, True, 'int'],
    },
    'multilabel_classification': {
        "auroc" : [roc_auc_score, True, 'float'],
        "auc": [roc_auc_score, True, 'float'],
        "auprc": [average_precision_score, True, 'float'],
        "log_loss" : [log_loss_with_label, False, 'float'],
        "acc": [accuracy_score, True, 'int'],
        "mcc": [matthews_corrcoef, True, 'int'],
    },
    'multilabel_regression':{
        "mae" : [mean_absolute_error, False, 'float'],
        "mse": [mean_squared_error, False, 'float'],
        "r2": [r2_score, True, 'float'],
    }
}

DEFAULT_METRICS = {
    'regression': ['mse', 'mae', 'r2', 'spearmanr', 'pearsonr'],
    'classification': ['log_loss', 'auc', 'f1_score', 'mcc', 'acc', 'precision', 'recall'],
    'multiclass': ['log_loss', 'acc'],
    "multilabel_classification": ['log_loss', 'auc', 'auprc'],
    "multilabel_regression": ['mse', 'mae', 'r2'],
}
class Metrics(object):
    def __init__(self, task = 'regression', metrics_str = 'mse', **params):
        self.task = task
        self.threshold = np.arange(0, 1., 0.1)
        self.metric_dict = self._init_metrics(self.task, metrics_str, **params)
        self.METRICS_REGISTER = METRICS_REGISTER[task]

    def _init_metrics(self, task, metrics_str, **params):
        if task not in METRICS_REGISTER:
            raise ValueError('Unknown task: {}'.format(self.task))
        if not isinstance(metrics_str, str) or metrics_str == '' or metrics_str == 'none':
            metric_dict = {key:METRICS_REGISTER[task][key] for key in DEFAULT_METRICS[task]}
        else:
            for key in metrics_str.split(','):
                if key not in METRICS_REGISTER[task]:
                    raise ValueError('Unknown metric: {}'.format(key))

            priority_metric_list = metrics_str.split(',')
            metric_list = priority_metric_list + [key for key in METRICS_REGISTER[task] if key not in priority_metric_list]
            metric_dict = {key:METRICS_REGISTER[task][key] for key in metric_list}
        
        return metric_dict
    
    def cal_classification_metric(self, label, predict, nan_value = -1.0, threshold = None):
        r"""
            :param label:int
            :param predict:float
        """
        res_dict = {}
        for metric_type, metric_value in self.metric_dict.items():
            metric, _, value_type =  metric_value
            nan_metric = lambda label, predict : cal_nan_metric(label, predict, nan_value, metric)
            if value_type == 'float':
                res_dict[metric_type] = nan_metric(label.astype(int), predict.astype(np.float))
            elif value_type == 'int':
                thre = 0.5 if threshold is None else threshold
                res_dict[metric_type] = nan_metric(label.astype(int), (predict > thre).astype(int))

        ## TO DO : add more metrics by grid search threshold

        return res_dict

    def cal_reg_metric(self, label, predict, nan_value = -1.0):
        r"""
            :param label:int
            :param predict:float
        """
        res_dict = {}
        for metric_type, metric_value in self.metric_dict.items():
            metric, _, _ =  metric_value
            nan_metric = lambda label, predict : cal_nan_metric(label, predict, nan_value, metric)
            res_dict[metric_type] = nan_metric(label, predict)

        return res_dict
    
    def cal_multiclass_metric(self, label, predict, nan_value = -1.0, label_cnt=-1):
        r"""
            :param label:int
            :param predict:float
        """
        res_dict = {}
        for metric_type, metric_value in self.metric_dict.items():
            metric, _, _ =  metric_value
            if metric_type == 'log_loss' and label_cnt is not None:
                labels = list(range(label_cnt))
                res_dict[metric_type] = metric(label, predict, labels)
            else:
                res_dict[metric_type] = metric(label, predict)

        return res_dict
       

    def cal_metric(self, label, predict, nan_value = -1.0, threshold = 0.5, label_cnt=None):
        if self.task in ['regression', 'multilabel_regression']:
            return self.cal_reg_metric(label, predict, nan_value)
        elif self.task in ['classification', 'multilabel_classification']:
            return self.cal_classification_metric(label, predict, nan_value)
        elif self.task in ['multiclass']:
            return self.cal_multiclass_metric(label, predict, nan_value, label_cnt)
        else:
            raise ValueError("We will add more tasks soon")

    def _early_stop_choice(self, wait, min_score, metric_score, max_score, model, dump_dir, fold, patience, epoch):
        score = list(metric_score.values())[0]
        judge_metric = list(metric_score.keys())[0]
        is_increase = METRICS_REGISTER[self.task][judge_metric][1]
        if is_increase:
            is_early_stop, max_score, wait = self._judge_early_stop_increase(wait, score, max_score, model, dump_dir, fold, patience, epoch)
        else:
            is_early_stop, min_score, wait = self._judge_early_stop_decrease(wait, score, min_score, model, dump_dir, fold, patience, epoch)
        return is_early_stop, min_score, wait, max_score

    def _judge_early_stop_decrease(self, wait, score, min_score, model, dump_dir, fold, patience, epoch):
        is_early_stop = False
        if score <= min_score :
            min_score = score
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model.pth'))
        elif score >= min_score:
            wait += 1
            if wait == patience:
                logging.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_score, wait

    def _judge_early_stop_increase(self, wait, score, max_score, model, dump_dir, fold, patience, epoch):
        is_early_stop = False
        if score >= max_score :
            max_score = score
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model.pth'))
        elif score <= max_score:
            wait += 1
            if wait == patience:
                logging.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, max_score, wait


    def calculate_single_classification_threshold(self, target, pred, metrics_key=None, step=20):
        data = copy.deepcopy(pred)
        range_min = np.min(data).item()
        range_max = np.max(data).item()

        for metric_type, metric_value in self.metric_dict.items():
            metric, is_increase, value_type =  metric_value
            if value_type == 'int':
                metrics_key = metric_value
                break
        # default threshold metrics
        if metrics_key is None:
            metrics_key = METRICS_REGISTER['classification']['f1_score']
        logging.info("metrics for threshold: {0}".format(metrics_key[0].__name__))
        metrics = metrics_key[0]
        if metrics_key[1]:
            # increase metric
            best_metric = float('-inf')
            best_threshold = 0.5
            for threshold in np.linspace(range_min, range_max, step):
                pred_label = np.zeros_like(pred)
                pred_label[pred > threshold] = 1
                # print ("threshold: ", threshold, metric(target, pred_label))
                if metric(target, pred_label) > best_metric:
                    best_metric = metric(target, pred_label)
                    best_threshold = threshold
            logging.info("best threshold: {0}, metrics: {1}".format(best_threshold, best_metric))
        else:
            # increase metric
            best_metric = float('inf')
            best_threshold = 0.5
            for threshold in np.linspace(range_min, range_max, step):
                pred_label = np.zeros_like(pred)
                pred_label[pred > threshold] = 1
                if metric(target, pred_label) < best_metric:
                    best_metric = metric(target, pred_label)
                    best_threshold = threshold
            logging.info("best threshold: {0}, metrics: {1}".format(best_threshold, best_metric))

        return best_threshold

    def calculate_classification_threshold(self, target, pred):
        threshold = np.zeros(target.shape[1])
        for idx in range(target.shape[1]):
            threshold[idx] = self.calculate_single_classification_threshold(target[:, idx].reshape(-1,1), \
                pred[:, idx].reshape(-1, 1), metrics_key=None, step=20)
        return threshold

NNDATALOADER_REGISTER = {
    # '2D_graph': GraphDataLoader,
    # '2D_graph_1': GraphDataLoader,
}


DECORATE_REGISTER = [
    '2D_graph',
    '2D_graph_1'
]

ACTIVATION_FN = {
    ### predict prob shape should be (N, K), especially for binary classification, K equals to 1.
    'classification': lambda x: F.softmax(x, dim=-1)[:, 1:],
    'multiclass': lambda x: F.softmax(x, dim=-1),   ### softmax is used for multiclass classification
    'regression': lambda x: x,
    'multilabel_classification': lambda x: F.sigmoid(x),   ### sigmoid is used for multilabel classification
    'multilabel_regression': lambda x: x,   #### no activation function is used for multilabel regression
}


class Trainer(object):
    def __init__(self, task='regression', metrics_str= 'mse', out_dir=None, epoch=10, bsz=512, **params):
        self.task = task
        self.metrics_str = metrics_str.lower()
        self.metrics = Metrics(self.task, self.metrics_str, **params)
        self.out_dir = out_dir
        self._init_trainer(epoch, bsz)
    
    def _init_trainer(self, epoch, bsz):
        ### init common params ###
        self.seed = 42
        self.set_seed(self.seed)
        ### init NN trainer params ###
        self.learning_rate = 1e-4
        self.batch_size = bsz
        self.max_epochs = epoch #50
        self.warmup_ratio = 0.1
        self.patience = 10
        self.max_norm = 1.0
        self.cuda = True
        self.amp = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' and self.amp==True else None

    def decorate_batch(self, batch, feature_name):
        if feature_name in DECORATE_REGISTER:
            net_input, net_target = self.decorate_graph_batch(batch)
        else:
            net_input, net_target = self.decorate_torch_batch(batch)
        return net_input, net_target

    def decorate_graph_batch(self, batch):
        net_input, net_target = {'net_input':batch.to(self.device)}, batch.y.to(self.device)
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def decorate_torch_batch(self, batch):
        """function used to decorate batch data
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = {k: v.to(self.device) for k, v in net_input.items()}, net_target.to(self.device)
        else:
            net_input, net_target = {'net_input':net_input.to(self.device)}, net_target.to(self.device)
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target
    
    def fit_predict(self, model, train_dataset, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, feature_name):
        model = model.to(self.device)
        train_dataloader = NNDataLoader(
                                        feature_name=feature_name,
                                        dataset=train_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        collate_fn=model.batch_collate_fn,
                                        drop_last=True,
                                        )
                                        # remove last batch, bs=1 can not work on batchnorm1d
        min_val_loss = float("inf")
        max_score = float("-inf")
        wait = 0
        ### init optimizer ###
        num_training_steps = len(train_dataloader) * self.max_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        optimizer = Adam(model.parameters(), lr=self.learning_rate, eps=1e-6)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        for epoch in range(self.max_epochs):
            model = model.train()
            # Progress Bar
            start_time = time.time()
            batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5) 
            trn_loss = []
            for i, batch in enumerate(train_dataloader):
                net_input, net_target = self.decorate_batch(batch, feature_name)
                optimizer.zero_grad() # Zero gradients
                if self.scaler and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(**net_input)
                        loss = loss_func(outputs, net_target)
                else:
                    with torch.set_grad_enabled(True):
                        outputs = model(**net_input)
                        loss = loss_func(outputs, net_target)
                trn_loss.append(float(loss.data))
                # tqdm lets you add some details so you can monitor training as you train.
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                    lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                if self.scaler and self.device.type == 'cuda':
                    self.scaler.scale(loss).backward() # This is a replacement for loss.backward()
                    self.scaler.unscale_(optimizer) # unscale the gradients of optimizer's assigned params in-place
                    clip_grad_norm_(model.parameters(), self.max_norm)  # Clip the norm of the gradients to max_norm.
                    self.scaler.step(optimizer) # This is a replacement for optimizer.step()
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.max_norm)
                    optimizer.step()
                scheduler.step()
                batch_bar.update()
                                
            batch_bar.close()
            total_trn_loss = np.mean(trn_loss)

            y_preds, val_loss, metric_score = self.predict(model, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, epoch, load_model=False, feature_name=feature_name)
            end_time = time.time()
            total_val_loss = np.mean(val_loss)
            _score = list(metric_score.values())[0]
            _metric = list(metric_score.keys())[0]
            message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}, lr: {:.6f}, ' \
                '{:.1f}s'.format(epoch+1, self.max_epochs,
                                total_trn_loss, total_val_loss, 
                                _metric, _score,
                                optimizer.param_groups[0]['lr'],
                                (end_time - start_time))
            logging.info(message)
            is_early_stop, min_val_loss, wait, max_score = self._early_stop_choice(wait, total_val_loss, min_val_loss, metric_score, max_score, model, dump_dir, fold, self.patience, epoch)
            if is_early_stop:
                break

        y_preds, _, _ = self.predict(model, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, epoch, load_model=True, feature_name=feature_name)
        return y_preds


    def _early_stop_choice(self, wait, loss, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch):
        ### hpyerparameter need to tune if you want to use early stop, currently find use loss is suitable in benchmark test. ###
        if not isinstance(self.metrics_str, str) or self.metrics_str in ['loss', 'none', '']:
            ## loss 作为早停 直接用trainer里面的早停函数
            is_early_stop, min_val_loss, wait = self._judge_early_stop_loss(wait, loss, min_loss, model, dump_dir, fold, patience, epoch)
        else:
            ## 到metric进行判断
            is_early_stop, min_val_loss, wait, max_score = self.metrics._early_stop_choice(wait, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch)
        return is_early_stop,min_val_loss,wait, max_score

    def _judge_early_stop_loss(self, wait, loss, min_loss, model, dump_dir, fold, patience, epoch):
        is_early_stop = False
        if loss <= min_loss :
            min_loss = loss
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model.pth'))
        elif loss >= min_loss:
            wait += 1
            if wait == self.patience:
                logging.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_loss, wait
        

    def predict(self, model, dataset, loss_func, activation_fn, dump_dir, fold, target_scaler=None, epoch=1, load_model=False, feature_name=None):
        model = model.to(self.device)
        if load_model == True:
            load_model_path = os.path.join(dump_dir, f'model.pth')
            model_dict = torch.load(load_model_path, map_location=self.device)["model_state_dict"]
            model.load_state_dict(model_dict)
            logging.info("load model success!")
        dataloader = NNDataLoader(
                                feature_name=feature_name,
                                dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                collate_fn=model.batch_collate_fn,
                                )
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='val', ncols=5)
        val_loss = []
        y_preds = []
        y_truths = []
        for i, batch in enumerate(dataloader):
            net_input, net_target = self.decorate_batch(batch, feature_name)
            # Get model outputs
            with torch.no_grad():
                outputs = model(**net_input)
                if not load_model:
                    loss = loss_func(outputs, net_target)
                    val_loss.append(float(loss.data))
            y_preds.append(activation_fn(outputs).cpu().numpy())
            y_truths.append(net_target.detach().cpu().numpy())
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1))))

            batch_bar.update()
        y_preds = np.concatenate(y_preds)
        y_truths = np.concatenate(y_truths)

        try:
            label_cnt = model.output_dim
        except:
            label_cnt = None

        if target_scaler is not None:
            inverse_y_preds = target_scaler.inverse_transform(y_preds)
            inverse_y_truths = target_scaler.inverse_transform(y_truths)
            metric_score = self.metrics.cal_metric(inverse_y_truths, inverse_y_preds, label_cnt=label_cnt) if not load_model else None
        else:
            metric_score = self.metrics.cal_metric(y_truths, y_preds, label_cnt=label_cnt) if not load_model else None
        batch_bar.close()
        return y_preds, val_loss, metric_score

    def set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


def NNDataLoader(feature_name=None, dataset=None, batch_size=None, shuffle=False, collate_fn=None, drop_last=False):

    dataloader_func = NNDATALOADER_REGISTER.get(feature_name, TorchDataLoader)
    dataloader = dataloader_func(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    collate_fn=collate_fn,
                                    drop_last=drop_last)
    return dataloader

## python final_bert_eval.py full
## python final_bert_eval.py sample
if __name__ == '__main__':
    mode = sys.argv[1]
    data = sys.argv[2]
    # mode = 'full'
    # data = 'opv'
    print(data)
    print(mode)
    if mode == 'full':
        ## full
        
        train_smi = np.load(f'../../example_data/{data}/train_canonical_smiles.npy', allow_pickle=True).tolist()
        train_con = np.load(f'../../example_data/{data}/train_homo_condition.npy', allow_pickle=True)
        index = [idx for idx, smile in enumerate(train_smi) if smile is not None]
        train_smi = np.array([train_smi[i] for i in index])
        train_con = train_con[index]
        print(train_smi.shape, train_con.shape)#(1805633,)  ## opv (7803655,) (7803655,) (7803782,) (7803782,)
        assert train_smi.shape == train_con.shape
        if not os.path.exists(os.path.join('./', f'{data}_full_data_model.pth')):
            model = Bert(output_dim=1).cuda()
            # model.load_state_dict(torch.load('/data/users/guojianz/FuncMG/baseline/FunctionalMoleculeGeneration/unimol/utils/test.pth'), strict=False)
        else:
            model = Bert(output_dim=1).cuda()
            model.load_state_dict(torch.load(f'{data}_full_data_model.pth'), strict=False)

        ## train 
        if not os.path.exists(os.path.join('./', f'{data}_1d_feature.npy')):
            print('tokenizing......')
            feature = SmileTokenizer().transform(train_smi.tolist())
            print(len(feature))
            np.save(f'./{data}_1d_feature.npy', feature)
        else:
            feature = np.load(f'./{data}_1d_feature.npy', allow_pickle=True)
            del train_smi
        X_train, X_valid, y_train, y_valid = train_test_split(np.array(feature), train_con[:, None], test_size=0.2, random_state=42)
        traindataset = NNDataset(X_train, y_train)
        validdataset = NNDataset(X_valid, y_valid)
        trainer = Trainer(task='regression', metrics_str= 'r2', out_dir='./', epoch=10, bsz=256)
        loss_func = nn.MSELoss()
        activation_fn = ACTIVATION_FN['regression']
        _y_pred = trainer.fit_predict(model, traindataset, validdataset, loss_func, activation_fn, './', 0, None, None)
        print('mse: ', mean_squared_error(y_valid, _y_pred))
        print('r2: ', r2_score(y_valid, _y_pred))
        print('mae: ', mean_absolute_error(y_valid, _y_pred))
        torch.save(model.state_dict(), f'{data}_full_data_model.pth')
        ##  如果想直接推理可以注释掉上面


        # print("infer")
        model.load_state_dict(torch.load(f'{data}_full_data_model.pth'), strict=False)
        valid_smi = np.load('/vepfs/fs_users/guojianz/FuncMG/vdgen_job/finetune_tmp1/generate_results_name/smi_lists.npy', allow_pickle=True)
        condition_gt = np.load('/vepfs/fs_users/guojianz/FuncMG/vdgen_job/finetune_tmp1/generate_results_name/conditions.npy', allow_pickle=True)[:, None]
        print(valid_smi.shape, condition_gt.shape)
        feature = SmileTokenizer().transform(valid_smi.tolist())
        with torch.no_grad():
            output = model(torch.tensor(np.array(feature)).cuda())
            print(mean_absolute_error(condition_gt, output.detach().cpu()))
            print(r2_score(condition_gt, output.detach().cpu()))
        # os.system('rm -rf model.pth')

    elif mode == 'sample':

        ## sample 50000
        train_smi = np.load('/data/users/guojianz/FuncMG/baseline/FunctionalMoleculeGeneration/sample_train_smile.npy', allow_pickle=True)
        train_con = np.load('/data/users/guojianz/FuncMG/baseline/FunctionalMoleculeGeneration/sample_train_condition.npy', allow_pickle=True)
        print(train_smi.shape, train_con.shape)
        feature = SmileTokenizer().transform(train_smi.tolist())
        model = Bert(output_dim=1).cuda()
        # model.load_state_dict(torch.load('/data/users/guojianz/FuncMG/baseline/FunctionalMoleculeGeneration/unimol/utils/test.pth'), strict=False)
        X_train, X_valid, y_train, y_valid = train_test_split(np.array(feature), train_con[:, None], test_size=0.2, random_state=42)
        traindataset = NNDataset(X_train, y_train)
        validdataset = NNDataset(X_valid, y_valid)
        trainer = Trainer(task='regression', metrics_str= 'r2', out_dir='./', epoch=1, bsz=256)
        loss_func = nn.MSELoss()
        activation_fn = ACTIVATION_FN['regression']
        _y_pred = trainer.fit_predict(model, traindataset, validdataset, loss_func, activation_fn, './', 0, None, None)
        print('mse: ', mean_squared_error(y_valid, _y_pred))
        print('r2: ', r2_score(y_valid, _y_pred))
        print("infer")
        torch.save(model.state_dict(), 'test1.pth')
        model.load_state_dict(torch.load('./test1.pth'), strict=False)
        valid_smi = np.load('../../infer_res/valid_smi.npy', allow_pickle=True)
        condition_gt = np.load('../../infer_res/valid_smi.npy', allow_pickle=True)[:, None]
        print(valid_smi.shape, condition_gt.shape)
        feature = SmileTokenizer().transform(valid_smi.tolist())
        with torch.no_grad():
            output = model(torch.tensor(np.array(feature)).cuda())
            print(mean_squared_error(condition_gt, output.detach().cpu()))
            print(r2_score(condition_gt, output.detach().cpu()))
        os.system('rm -rf model.pth')
        