import torch
from unicore.losses import register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
import logging
import lmdb
import pickle
import numpy as np
logger = logging.getLogger(__name__)

OLED_CONDITION_REGISTER = {
    'homo': 0,
    'lumo': 1,
}

OLED_BIN_CONDITION_REGISTER = {
    'homo': 
    {
        'max': -7.4439,
        'min': -11.9579,
    },
    'lumo': 
    {
        'max': -6.1116,
        'min': -11.0018,
    }
}

ATTR_REGESTRY = {
    'qm9': [-411.5369296957087, 40.05903843480452, 'standardization'],
    'oled': [-10.150410724524885, 0.4663760450930604, 'standardization'],
    'opv': [-4.701612763241587, 0.3305236380777162, 'standardization']
}


class Normalization(object):
    def __init__(self, mean=None, std=None, normal_type=None):
        self.mean = mean
        self.std = std
        self.normal_type = normal_type
    
    def transform(self, x):
        if self.normal_type == 'log1p_standardization':
            return (torch.log1p(x) - self.mean) / self.std
        elif self.normal_type == 'standardization':
            return (x - self.mean) / self.std
        elif self.normal_type == 'none':
            return x
        else:
            raise ValueError('normal_type should be log1p_standardization or standardization')
    
    def inverse_transform(self, x):
        if self.normal_type == 'log1p_standardization':
            return torch.expm1(x * self.std + self.mean)
        elif self.normal_type == 'standardization':
            return x * self.std + self.mean
        elif self.normal_type == 'none':
            return x
        else:
            raise ValueError('normal_type should be log1p_standardization or standardization')

@register_loss("finetune_cvae_infer")
class FinetuneCvaeInferLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)
        self.smi_dictionary = task.smi_dictionary
        self.eos_idx = task.smi_dictionary.eos()
        self.bos_idx = task.smi_dictionary.bos()
        self.indices = {v:k for k,v in self.smi_dictionary.indices.items()}
        self.task = task
        self.sample_size = self.task.sample_size
        self.condition = self.task.condition
        self.condition_method = self.task.condition_method
        self.max_len = self.task.max_len
        self.num_bins = self.task.num_bins
        _mean, _std, _normal_type = ATTR_REGESTRY[self.args.task_name]
        self.normalizer = Normalization(_mean, _std, _normal_type)


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        bsz = self.task.batch_size
        inference_output, condition = self.inference(model, self.condition, bsz, self.max_len, self.bos_idx, self.eos_idx, self.task)

        logging_output = {
                "sample_size": self.sample_size,
            }
        
        if True:
            def decode_smiles_from_indexes(vec_list):
                vec_list = vec_list.cpu().numpy().tolist()
                ss_list = []
                for vec in vec_list:
                    ss = ""
                    for item in vec:
                        if item == self.eos_idx or item == self.bos_idx:
                            break
                        ss += self.indices[item]
                    ss_list.append(ss.strip())
                return ss_list
            with torch.no_grad():
                sample_output_smi = decode_smiles_from_indexes(inference_output)
                logging_output['output_smi'] = sample_output_smi
                logging_output['condition'] = self.normalizer.inverse_transform(condition)
                print('sample smiles\n', sample_output_smi)

        return None, 1, logging_output

    def inference(self, model, condition, bsz, max_len, bos_idx, eos_idx, task):
        if isinstance(condition, float):
            condition = torch.tensor([condition]).repeat(bsz)
        elif isinstance(condition, str):
            ## TODO: Add more choices
            if 'lmdb' in condition:
                condition = self.load_lmdb(condition)
        else:
            raise ValueError("Invalid type for args.condition")
        condition_info = condition.unsqueeze(1).to(model.linear_1.weight.device)
        mean = torch.zeros(bsz, model.latent_dim)
        logvar = torch.ones(bsz, model.latent_dim)
        with torch.no_grad():
            if task.teacher_forcing:
                dec_in = torch.tensor([bos_idx] * bsz).unsqueeze(1)
                inference_output = model.inference(mean, logvar, condition_info, max_len, dec_in, eos_idx)
            else:
                inference_output = model.inference(mean, logvar, condition_info, max_len, eos_idx=eos_idx)
        return inference_output, condition
    
    ## debug: fix condition input
    def load_lmdb(self, path):
        
        env = lmdb.open(
            path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )

        condition_info_list = []

        with env.begin() as txn:
            for idx in range(self.task.batch_size):
                datapoint_pickled = txn.get(f"{idx}".encode("ascii"))
                if datapoint_pickled is None:
                    continue
                data = pickle.loads(datapoint_pickled)
                condition_info = data['energy'][0] ##TODO: Add more choices, refactor

                if self.condition_method == 'bin':
                    min_value = OLED_BIN_CONDITION_REGISTER['homo']['min']
                    max_value = OLED_BIN_CONDITION_REGISTER['homo']['max']
                    bin_edges = np.linspace(min_value, max_value, self.num_bins)
                    bin_labels = np.arange(self.num_bins)
                    condition_info = bin_labels[np.digitize(condition_info, bin_edges) - 1]
                condition_info_list.append(condition_info)
                condition = torch.tensor(condition_info_list)

        if self.condition_method == 'val':
            condition = self.normalizer.transform(torch.tensor(condition_info_list))
            
        return condition

        
