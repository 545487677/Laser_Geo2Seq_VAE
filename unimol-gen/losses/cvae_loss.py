import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
import logging
logger = logging.getLogger(__name__)

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

@register_loss("finetune_cvae_loss")
class FinetuneCvaeLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)
        self.smi_dictionary = task.smi_dictionary
        self.smi_padding_idx = task.smi_dictionary.pad()
        self.eos_idx = task.smi_dictionary.eos()
        self.bos_idx = task.smi_dictionary.bos()
        self.indices = {v:k for k,v in self.smi_dictionary.indices.items()}
        self.task = task

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if self.task.condition_method == 'val':
            _mean, _std, _normal_type = ATTR_REGESTRY[self.args.task_name]
            normalizer = Normalization(_mean, _std, _normal_type)
            sample["net_input"]['condition_info'] = normalizer.transform(sample["net_input"]['condition_info'])
            

        if self.task.teacher_forcing:
            net_output = model(
                **sample["net_input"],
            )
            target_smi_token = sample['net_input']['tgt_tokens'][:, 1:]
            output = net_output[0][:, :-1]
        else:
            net_output = model(
                **sample["net_input"],
            )
            target_smi_token = sample['net_input']['tgt_tokens']
            output = net_output[0]
        mean = net_output[1] 
        logvar = net_output[2]

        target_smi_token = target_smi_token.to(output.dtype)
        bsz = target_smi_token.size(0)

        xent_loss, kl_loss = self.cvae_loss(output.float(), target_smi_token.float(), mean.float(), logvar.float())
        loss = xent_loss + kl_loss
        logging_output = {
                "loss": loss.data,
                "xent_loss": xent_loss.data,
                "kl_loss": kl_loss.data,
                "sample_size": 1,
                "bsz": bsz
            }
        if not self.training:
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
                output_smi = decode_smiles_from_indexes(torch.argmax(output, dim=-1))
                print('output_smi:\n', output_smi[:2])
                print('smi:\n', sample['smi'][:2])


        return loss, 1, logging_output
    
    ## cross entropy loss
    def cvae_loss(self, x_decoded_mean, x, z_mean, z_logvar):
        sz = x_decoded_mean.size(-1)
        mask = x != self.smi_padding_idx
        xent_loss = F.cross_entropy(x_decoded_mean[mask].view(-1, sz), x[mask].long(), reduction='sum')
        xent_loss = xent_loss / mask.size(0)
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kl_loss = kl_loss / mask.size(0)
        return xent_loss, kl_loss



    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        xent_loss_sum = sum(log.get("xent_loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "xent_loss", xent_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "kl_loss", kl_loss_sum / sample_size, sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
        