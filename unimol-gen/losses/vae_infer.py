import torch
import logging
from unicore.losses import register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss

logger = logging.getLogger(__name__)
@register_loss("finetune_vae_infer")
class FinetunevaeInferLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)
        self.smi_dictionary = task.smi_dictionary
        self.eos_idx = task.smi_dictionary.eos()
        self.bos_idx = task.smi_dictionary.bos()
        self.indices = {v:k for k,v in self.smi_dictionary.indices.items()}
        self.task = task
        self.sample_size = self.task.sample_size
        self.max_len = self.task.max_len
        self.scale = self.task.scale


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        bsz = self.task.batch_size
        inference_output = self.inference(model, bsz, self.max_len, self.bos_idx, self.eos_idx, self.task, self.scale)

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

        return None, 1, logging_output

    def inference(self, model, bsz, max_len, bos_idx, eos_idx, task, scale):
        mean = torch.zeros(bsz, model.latent_dim)
        logvar = torch.ones(bsz, model.latent_dim)
        logvar *= scale

        with torch.no_grad():
            if task.teacher_forcing:
                dec_in = torch.tensor([bos_idx] * bsz).unsqueeze(1)
                inference_output = model.inference(mean, logvar, None, max_len, dec_in, eos_idx)
            else:
                inference_output = model.inference(mean, logvar, None, max_len, eos_idx=eos_idx)
        return inference_output


        
