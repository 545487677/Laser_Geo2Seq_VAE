import torch
import time
import logging
import numpy as np
from unicore.losses import register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss

logger = logging.getLogger(__name__)

def unbatch_v_traj(v_traj, n_data):
    all_step_v = [[] for _ in range(n_data)]
    for v in v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[k])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v

@register_loss("finetune_ddpm_infer")
class FinetuneddpmInferLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)
        self.dictionary = task.dictionary
        self.eos_idx = task.dictionary.eos()
        self.bos_idx = task.dictionary.bos()
        self.prior_dis = task.prior_dis
        self.indices = {v:k for k,v in self.dictionary.indices.items()}
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

        # get max_len from fix value  qm9 24
        if self.max_len is  None:   
            self.max_len = list(self.prior_dis.keys())[0]

        # get max_len from prior distribution
        # num_atom_list, prob_list = list(self.prior_dis.keys()), list(self.prior_dis.values())
        # self.max_len = np.random.choice(num_atom_list, p=prob_list)
        
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = self.inference(model, bsz, self.max_len)

        result = {
            'pred_pos': pred_pos,
            'pred_v': pred_v,
            'pred_pos_traj': pred_pos_traj,
            'pred_v_traj': pred_v_traj,
            'time': time_list,
            'indices': self.indices
        }

        logging_output = {
                "sample_size": self.sample_size,
                **result
        }
        
        return None, 1, logging_output

    def inference(self, model, bsz, max_len, bos_idx=None, eos_idx=None, task=None, scale=None):

        all_pred_pos, all_pred_v = [], []
        all_pred_pos_traj, all_pred_v_traj = [], []
        all_pred_v0_traj, all_pred_vt_traj = [], []
        time_list = []
        t1 = time.time()

        # Fix value or random gen? 
        src_pair_charge = torch.randn((bsz, max_len, max_len, 2))

        with torch.no_grad():

            inference_output = model.inference(
               bsz, 
               max_len, 
               **{'src_pair_charge': src_pair_charge}
            )

            pos, v, pos_traj, v_traj = inference_output['pos'], inference_output['v'], inference_output['pos_traj'], inference_output['v_traj']
            v0_traj, vt_traj = inference_output['v0_traj'], inference_output['vt_traj']

            num_atoms = [self.max_len for _ in range(bsz)]
            cum_atoms = np.cumsum([0] + num_atoms)

            pos_array = pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [pos_array[i] for i in range(bsz)] # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(bsz)]    

            for p in pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(bsz):
                    all_step_pos[k].append(p_array[k])     

            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]


            v_array = v.cpu().numpy()
            all_pred_v += [v_array[k] for k in range(bsz)]

            all_step_v = unbatch_v_traj(v_traj, bsz)
            all_pred_v_traj += [v for v in all_step_v]

            all_step_v0 = unbatch_v_traj(v0_traj, bsz)
            all_pred_v0_traj += [v for v in all_step_v0]

            all_step_vt = unbatch_v_traj(vt_traj, bsz)
            all_pred_vt_traj += [v for v in all_step_vt]

            t2 = time.time()
            time_list.append(t2 - t1)



        return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


        
