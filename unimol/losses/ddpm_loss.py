import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
import logging
import time
from tqdm import tqdm
logger = logging.getLogger(__name__)


def unbatch_v_traj(v_traj, n_data):
    all_step_v = [[] for _ in range(n_data)]
    for v in v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[k])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v

@register_loss("finetune_ddpm_loss")
class FinetuneddpmLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)
        self.dictionary = task.dictionary
        self.eos_idx = task.dictionary.eos()
        self.bos_idx = task.dictionary.bos()
        self.indices = {v:k for k,v in self.dictionary.indices.items()}
        self.task = task

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """


        net_output = model(
            **sample["net_input"]
        )

        bsz = net_output['x0'].size(0)
        loss, loss_v, loss_pos = net_output['loss'], net_output['loss_v'], net_output['loss_pos']

        logging_output = {
                "loss_pos": loss_pos.data,
                "loss_v": loss_v.data,
                "loss": loss.data,
                "sample_size": 1,
                "bsz": bsz,
                # "acc": accuracy,
        }

        ####test
        # if True:
        #     logger.info("valid atom type")
        #     with torch.no_grad():
        #         model.eval()
        #         special_idx_tensor = torch.tensor(model.special_idx)
        #         mask = (~torch.isin(sample["net_input"]['src_tokens'], special_idx_tensor.to(sample["net_input"]['src_tokens']))).int().cpu().numpy()
                
        #         time_step = torch.tensor([0] * bsz).to(sample["net_input"]['src_tokens'])
        #         net_output = model(
        #                     **sample["net_input"],
        #                     time_step = time_step,
        #             )

        #         gt = sample["net_input"]['src_tokens'].detach().cpu().numpy() * mask
        #         pred_v = net_output['v_recon'].cpu().numpy() * mask
        #         correct_predictions = (pred_v == gt).sum()
        #         total_predictions = pred_v.size
        #         accuracy = (correct_predictions / total_predictions) * 100.0 # also record special num
        #         print("gt atom type list:\n", gt[:4])
        #         print("pred atom type list:\n", pred_v[:4])
        #         print("acc:\n",accuracy)

        #         ## test t=0 pred pos
        #         gt_pos = sample["net_input"]['src_coord'].detach().cpu().numpy() * mask[..., None]
        #         pred_v_pos = net_output['pred_pos'].cpu().numpy() * mask[..., None]
        #         result = {
        #             'pred_pos': pred_v_pos,
        #             'pred_v': pred_v,
        #             'indices': self.indices,
        #             'gt_pos':gt_pos,
        #             'gt_v': gt,
        #         }
        #         # os.system('mkdir -p ./test')
        #         # np.save(f'./test/{self.args.task_name}_test_pred.npy', result)

        #     logger.info('valid sampling check')

        #     with torch.no_grad():
        #         pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, variance_list, variance_gt_list, sqrt_one_minus_alpha_cum = self.inference(model, bsz, sample["net_input"]['src_tokens'].size(1), sample["net_input"]['src_coord'], sample["net_input"]['src_tokens'])
        #         result = {
        #             'pred_pos': pred_pos,
        #             'pred_v': pred_v,
        #             'pred_pos_traj': pred_pos_traj,
        #             'pred_v_traj': pred_v_traj,
        #             'time': time_list,
        #             'indices': self.indices,
        #             'pred_v0_traj': pred_v0_traj,
        #             'pred_vt_traj': pred_vt_traj,
        #             'gt_pos':sample["net_input"]['src_coord'].detach().cpu().numpy(),
        #             'gt_v': sample["net_input"]['src_tokens'].detach().cpu().numpy(),
        #             'variance_list': [i.detach().cpu().numpy() for i in variance_list],
        #             'variance_gt_list': [i.detach().cpu().numpy() for i in variance_gt_list],
        #             'sqrt_one_minus_alpha_cum': [i.detach().cpu().numpy() for i in sqrt_one_minus_alpha_cum],

        #         }
        #         os.system('mkdir -p ./test')
        #         np.save(f'./test/{self.args.task_name}_result_valid_sample.npy', result)





                # def decode_atom_from_index_skip_zero(vec_list, indices):
                #     decoded = []
                #     for row in vec_list:
                #         decoded_row = [indices[int(i.item())] for i in row]
                #         decoded.append(decoded_row)
                #     return decoded
                # output_atom = decode_atom_from_index_skip_zero(net_output['v_recon'].detach().cpu().numpy(), self.indices)
                # gt_atom = decode_atom_from_index_skip_zero(sample["net_input"]['src_tokens'].detach().cpu().numpy(), self.indices)
                # print("gt atom type list:\n", gt_atom[:2])
                # print("pred atom type list:\n", output_atom[:2])



        if not self.training:
            logger.info("valid atom type")
            with torch.no_grad():
                model.eval()
                special_idx_tensor = torch.tensor(model.special_idx)
                mask = (~torch.isin(sample["net_input"]['src_tokens'], special_idx_tensor.to(sample["net_input"]['src_tokens']))).int().cpu().numpy()
                
                time_step = torch.tensor([0] * bsz).to(sample["net_input"]['src_tokens'])
                net_output = model(
                            **sample["net_input"],
                            time_step = time_step,
                    )

                gt = sample["net_input"]['src_tokens'].detach().cpu().numpy() * mask
                pred_v = net_output['v_recon'].cpu().numpy() * mask
                correct_predictions = (pred_v == gt).sum()
                total_predictions = pred_v.size
                accuracy = (correct_predictions / total_predictions) * 100.0 # also record special num
                ## test t=0 pred pos
                gt_pos = sample["net_input"]['src_coord'].detach().cpu().numpy() * mask[..., None]
                pred_v_pos = net_output['pred_pos'].cpu().numpy() * mask[..., None]
                print("gt atom type list:\n", gt[:4])
                print("pred atom type list:\n", pred_v[:4])
                print("acc:\n",accuracy)
                print("gt_pos mean:\n", gt_pos.mean())
                print("pred_v_pos mean:\n", pred_v_pos.mean())
                

            # logger.info('valid sampling check')

            # with torch.no_grad():
            #     pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, variance_list, variance_gt_list, sqrt_one_minus_alpha_cum = self.inference(model, bsz, sample["net_input"]['src_tokens'].size(1), sample["net_input"]['src_coord'], sample["net_input"]['src_tokens'])
            #     result = {
            #         'pred_pos': pred_pos,
            #         'pred_v': pred_v,
            #         'pred_pos_traj': pred_pos_traj,
            #         'pred_v_traj': pred_v_traj,
            #         'time': time_list,
            #         'indices': self.indices,
            #         'pred_v0_traj': pred_v0_traj,
            #         'pred_vt_traj': pred_vt_traj,
            #         'gt_pos':sample["net_input"]['src_coord'].detach().cpu().numpy(),
            #         'gt_v': sample["net_input"]['src_tokens'].detach().cpu().numpy(),
            #         'variance_list': [i.detach().cpu().numpy() for i in variance_list],
            #         'variance_gt_list': [i.detach().cpu().numpy() for i in variance_gt_list],
            #         'sqrt_one_minus_alpha_cum': [i.detach().cpu().numpy() for i in sqrt_one_minus_alpha_cum],

            #     }
            #     os.system('mkdir -p ./test')
            #     np.save(f'./test/{self.args.task_name}_result_valid_sample.npy', result)



        return loss, 1, logging_output


    def inference(self, model, bsz, max_len, gt_pos=None, gt_v=None):

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
               **{'src_pair_charge': src_pair_charge},
               **{'gt_pos': gt_pos},
               **{'gt_v': gt_v},
            )

            pos, v, pos_traj, v_traj = inference_output['pos'], inference_output['v'], inference_output['pos_traj'], inference_output['v_traj']
            v0_traj, vt_traj = inference_output['v0_traj'], inference_output['vt_traj']
            variance_list = inference_output['variance_list']
            variance_gt_list = inference_output['variance_gt_list']
            sqrt_one_minus_alpha_cum = inference_output['sqrt_one_minus_alpha_cum']

            # num_atoms = [max_len for _ in range(bsz)]
            # cum_atoms = np.cumsum([0] + num_atoms)

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



        return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list, variance_list, variance_gt_list, sqrt_one_minus_alpha_cum


    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        v_loss_sum = sum(log.get("loss_v", 0) for log in logging_outputs)
        pos_loss_sum = sum(log.get("loss_pos", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # acc = sum(log.get("acc", 0) for log in logging_outputs)

        
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss_v", v_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss_pos", pos_loss_sum / sample_size, sample_size, round=3
        )
        # metrics.log_scalar(
        #     "acc", acc / sample_size, sample_size, round=3
        # )


    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
        