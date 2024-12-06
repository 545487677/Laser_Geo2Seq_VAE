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
# import openbabel
from rdkit import Chem
from tqdm import tqdm
from ..utils import save_xyz_fileori, check_stabilityori
from ..models.geoori import HISTOGRAM
logger = logging.getLogger(__name__)

ATOM_DICT = {
    "qm9":{
        'H': 1,
        'B': 5,
        'C': 6,
        'N': 7,
        'O': 8,
        'S': 16,
        'Se': 34,
    }

}

def xyz2smi(atoms, coords):
    mol = openbabel.OBMol()
    for j in range(len(coords)):
        atom = mol.NewAtom()
        atom.SetAtomicNum(openbabel.GetAtomicNum(atoms[j]))
        x, y, z = map(float, coords[j])
        atom.SetVector(x, y, z)
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat('smi')
    smi = obConversion.WriteString(mol)
    # smi = smi.replace('[H].', '').replace('.[H]', '')
    return smi.split('\t\n')[0]

@register_loss("finetune_geoldm_infer")
class FinetunegeoldmInferLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)
        self.dictionary = task.dictionary
        self.eos_idx = task.dictionary.eos()
        self.bos_idx = task.dictionary.bos()
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
        with torch.no_grad():
            model.eval()
            bsz = self.task.batch_size
            inference_output = self.inference(model, bsz, self.dictionary, self.sample_size)

        logging_output = {
                "sample_size": self.sample_size,
                "output_smi": inference_output['smi_list'],
                "output_xyz": inference_output['output_xyz'],
                "output_atoms": inference_output['output_atoms'],
        }

        return None, 1, logging_output

    def inference(self, model, bsz, dictionary, n_samples):

        batch_size = min(bsz, n_samples)
        assert n_samples % batch_size == 0
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
        nodes_dist = model.nodes_dist
        start_time = time.time()
        logger.info(
            "num. model params: {:,} (num. trained: {:,})".format(
                sum(getattr(p, "_orig_size", p).numel() for p in model.parameters()),
                sum(getattr(p, "_orig_size", p).numel() for p in model.parameters() if p.requires_grad),
            )
        )
        for i in range(int(n_samples/batch_size)):
            
            nodesxsample = nodes_dist.sample(batch_size)
            one_hot, charges, x, node_mask = model.sample(
                                                prop_dist=None,
                                                nodesxsample=nodesxsample
                                            )
            molecules['one_hot'].append(one_hot.detach().cpu())
            molecules['x'].append(x.detach().cpu())
            molecules['node_mask'].append(node_mask.detach().cpu())
            
            current_num_samples = (i+1) * batch_size
            secs_per_sample = (time.time() - start_time) / current_num_samples
            print('\t %d/%d Molecules generated at %.2f secs/sample' % (
                current_num_samples, n_samples, secs_per_sample))

            dataset_info = {
                'atom_decoder':
                    list(self.dictionary.indices.keys())

            }

            id_from = i * batch_size
            save_xyz_fileori(f'{self.task.args.results_path}/outputs/{self.task.args.task_name}/sample_molecule/', one_hot, charges, x, dataset_info, id_from, name='molecule')

        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
        dataset_info = {
                'atom_decoder':
                    list(self.dictionary.indices.keys())

            }
        smi_list, atoms_list, coord_list, valid_dict, mol_list = self.compute_molecule_stability(molecules, dataset_info, check_stabilityori, self.task.args.task_name)

        # TODO: Need to fix error
        # return {'output_smi': smi_list, 'output_xyz': coord_list, 'output_atoms': atoms_list}
        return {'smi_list': smi_list, 'output_xyz': coord_list, 'output_atoms': atoms_list, 'valid_dict': valid_dict, 'mol_list': mol_list}


    def compute_molecule_stability(self, molecules, dataset_info, check_stability, task_name):

        one_hot = molecules['one_hot']
        x = molecules['x']
        node_mask = molecules['node_mask']

        if isinstance(node_mask, torch.Tensor):
            atomsxmol = torch.sum(node_mask, dim=1)
        else:
            atomsxmol = [torch.sum(m) for m in node_mask]

        n_samples = len(x)

        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0

        processed_list = []
        atoms_list = []
        coord_list = []
        smi_list = []
        mol_list = []       

        for i in range(n_samples):
            atom_type = one_hot[i].argmax(1).cpu().detach()
            pos = x[i].cpu().detach()
            atom_type = atom_type[0:int(atomsxmol[i])]
            pos = pos[0:int(atomsxmol[i])]
            processed_list.append((pos, atom_type))
        
        for mol in processed_list:
            pos, atom_type = mol

            validity_results = check_stability(pos, atom_type, dataset_info, task_name=task_name)

            atoms = [dataset_info['atom_decoder'][i.item()] for i in atom_type.numpy()]
            smi = xyz2smi(atoms, pos.numpy())

            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])
            
            if '.' not in smi:
                smi_list.append(smi)
            atoms_list.append(atoms)
            coord_list.append(pos.numpy())
            mol_list.append(validity_results[3])

        # Validity
        fraction_mol_stable = (molecule_stable / float(n_samples))*100
        fraction_atm_stable = (nr_stable_bonds / float(n_atoms))*100

        validity_dict = {
            'mol_stable': fraction_mol_stable,
            'atm_stable': fraction_atm_stable,
        }

        logger.info("mol_stable: mean={:.2f}%, sd={:.2f}%, vals={}%".format(np.mean(fraction_mol_stable), np.std(fraction_mol_stable), fraction_mol_stable))
        logger.info('atm_stable: mean={:.2f}%, sd={:.2f}%, vals={}%'.format(np.mean(fraction_atm_stable), np.std(fraction_atm_stable), fraction_atm_stable))

        return smi_list, atoms_list, coord_list, validity_dict, mol_list
        