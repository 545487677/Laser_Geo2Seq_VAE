# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os 
import numpy as np
import torch
import glob
import random
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit import RDLogger
rdBase.DisableLog('rdApp.error')
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
plt.style.use('default')
import imageio
import numpy as np
import torch.nn as nn


def draw_valid_smiles(path, smiles):
    from rdkit.Chem import Draw
    # Calculate the number of rows and columns for the subplot grid
    num_molecules = len(smiles)
    cols = 5
    rows = (num_molecules + cols - 1) // cols  # Calculate the number of rows dynamically

    # Create a new figure and set up subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))

    # Loop through each SMILES string and visualize the molecule in a subplot
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        img = Draw.MolToImage(mol, size=(250, 250))
        
        # Calculate the subplot position for the current molecule
        row_idx = i // cols
        col_idx = i % cols
        
        # Plot the molecule in the appropriate subplot
        axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].axis('off')
        axes[row_idx, col_idx].set_title(f"Molecule {i+1}")

    # Adjust layout and save the plot to a file
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'molecule_subplots.png'), dpi=200) 
    plt.show()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def inference_greedy(output):
    return torch.argmax(output[:,-1,:], dim=1, keepdim=True)
    
def inference_top_p(output, p=0.9, temperature=0.6):
    probs = torch.softmax(output[:, -1,: ] / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def inference_beam(output, beam_size=5):
    ## TODO: Implement
    pass

def inference_top_k(output, k=5):
    probs = torch.softmax(output[:, -1, :], dim=-1)
    _, top_k_indices = torch.topk(probs, k=k, dim=-1)
    rand_indices = torch.randint(0, k, top_k_indices.size()[:-1], device=top_k_indices.device).to(torch.int64)
    next_token = top_k_indices.gather(dim=-1, index=rand_indices.unsqueeze(-1))
    return next_token

class Metrics(object):
    
    @staticmethod
    def canonical_smiles(smi):
        try:
            RDLogger.DisableLog('rdApp.*')
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        except Chem.MolSanitizeException:
            return None

    @staticmethod
    def is_valid(smi, opt_mol=False):
        try:
            molecule = Chem.MolFromSmiles(smi)
            if molecule is None:
                return None
        except:
            return None
        molecule = Chem.AddHs(molecule)
        if opt_mol: ## Geometry optimization for molecules
            try:
                AllChem.EmbedMolecule(molecule)
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                return None
        return molecule

    @staticmethod
    def check_novelty(num_threads, gen_smiles, train_smiles):
        with multiprocessing.Pool(processes=num_threads) as pool:
            gen_smiles = list(tqdm(pool.imap(Metrics.canonical_smiles, gen_smiles), total=len(gen_smiles)))

        gen_smiles = [smi for smi in gen_smiles if smi is not None]
        
        train_smiles = [smi for smi in train_smiles if smi is not None]
        if len(gen_smiles) == 0:
            novelty_ratio = 0.
        else:
            duplicates = [1 for mol in gen_smiles if mol in train_smiles]
            novel = len(gen_smiles) - sum(duplicates)
            novelty_ratio = novel/len(gen_smiles)    

            # unique_generated = set(gen_smiles)
            # unique_train = set(train_smiles)
            # novel_molecules = unique_generated.difference(unique_train)
            # novelty_ratio = len(novel_molecules) / len(unique_generated)

        return novelty_ratio



    @staticmethod
    def evaluate(log_outputs, condition_truth, args):
        num_threads = multiprocessing.cpu_count()
        
        with multiprocessing.Pool(processes=num_threads) as pool:
            valid = list(tqdm(pool.imap(Metrics.is_valid, log_outputs), total=len(log_outputs)))

        valid_id = [idx for idx, smile in enumerate(valid) if smile is not None]
        valid = [mol for mol in valid if mol is not None]
        if hasattr(args, 'condition'):
            condition_truth = [condition_truth[i] for i in valid_id]
        
        
        valid_ratio = len(valid) / args.sample_size
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid]

        unique_smiles = list(set(valid_smiles))
        unique_ratio = len(unique_smiles) / len(valid)

        valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
        train_smiles = np.load(os.path.join(args.data, args.task_name, 'train_canonical_smiles.npy'), allow_pickle=True).tolist()

        novel_ratio = Metrics.check_novelty(num_threads, valid_smiles, train_smiles)
        os.makedirs(args.results_path, exist_ok=True)
        # np.save(os.path.join(args.results_path, 'valid_smi.npy'), valid_smiles)
        # np.save(os.path.join(args.results_path, 'valid_con.npy'), condition_truth)

        # load 80w origin smiles 
        ori_80w_smiles = np.load('/vepfs/fs_users/guojianz/dp_project/laser_vae_gen/check/canonical_smiles.npy', allow_pickle=True).tolist()
        ori_80w_smiles = set(ori_80w_smiles)
        unique_smiles = set(unique_smiles).difference(ori_80w_smiles)

        unique_smiles = list(unique_smiles)

        results = dict()
        results['valid_mols'] = valid_mols
        results['valid_smiles'] = valid_smiles
        results['unique_smiles'] = unique_smiles
        results['valid_ratio'] = valid_ratio * 100
        results['unique_ratio'] = unique_ratio * 100
        results['novel_ratio'] = novel_ratio * 100


        return results


# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161, 'Se': 146},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152},
          'Se': {'H': 146}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150, 'S': 143},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186, 'S': 149, 'O': 143},
          'Se': {'Se':215}
}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}


def print_table(bonds_dict):
    letters = ['H', 'C', 'O', 'N', 'P', 'S', 'F', 'Si', 'Cl', 'Br', 'I']

    new_letters = []
    for key in (letters + list(bonds_dict.keys())):
        if key in bonds_dict.keys():
            if key not in new_letters:
                new_letters.append(key)

    letters = new_letters

    for j, y in enumerate(letters):
        if j == 0:
            for x in letters:
                print(f'{x} & ', end='')
            print()
        for i, x in enumerate(letters):
            if i == 0:
                print(f'{y} & ', end='')
            if x in bonds_dict[y]:
                print(f'{bonds_dict[y][x]} & ', end='')
            else:
                print('- & ', end='')
        print()


# print_table(bonds3)


def check_consistency_bond_dictionaries():
    for bonds_dict in [bonds1, bonds2, bonds3]:
        for atom1 in bonds1:
            for atom2 in bonds_dict[atom1]:
                bond = bonds_dict[atom1][atom2]
                try:
                    bond_check = bonds_dict[atom2][atom1]
                except KeyError:
                    raise ValueError('Not in dict ' + str((atom1, atom2)))

                assert bond == bond_check, (
                    f'{bond} != {bond_check} for {atom1}, {atom2}')


stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}
margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
                 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
                 'Bi': [3, 5], 'Se': [2,4,6]}

def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if not (atom1 in bonds1 or atom1 in bonds2 or atom1 in bonds3 or 
            atom2 in bonds1.get(atom1, {}) or atom2 in bonds2.get(atom1, {}) or atom2 in bonds3.get(atom1, {})):
        return 0
    
    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3        # Triple
                return 2            # Double
        return 1                # Single
    return 0                    # No bond


def single_bond_only(threshold, length, margin1=5):
    if length < threshold + margin1:
        return 1
    return 0


def geom_predictor(p, l, margin1=5, limit_bonds_to_one=False):
    """ p: atom pair (couple of str)
        l: bond length (float)"""
    bond_order = get_bond_order(p[0], p[1], l, check_exists=True)

    # If limit_bonds_to_one is enabled, every bond type will return 1.
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    else:
        return bond_order



def save_xyz_file(path, one_hot, charges, positions, dataset_info, id_from=0, name='molecule', node_mask=None):
    indices = {v:k for k,v in dataset_info.indices.items()}
    try:
        os.makedirs(path)
    except OSError:
        pass

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)

    for batch_i in range(one_hot.size(0)):
        f = open(path + name + '_' + "%03d.txt" % (batch_i + id_from), "w")
        f.write("%d\n\n" % atomsxmol[batch_i])
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        n_atoms = int(atomsxmol[batch_i])
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = indices[atom.item()]
            if atom in list(dataset_info.specials):
                continue
            f.write("%s %.9f %.9f %.9f\n" % (atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
        f.close()


def save_xyz_fileori(path, one_hot, charges, positions, dataset_info, id_from=0, name='molecule', node_mask=None):
    try:
        os.makedirs(path)
    except OSError:
        pass

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)

    for batch_i in range(one_hot.size(0)):
        f = open(path + name + '_' + "%03d.txt" % (batch_i + id_from), "w")
        f.write("%d\n\n" % atomsxmol[batch_i])
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        n_atoms = int(atomsxmol[batch_i])
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = dataset_info['atom_decoder'][atom.item()]
            f.write("%s %.9f %.9f %.9f\n" % (atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
        f.close()

def load_molecule_xyz(file, dataset_info):
    with open(file, encoding='utf8') as f:
        n_atoms = int(f.readline())
        one_hot = torch.zeros(n_atoms, len(dataset_info))
        charges = torch.zeros(n_atoms, 1)
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        indices = {k:v for k, v in dataset_info.indices.items()}
        for i in range(n_atoms):
            atom = atoms[i].split(' ')
            atom_type = atom[0]
            one_hot[i, indices[atom_type]] = 1 
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot, charges


def load_xyz_files(path, shuffle=True):
    files = glob.glob(path + "/*.txt")
    if shuffle:
        random.shuffle(files)
    return files

def check_stability(positions, atom_type, dataset_info, debug=False, task_name=None):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = {k:v for v,k in dataset_info.indices.items()}
    
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i].item()], atom_decoder[atom_type[j].item()]
            pair = sorted([atom_type[i], atom_type[j]])
            if task_name == 'qm9' or task_name == 'qm9_second_half' or task_name  == 'qm9_first_half':
                order = get_bond_order(atom1, atom2, dist)
            elif task_name == 'geom':
                order = geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
            
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type_i.item()]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i.item()], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)

bond_list = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]

def check_stabilityori(positions, atom_type, dataset_info, debug=False, task_name=None, dictionary=None):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    # atoms
    mol = Chem.RWMol()
    for atom in atom_type:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    # add positions to Mol
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
    mol.AddConformer(conf)            

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i].item()], atom_decoder[atom_type[j].item()]
            pair = sorted([atom_type[i], atom_type[j]])
            if task_name == 'qm9' or task_name == 'qm9_second_half' or task_name  == 'qm9_first_half' or task_name == 'qm9_geoldm':
                order = get_bond_order(atom1, atom2, dist)
            elif task_name == 'geom' or task_name == 'oled' or task_name == 'oled_geoldm' or task_name == 'oled_geoldm1' or task_name == 'opv_geoldm':
                order = geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
            # add bond to RDKIT Mol
            if order > 0:
                mol.AddBond(i, j, bond_list[order])            

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        decoded_atom = atom_decoder[atom_type_i.item()]
        
        # Check if the decoded atom is in allowed_bonds
        if decoded_atom not in allowed_bonds:
            continue
        
        possible_bonds = allowed_bonds[decoded_atom]
        # possible_bonds = allowed_bonds[atom_decoder[atom_type_i.item()]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i.item()], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x), mol
    
# COLOR_DICS = ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14','#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b','#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d','#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b','#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
# RADIUS_DIC = [0.46, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77,0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77,0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77]

