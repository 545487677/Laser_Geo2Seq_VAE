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

        np.save(os.path.join(args.results_path, 'valid_smi.npy'), valid_smiles)
        np.save(os.path.join(args.results_path, 'valid_con.npy'), condition_truth)

        results = dict()
        results['valid_mols'] = valid_mols
        results['valid_smiles'] = valid_smiles
        results['unique_smiles'] = unique_smiles
        results['valid_ratio'] = valid_ratio * 100
        results['unique_ratio'] = unique_ratio * 100
        results['novel_ratio'] = novel_ratio * 100


        return results



# """
# https://github.com/mattragoza/liGAN/blob/master/fitting.py

# License: GNU General Public License v2.0
# https://github.com/mattragoza/liGAN/blob/master/LICENSE
# """
# import itertools
# import py3Dmol

# import numpy as np
# from rdkit.Chem import AllChem as Chem
# from rdkit import Geometry
# from openbabel import openbabel as ob
# from scipy.spatial.distance import pdist
# from scipy.spatial.distance import squareform


# class MolReconsError(Exception):
#     pass


# def reachable_r(a, b, seenbonds):
#     '''Recursive helper.'''

#     for nbr in ob.OBAtomAtomIter(a):
#         bond = a.GetBond(nbr).GetIdx()
#         if bond not in seenbonds:
#             seenbonds.add(bond)
#             if nbr == b:
#                 return True
#             elif reachable_r(nbr, b, seenbonds):
#                 return True
#     return False


# def reachable(a, b):
#     '''Return true if atom b is reachable from a without using the bond between them.'''
#     if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
#         return False  # this is the _only_ bond for one atom
#     # otherwise do recursive traversal
#     seenbonds = set([a.GetBond(b).GetIdx()])
#     return reachable_r(a, b, seenbonds)


# def forms_small_angle(a, b, cutoff=60):
#     '''Return true if bond between a and b is part of a small angle
#     with a neighbor of a only.'''

#     for nbr in ob.OBAtomAtomIter(a):
#         if nbr != b:
#             degrees = b.GetAngle(a, nbr)
#             if degrees < cutoff:
#                 return True
#     return False


# def make_obmol(xyz, atomic_numbers):
#     mol = ob.OBMol()
#     mol.BeginModify()
#     atoms = []
#     for xyz, t in zip(xyz, atomic_numbers):
#         x, y, z = xyz
#         # ch = struct.channels[t]
#         atom = mol.NewAtom()
#         atom.SetAtomicNum(t)
#         atom.SetVector(x, y, z)
#         atoms.append(atom)
#     return mol, atoms


# def connect_the_dots(mol, atoms, indicators, covalent_factor=1.3):
#     '''Custom implementation of ConnectTheDots.  This is similar to
#     OpenBabel's version, but is more willing to make long bonds 
#     (up to maxbond long) to keep the molecule connected.  It also 
#     attempts to respect atom type information from struct.
#     atoms and struct need to correspond in their order
#     Assumes no hydrogens or existing bonds.
#     '''

#     """
#     for now, indicators only include 'is_aromatic'
#     """
#     pt = Chem.GetPeriodicTable()

#     if len(atoms) == 0:
#         return

#     mol.BeginModify()

#     # just going to to do n^2 comparisons, can worry about efficiency later
#     coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])
#     dists = squareform(pdist(coords))
#     # types = [struct.channels[t].name for t in struct.c]

#     for i, j in itertools.combinations(range(len(atoms)), 2):
#         a = atoms[i]
#         b = atoms[j]
#         a_r = ob.GetCovalentRad(a.GetAtomicNum()) * covalent_factor
#         b_r = ob.GetCovalentRad(b.GetAtomicNum()) * covalent_factor
#         if dists[i, j] < a_r + b_r:
#             flag = 0
#             if indicators and indicators[i] and indicators[j]:
#                 flag = ob.OB_AROMATIC_BOND
#             mol.AddBond(a.GetIdx(), b.GetIdx(), 1, flag)

#     atom_maxb = {}
#     for (i, a) in enumerate(atoms):
#         # set max valance to the smallest max allowed by openbabel or rdkit
#         # since we want the molecule to be valid for both (rdkit is usually lower)
#         maxb = min(ob.GetMaxBonds(a.GetAtomicNum()), pt.GetDefaultValence(a.GetAtomicNum()))

#         if a.GetAtomicNum() == 16:  # sulfone check
#             if count_nbrs_of_elem(a, 8) >= 2:
#                 maxb = 6

#         # if indicators[i][ATOM_FAMILIES_ID['Donor']]:
#         #     maxb -= 1 #leave room for hydrogen
#         # if 'Donor' in types[i]:
#         #     maxb -= 1 #leave room for hydrogen
#         atom_maxb[a.GetIdx()] = maxb

#     # remove any impossible bonds between halogens
#     for bond in ob.OBMolBondIter(mol):
#         a1 = bond.GetBeginAtom()
#         a2 = bond.GetEndAtom()
#         if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
#             mol.DeleteBond(bond)

#     def get_bond_info(biter):
#         '''Return bonds sorted by their distortion'''
#         bonds = [b for b in biter]
#         binfo = []
#         for bond in bonds:
#             bdist = bond.GetLength()
#             # compute how far away from optimal we are
#             a1 = bond.GetBeginAtom()
#             a2 = bond.GetEndAtom()
#             ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum())
#             stretch = bdist / ideal
#             binfo.append((stretch, bond))
#         binfo.sort(reverse=True, key=lambda t: t[0])  # most stretched bonds first
#         return binfo

#     binfo = get_bond_info(ob.OBMolBondIter(mol))
#     # now eliminate geometrically poor bonds
#     for stretch, bond in binfo:

#         # can we remove this bond without disconnecting the molecule?
#         a1 = bond.GetBeginAtom()
#         a2 = bond.GetEndAtom()

#         # as long as we aren't disconnecting, let's remove things
#         # that are excessively far away (0.45 from ConnectTheDots)
#         # get bonds to be less than max allowed
#         # also remove tight angles, because that is what ConnectTheDots does
#         if stretch > 1.2 or forms_small_angle(a1, a2) or forms_small_angle(a2, a1):
#             # don't fragment the molecule
#             if not reachable(a1, a2):
#                 continue
#             mol.DeleteBond(bond)

#     # prioritize removing hypervalency causing bonds, do more valent
#     # constrained atoms first since their bonds introduce the most problems
#     # with reachability (e.g. oxygen)
#     hypers = [(atom_maxb[a.GetIdx()], a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms]
#     hypers = sorted(hypers, key=lambda aa: (aa[0], -aa[1]))
#     for mb, diff, a in hypers:
#         if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
#             continue
#         binfo = get_bond_info(ob.OBAtomBondIter(a))
#         for stretch, bond in binfo:

#             if stretch < 0.9:  # the two atoms are too closed to remove the bond
#                 continue
#             # can we remove this bond without disconnecting the molecule?
#             a1 = bond.GetBeginAtom()
#             a2 = bond.GetEndAtom()

#             # get right valence
#             if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
#                 # don't fragment the molecule
#                 if not reachable(a1, a2):
#                     continue
#                 mol.DeleteBond(bond)
#                 if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
#                     break  # let nbr atoms choose what bonds to throw out

#     mol.EndModify()


# def convert_ob_mol_to_rd_mol(ob_mol, struct=None):
#     '''Convert OBMol to RDKit mol, fixing up issues'''
#     ob_mol.DeleteHydrogens()
#     n_atoms = ob_mol.NumAtoms()
#     rd_mol = Chem.RWMol()
#     rd_conf = Chem.Conformer(n_atoms)

#     for ob_atom in ob.OBMolAtomIter(ob_mol):
#         rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
#         # TODO copy format charge
#         if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
#             # don't commit to being aromatic unless rdkit will be okay with the ring status
#             # (this can happen if the atoms aren't fit well enough)
#             rd_atom.SetIsAromatic(True)
#         i = rd_mol.AddAtom(rd_atom)
#         ob_coords = ob_atom.GetVector()
#         x = ob_coords.GetX()
#         y = ob_coords.GetY()
#         z = ob_coords.GetZ()
#         rd_coords = Geometry.Point3D(x, y, z)
#         rd_conf.SetAtomPosition(i, rd_coords)

#     rd_mol.AddConformer(rd_conf)

#     for ob_bond in ob.OBMolBondIter(ob_mol):
#         i = ob_bond.GetBeginAtomIdx() - 1
#         j = ob_bond.GetEndAtomIdx() - 1
#         bond_order = ob_bond.GetBondOrder()
#         if bond_order == 1:
#             rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
#         elif bond_order == 2:
#             rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
#         elif bond_order == 3:
#             rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
#         else:
#             raise Exception('unknown bond order {}'.format(bond_order))

#         if ob_bond.IsAromatic():
#             bond = rd_mol.GetBondBetweenAtoms(i, j)
#             bond.SetIsAromatic(True)

#     rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)

#     pt = Chem.GetPeriodicTable()
#     # if double/triple bonds are connected to hypervalent atoms, decrement the order

#     positions = rd_mol.GetConformer().GetPositions()
#     nonsingles = []
#     for bond in rd_mol.GetBonds():
#         if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE:
#             i = bond.GetBeginAtomIdx()
#             j = bond.GetEndAtomIdx()
#             dist = np.linalg.norm(positions[i] - positions[j])
#             nonsingles.append((dist, bond))
#     nonsingles.sort(reverse=True, key=lambda t: t[0])

#     for (d, bond) in nonsingles:
#         a1 = bond.GetBeginAtom()
#         a2 = bond.GetEndAtom()

#         if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
#                 calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
#             btype = Chem.BondType.SINGLE
#             if bond.GetBondType() == Chem.BondType.TRIPLE:
#                 btype = Chem.BondType.DOUBLE
#             bond.SetBondType(btype)

#     for atom in rd_mol.GetAtoms():
#         # set nitrogens with 4 neighbors to have a charge
#         if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
#             atom.SetFormalCharge(1)

#     rd_mol = Chem.AddHs(rd_mol, addCoords=True)

#     positions = rd_mol.GetConformer().GetPositions()
#     center = np.mean(positions[np.all(np.isfinite(positions), axis=1)], axis=0)
#     for atom in rd_mol.GetAtoms():
#         i = atom.GetIdx()
#         pos = positions[i]
#         if not np.all(np.isfinite(pos)):
#             # hydrogens on C fragment get set to nan (shouldn't, but they do)
#             rd_mol.GetConformer().SetAtomPosition(i, center)

#     try:
#         Chem.SanitizeMol(rd_mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
#     except:
#         raise MolReconsError()
#     # try:
#     #     Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
#     # except: # mtr22 - don't assume mols will pass this
#     #     pass
#     #     # dkoes - but we want to make failures as rare as possible and should debug them
#     #     m = pybel.Molecule(ob_mol)
#     #     i = np.random.randint(1000000)
#     #     outname = 'bad%d.sdf'%i
#     #     print("WRITING",outname)
#     #     m.write('sdf',outname,overwrite=True)
#     #     pickle.dump(struct,open('bad%d.pkl'%i,'wb'))

#     # but at some point stop trying to enforce our aromaticity -
#     # openbabel and rdkit have different aromaticity models so they
#     # won't always agree.  Remove any aromatic bonds to non-aromatic atoms
#     for bond in rd_mol.GetBonds():
#         a1 = bond.GetBeginAtom()
#         a2 = bond.GetEndAtom()
#         if bond.GetIsAromatic():
#             if not a1.GetIsAromatic() or not a2.GetIsAromatic():
#                 bond.SetIsAromatic(False)
#         elif a1.GetIsAromatic() and a2.GetIsAromatic():
#             bond.SetIsAromatic(True)

#     return rd_mol


# def calc_valence(rdatom):
#     '''Can call GetExplicitValence before sanitize, but need to
#     know this to fix up the molecule to prevent sanitization failures'''
#     cnt = 0.0
#     for bond in rdatom.GetBonds():
#         cnt += bond.GetBondTypeAsDouble()
#     return cnt


# def count_nbrs_of_elem(atom, atomic_num):
#     '''
#     Count the number of neighbors atoms
#     of atom with the given atomic_num.
#     '''
#     count = 0
#     for nbr in ob.OBAtomAtomIter(atom):
#         if nbr.GetAtomicNum() == atomic_num:
#             count += 1
#     return count


# def fixup(atoms, mol, indicators=None):
#     '''Set atom properties to match channel.  Keep doing this
#     to beat openbabel over the head with what we want to happen.'''

#     """
#     for now, indicators only include 'is_aromatic'
#     """
#     mol.SetAromaticPerceived(True)  # avoid perception
#     for i, atom in enumerate(atoms):
#         # ch = struct.channels[t]
#         if indicators is not None:
#             if indicators[i]:
#                 atom.SetAromatic(True)
#                 atom.SetHyb(2)
#             else:
#                 atom.SetAromatic(False)

#         # if ind[ATOM_FAMILIES_ID['Donor']]:
#         #     if atom.GetExplicitDegree() == atom.GetHvyDegree():
#         #         if atom.GetHvyDegree() == 1 and atom.GetAtomicNum() == 7:
#         #             atom.SetImplicitHCount(2)
#         #         else:
#         #             atom.SetImplicitHCount(1) 

#         # elif ind[ATOM_FAMILIES_ID['Acceptor']]: # NOT AcceptorDonor because of else
#         #     atom.SetImplicitHCount(0)   

#         if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():  # Nitrogen, Oxygen
#             # this is a little iffy, ommitting until there is more evidence it is a net positive
#             # we don't have aromatic types for nitrogen, but if it
#             # is in a ring with aromatic carbon mark it aromatic as well
#             acnt = 0
#             for nbr in ob.OBAtomAtomIter(atom):
#                 if nbr.IsAromatic():
#                     acnt += 1
#             if acnt > 1:
#                 atom.SetAromatic(True)


# def raw_obmol_from_generated(data):
#     xyz = data.ligand_context_pos.clone().cpu().tolist()
#     atomic_nums = data.ligand_context_element.clone().cpu().tolist()
#     # indicators = data.ligand_context_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()

#     mol, atoms = make_obmol(xyz, atomic_nums)
#     return mol, atoms


# UPGRADE_BOND_ORDER = {Chem.BondType.SINGLE: Chem.BondType.DOUBLE, Chem.BondType.DOUBLE: Chem.BondType.TRIPLE}


# def postprocess_rd_mol_1(rdmol):
#     rdmol = Chem.RemoveHs(rdmol)

#     # Construct bond nbh list
#     nbh_list = {}
#     for bond in rdmol.GetBonds():
#         begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         if begin not in nbh_list:
#             nbh_list[begin] = [end]
#         else:
#             nbh_list[begin].append(end)

#         if end not in nbh_list:
#             nbh_list[end] = [begin]
#         else:
#             nbh_list[end].append(begin)

#     # Fix missing bond-order
#     for atom in rdmol.GetAtoms():
#         idx = atom.GetIdx()
#         num_radical = atom.GetNumRadicalElectrons()
#         if num_radical > 0:
#             for j in nbh_list[idx]:
#                 if j <= idx: continue
#                 nb_atom = rdmol.GetAtomWithIdx(j)
#                 nb_radical = nb_atom.GetNumRadicalElectrons()
#                 if nb_radical > 0:
#                     bond = rdmol.GetBondBetweenAtoms(idx, j)
#                     bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
#                     nb_atom.SetNumRadicalElectrons(nb_radical - 1)
#                     num_radical -= 1
#             atom.SetNumRadicalElectrons(num_radical)

#         num_radical = atom.GetNumRadicalElectrons()
#         if num_radical > 0:
#             atom.SetNumRadicalElectrons(0)
#             num_hs = atom.GetNumExplicitHs()
#             atom.SetNumExplicitHs(num_hs + num_radical)

#     return rdmol


# def postprocess_rd_mol_2(rdmol):
#     rdmol_edit = Chem.RWMol(rdmol)

#     ring_info = rdmol.GetRingInfo()
#     ring_info.AtomRings()
#     rings = [set(r) for r in ring_info.AtomRings()]
#     for i, ring_a in enumerate(rings):
#         if len(ring_a) == 3:
#             non_carbon = []
#             atom_by_symb = {}
#             for atom_idx in ring_a:
#                 symb = rdmol.GetAtomWithIdx(atom_idx).GetSymbol()
#                 if symb != 'C':
#                     non_carbon.append(atom_idx)
#                 if symb not in atom_by_symb:
#                     atom_by_symb[symb] = [atom_idx]
#                 else:
#                     atom_by_symb[symb].append(atom_idx)
#             if len(non_carbon) == 2:
#                 rdmol_edit.RemoveBond(*non_carbon)
#             if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
#                 rdmol_edit.RemoveBond(*atom_by_symb['O'])
#                 rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
#                     rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
#                 )
#                 rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
#                     rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
#                 )
#     rdmol = rdmol_edit.GetMol()

#     for atom in rdmol.GetAtoms():
#         if atom.GetFormalCharge() > 0:
#             atom.SetFormalCharge(0)

#     return rdmol



# def reconstruct(coords, atoms_num):
#     mol, atoms = make_obmol(coords, atoms_num)
#     fixup(atoms, mol)
#     connect_the_dots(mol, atoms, None, covalent_factor=1.3)
#     fixup(atoms, mol)
#     mol.AddPolarHydrogens()
#     mol.PerceiveBondOrders()
#     fixup(atoms, mol)
#     for (i, a) in enumerate(atoms):
#         ob.OBAtomAssignTypicalImplicitHydrogens(a)
#     fixup(atoms, mol)

#     mol.AddHydrogens()
#     fixup(atoms, mol)
#     # make rings all aromatic if majority of carbons are aromatic
#     for ring in ob.OBMolRingIter(mol):
#         if 5 <= ring.Size() <= 6:
#             carbon_cnt = 0
#             aromatic_ccnt = 0
#             for ai in ring._path:
#                 a = mol.GetAtom(ai)
#                 if a.GetAtomicNum() == 6:
#                     carbon_cnt += 1
#                     if a.IsAromatic():
#                         aromatic_ccnt += 1
#             if aromatic_ccnt >= carbon_cnt / 2 and aromatic_ccnt != ring.Size():
#                 # set all ring atoms to be aromatic
#                 for ai in ring._path:
#                     a = mol.GetAtom(ai)
#                     a.SetAromatic(True)
#     # bonds must be marked aromatic for smiles to match
#     for bond in ob.OBMolBondIter(mol):
#         a1 = bond.GetBeginAtom()
#         a2 = bond.GetEndAtom()
#         if a1.IsAromatic() and a2.IsAromatic():
#             bond.SetAromatic(True)
#     mol.PerceiveBondOrders()
#     rd_mol = convert_ob_mol_to_rd_mol(mol)
#     try:
#         # Post-processing
#         rd_mol = postprocess_rd_mol_1(rd_mol)
#         rd_mol = postprocess_rd_mol_2(rd_mol)
#     except:
#         raise MolReconsError()
#     smi = Chem.MolToSmiles(rd_mol)
#     mol = Chem.MolFromSmiles(smi)
#     return mol 

# def visualize_generated_mol(mol, show_surface=False, opacity=0.5):
#     view = py3Dmol.view()

#     mblock = Chem.MolToMolBlock(mol)
#     view.addModel(mblock, 'mol')
#     view.setStyle({'model': -1}, {'stick': {}, 'sphere': {'radius': 0.35}})
#     if show_surface:
#         view.addSurface(py3Dmol.SAS, {'opacity': opacity}, {'model': -1})

#     view.zoomTo()
#     return view


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

