# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import torch
import numpy as np
from unicore.data import BaseWrapperDataset

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


class KeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.dataset[idx][self.key]

    
class ListDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return list(self.dataset[idx])

class RawListDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.dataset[idx]))

class ConditionDataset(BaseWrapperDataset):
    def __init__(self, dataset, condition, choice, method=None, num_bins=None):
        super().__init__(dataset)
        self.dataset = dataset
        self.condition = condition
        self.method = method
        self.choice = OLED_CONDITION_REGISTER[choice]

        if method == 'bin':
            self.min_value = OLED_BIN_CONDITION_REGISTER[choice]['min']
            self.max_value = OLED_BIN_CONDITION_REGISTER[choice]['max']
            self.num_bins = num_bins
            self.bin_edges = np.linspace(self.min_value, self.max_value, self.num_bins)
            self.bin_labels = np.arange(self.num_bins)

            self.method_functions = {
                'bin': self._process_bin
            }
        else:
            self.method_functions = {
                'val': self._process_val
            }

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.method_functions[self.method](idx)

    def _process_val(self, idx):
        return self.dataset[idx][self.condition][self.choice]

    def _process_bin(self, idx):
        value = self.dataset[idx][self.condition][self.choice]
        return self.bin_labels[np.digitize(value, self.bin_edges) - 1]


class AtomsDataset(BaseWrapperDataset):
    def __init__(self, dataset, dictionary, chargedict):
        self.dataset = dataset
        self.dictionary = dictionary
        self.chargedict = chargedict
        self.indices = {v:k for k,v in self.dictionary.indices.items()}

    def __len__(self):
        return len(self.dataset)

    def convert_to_string(self, data_array, chargedict, indices):
        result = []
        for num in data_array:
            if num == 0:
                result.append(indices[0])
            else:
                result.append(chargedict[num])
        return result

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.convert_to_string(self.dataset[idx], self.chargedict, self.indices)
    
class RemoveHydrogenDataset(BaseWrapperDataset):
    def __init__(self, dataset, atoms, coordinates, remove_hydrogen=False, remove_polar_hydrogen=False, dictionary=None, set_frag=False, set_residue=False):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen
        self.dictionary = dictionary
        self.set_frag = set_frag
        self.set_residue = set_residue
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):

        if  self.dataset[index][self.atoms] is not None:
            dd = self.dataset[index].copy()
            atoms = dd[self.atoms]
            coordinates = dd[self.coordinates]
            if self.set_residue:
                residue = dd['residue']

            # if self.remove_hydrogen:
            #     mask_hydrogen = [x != 1 for x in atoms]
            #     atoms = atoms[mask_hydrogen]
            #     coordinates = coordinates[mask_hydrogen]
            #     if self.set_residue:
            #         residue = residue[mask_hydrogen]


            if self.remove_hydrogen:
                # Convert lists to numpy arrays
                atoms_array = np.array(atoms)
                if self.set_residue:
                    residue_array = np.array(residue)

                # Create a boolean mask where each True value corresponds to non-hydrogen atoms
                mask_hydrogen = atoms_array != 1

                # Apply the mask to filter out the hydrogen atoms
                atoms_filtered = atoms_array[mask_hydrogen]
                coordinates_filtered = coordinates[0][mask_hydrogen]
                if self.set_residue:
                    residue_filtered = residue_array[mask_hydrogen]

                # Convert numpy arrays back to lists
                atoms = atoms_filtered.tolist()
                coordinates = [coordinates_filtered]
                if self.set_residue:
                    residue = residue_filtered.tolist()


            if not self.remove_hydrogen and self.remove_polar_hydrogen:
                end_idx = 0 
                for i, atom in enumerate(atoms[::-1]):
                    if atom != 1:
                        break
                    else:
                        end_idx = i + 1
                if end_idx != 0:
                    atoms = atoms[:-end_idx]
                    coordinates = coordinates[:-end_idx]
                    if self.set_residue:
                        residue = residue[:-end_idx]
            dd[self.atoms] = atoms
            dd[self.coordinates] = coordinates#.astype(np.float32)
            if self.set_residue:
                dd['residue'] = residue
            if self.set_frag:
                dd['frag_atom_id'] = atoms
            dd['atom_num'] = len(atoms)
        else:
            dd = self.dataset[index].copy()
            dd['atom_num'] = 0
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)