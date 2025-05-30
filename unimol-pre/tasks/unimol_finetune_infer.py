# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
    RawArrayDataset,
    FromNumpyDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    data_utils,
)

from unimol.data.tta_dataset import TTADataset
from unicore.tasks import UnicoreTask, register_task
from unicore import checkpoint_utils

logger = logging.getLogger(__name__)

task_metainfo = {
    "esol": {
        "mean": -3.0501019503546094,
        "std": 2.096441210089345,
        "target_name": "logSolubility",
    },
    "freesolv": {
        "mean": -3.8030062305295944,
        "std": 3.8478201171088138,
        "target_name": "freesolv",
    },
    "lipo": {"mean": 2.186336, "std": 1.203004, "target_name": "lipo"},
    "qm7dft": {
        "mean": -1544.8360893118609,
        "std": 222.8902092792289,
        "target_name": "u0_atom",
    },
    "qm8dft": {
        "mean": [
            0.22008500524052105,
            0.24892658759891675,
            0.02289283121913152,
            0.043164444107224746,
            0.21669716560818883,
            0.24225989336408812,
            0.020287111373358993,
            0.03312609817084387,
            0.21681478862847584,
            0.24463634931699113,
            0.02345177178004201,
            0.03730141834205415,
        ],
        "std": [
            0.043832862248693226,
            0.03452326954549232,
            0.053401140662012285,
            0.0730556474716259,
            0.04788020599385645,
            0.040309670766319,
            0.05117163534626215,
            0.06030064428723054,
            0.04458294838213221,
            0.03597696243350195,
            0.05786865052149905,
            0.06692733477994665,
        ],
        "target_name": [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "E1-PBE0",
            "E2-PBE0",
            "f1-PBE0",
            "f2-PBE0",
            "E1-CAM",
            "E2-CAM",
            "f1-CAM",
            "f2-CAM",
        ],
    },
    "qm9dft": {
        "mean": [-0.23997669940621352, 0.011123767412331285, 0.2511003712141015],
        "std": [0.02213143402267657, 0.046936069870866196, 0.04751888787058615],
        "target_name": ["homo", "lumo", "gap"],
    },
    "opv": {
        "mean": [-5.588, -2.3508],
        "std": [0.5151, 0.7422],
        "target_name": ["homo", "lumo"],        
    },
    "opv_rdkit_dft": {
        "mean": [-5.588, -2.3508],
        "std": [0.5151, 0.7422],
        "target_name": ["homo", "lumo"],        
    },
    "cmr_rdkit": {
        "mean": [-5.2196, -2.6620],
        "std": [0.3069, 0.4254],
        "target_name": ["homo", "lumo"],        
    },    
    "TADF": {
        "mean": [501.3328],
        "std": [85.6838],
        "target_name": ["ew"],        
    },    
    "opv_rdkit_gen": {
        # without adding opv_500_2_006
        # "mean": [-5.6457, -2.3304],
        # "std": [0.4258, 0.7460],
        # add opv_500_2_006
        "mean": [-5.6729, -2.4299],
        "std": [0.4009, 0.7553],        
        "target_name": ["homo", "lumo"],              
    },      
    # infer
    "moleuclar_optimization_res_select_test": {
        "mean": [-5.6457, -2.3304],
        "std": [0.4258, 0.7460],
        "target_name": ["homo", "lumo"],        
    },
    "opv_more_cond": {
        "mean": [-5.6477, -2.3296, 0.3613, 0.5947],
        "std": [0.4243, 0.7460, 0.2843, 0.4477],
        "target_name": ["homo", "lumo", "delta_homo", "delta_lumo"],        
    },   
    "opv_2_0006": {
        # without adding opv_500_2_006
        # "mean": [-5.6457, -2.3304],
        # "std": [0.4258, 0.7460],
        # add opv_500_2_006
        "mean": [-5.6729, -2.4299],
        "std": [0.4009, 0.7553],        
        "target_name": ["homo", "lumo"],    
    },      
    "opt_2_0006_200_0311": {
        # without adding opv_500_2_006
        # "mean": [-5.6457, -2.3304],
        # "std": [0.4258, 0.7460],
        # add opv_500_2_006
        "mean": [-5.6729, -2.4299],
        "std": [0.4009, 0.7553],        
        "target_name": ["homo", "lumo"],         
    },      
    "opt_dft": {
        "mean": [-5.6948, -2.4484],
        "std": [0.3950, 0.7466],      
        "target_name": ["homo", "lumo"],              
    },     
    "general_dft": {
        "mean": [-5.6948, -2.4484],
        "std": [0.3950, 0.7466], 
        "target_name": ["homo", "lumo"],              
    },       
    "finetune_dft": {
        "mean": [-5.6948, -2.4484],
        "std": [0.3950, 0.7466],
        "target_name": ["homo", "lumo"],        
    },   
    'laser_data': {'mean': [2.3066546218487396, 1.3781203678151261, -2.886733447647059, 195.37968361966531, 2.890754421075412], 'std': [0.2927775952256696, 0.8827228436461901, 2.541324877950363, 1377.9220866400378, 2.1002335226811937], 'target_name': ['S1', 'Sigma_opt (1e-16)', 'Sigma_ele (1e-16)', 'kic (10E8)', 'kr (10E8)']},
    'infer_laser_data': {'mean': [2.3066546218487396, 1.3781203678151261, -2.886733447647059, 195.37968361966531, 2.890754421075412], 'std': [0.2927775952256696, 0.8827228436461901, 2.541324877950363, 1377.9220866400378, 2.1002335226811937], 'target_name': ['S1', 'Sigma_opt (1e-16)', 'Sigma_ele (1e-16)', 'kic (10E8)', 'kr (10E8)']},
    '0702_exclude_kic': {'mean': [2.361733522727273, 1.7487617190113636, -2.3649937849318183, 3.721583082696591], 'std': [0.26623040335211995, 1.0871789882142329, 2.5751666409146403, 2.4000200179936804], 'target_name': ['S1 (eV)', 'Sigma_opt (1e-16)', 'Sigma_ele (1e-16)', 'kr (10E8)']},
    'infer_results': {'mean': [2.361733522727273, 1.7487617190113636, -2.3649937849318183, 3.721583082696591], 'std': [0.26623040335211995, 1.0871789882142329, 2.5751666409146403, 2.4000200179936804], 'target_name': ['S1 (eV)', 'Sigma_opt (1e-16)', 'Sigma_ele (1e-16)', 'kr (10E8)']},
    '0702_plqy':{'mean': [0.6090506972422611], 'std': [0.257722037860837], 'target_name': ['PLQY']},
    'infer_plqy':{'mean': [0.6090506972422611], 'std': [0.257722037860837], 'target_name': ['PLQY']},
}


@register_task("mol_finetune_infer")
class UniMolFinetuneInferTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--finetune-from-model",
            default="",
            help="pretrain weight",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen ",
        )
        parser.add_argument(
            "--todft",
            action='store_true',
            help="",
        )
        parser.add_argument(
            "--tolmdb",
            action='store_true',
            help="",
        )
        parser.add_argument(
            "--strict_filter",
            action='store_true',
            help="",
        )
        parser.add_argument(
            "--gap_type",
            type=str, 
            choices=['max', 'min'],
            help="",
        )


    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
        if self.args.task_name in task_metainfo:
            # for regression task, pre-compute mean and std
            self.mean = task_metainfo[self.args.task_name]["mean"]
            self.std = task_metainfo[self.args.task_name]["std"]

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.task_name, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        
        # ori dataset 
        tgt_dataset = KeyDataset(dataset, "target")
        tgt_dataset = RawLabelDataset(tgt_dataset)
        name_dataset = KeyDataset(dataset, "name")
        dataset = ConformerSampleDataset(dataset, self.seed, 'atoms', 'coordinates')
        dataset = RemoveHydrogenDataset(dataset, 'atoms', 'coordinates', True, True)
        dataset = CroppingDataset(dataset, self.seed, 'atoms', 'coordinates', self.args.max_atoms)

  
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "name": RawArrayDataset(name_dataset),


            },
        )
        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        # model.register_classification_head(
        #     self.args.classification_head_name,
        #     num_classes=self.args.num_classes,
        # )
        if args.finetune_from_model is not None:
                print("load pretrain model weight from...", args.finetune_from_model)
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.finetune_from_model,
                )
                model.load_state_dict(state["model"], strict=False)
        return model
