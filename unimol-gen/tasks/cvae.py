# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    FromNumpyDataset,
    RawLabelDataset,
)
from unimol.data import (
    KeyDataset,
    ListDataset,
    ConditionDataset,
    ConformerSampleDataset,
    RightPadDatasetCoord,
    DistanceDataset,
    EdgeTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RawListDataset,
    PairChargeDataset,
    PrependAndAppendPairChargeDataset,
    RightPadDataset2D,
)

from unicore.tasks import UnicoreTask, register_task
from unicore import checkpoint_utils

logger = logging.getLogger(__name__)

STRATEGY_DICT = {
    'TFM': 'teacher_forcing'
}


@register_task("cvae")
class UniMolFinetuneCvaeTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--finetune-encoder-model",
            type=str,
            default=None,
            help="pretrain encoder model path",
        )        
        parser.add_argument(
            "--condition_method",
            type=str,
            default='val',
            choices=['val', 'bin'],
            help="the way to use conditon",
        ),
        parser.add_argument(
            "--num_bins",
            type=int,
            default=None,
            help="bin number",
        ),    


    def __init__(self, args, dictionary, smi_dictionary):
        super().__init__(args)
        self.teacher_forcing = True if args.decoder == 'TFM' else False
        self.dictionary = dictionary
        self.smi_dictionary = smi_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pad_idx = self.dictionary.pad()
        self.smi_pad_idx = self.smi_dictionary.pad()
        self.condition_method = args.condition_method
        self.num_bins = args.num_bins
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        if args.encoder == 'unimol-oled':
            dictionary = Dictionary.load(os.path.join(args.data, args.task_name, "oled.dict.txt"))
        elif args.encoder == 'unimol-opv':
            dictionary = Dictionary.load(os.path.join(args.data, args.task_name, "opv.dict.txt"))
        else:
            dictionary = Dictionary.load(os.path.join(args.data, args.task_name, "dict.txt"))
        smi_dictionary = Dictionary.load(os.path.join(args.data, args.task_name, "smi.dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        logger.info("smi dictionary: {} types".format(len(smi_dictionary)))
        return cls(args, dictionary, smi_dictionary)



    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        smi_dataset = KeyDataset(dataset, "smi")
        smi_tokens_dataset = KeyDataset(dataset, "smi_tokens")
        condition_info_dataset = ConditionDataset(dataset, "energy", "homo", self.condition_method, self.num_bins) ## TOOD: Add more choices, refactor
        dataset = ConformerSampleDataset(dataset, self.args.seed, "atoms", "coordinates")
        dataset = CroppingDataset(
            dataset, 
            self.seed, 
            "atoms", 
            "coordinates", 
            self.args.max_atoms
        )
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")
        def PrependAndAppend(dataset, pre_token=None, app_token=None):
            if pre_token is not None:
                dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)
        
        if self.args.encoder == 'unimol-oled':
            charge_dataset = KeyDataset(dataset, "atoms_charge")
            charge_dataset = RawListDataset(charge_dataset)
            pair_charge_dataset = PairChargeDataset(charge_dataset)
            pair_charge_dataset = PrependAndAppendPairChargeDataset(pair_charge_dataset, 0.0)
        else:
            charge_dataset = None
            pair_charge_dataset = None

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), app_token=self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        
        smi_tokens_dataset = ListDataset(smi_tokens_dataset)
        tgt_dataset = TokenizeDataset(
            smi_tokens_dataset, self.smi_dictionary, max_seq_len=self.args.max_seq_len
        )
        if self.teacher_forcing:
            tgt_dataset = PrependTokenDataset(tgt_dataset, self.dictionary.bos())

        tgt_dataset = AppendTokenDataset(tgt_dataset, self.dictionary.eos())

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "src_pair_charge": RightPadDataset2D(
                        pair_charge_dataset,
                        pad_idx=0,
                    ) if pair_charge_dataset is not None else None,
                    "condition_info": RawLabelDataset(
                        condition_info_dataset
                    ),
                    "tgt_tokens": RightPadDataset(
                        tgt_dataset,
                        pad_idx=self.smi_dictionary.pad(),
                    ),
                },
                "smi":smi_dataset,
            },
        )
        if split in ["train", "train.small"]:
            nest_dataset = EpochShuffleDataset(nest_dataset, len(nest_dataset), self.args.seed)
        self.datasets[split] = nest_dataset


    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        if args.finetune_encoder_model is not None:
                print("load pretrain model weight from...", args.finetune_encoder_model)
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.finetune_encoder_model,
                )
                model.encoder.load_state_dict(state["model"], strict=False)
        return model