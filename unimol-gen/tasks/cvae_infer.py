# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from unicore.data import Dictionary
from unicore.tasks import UnicoreTask, register_task
from unicore import checkpoint_utils


logger = logging.getLogger(__name__)


STRATEGY_DICT = {
    'TFM': 'teacher_forcing'
}

def parse_condition(value):
    try:
        return float(value)
    except ValueError:
        return value

@register_task("cvae_infer")
class UniMolFinetuneCvaeInferTask(UnicoreTask):
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
            "--sample_size",
            type=int,
            default=50,
            help="sample size for inferencing",
        )
        parser.add_argument(
            "--condition_method",
            type=str,
            default='val',
            choices=['val', 'bin'],
            help="the way to use conditon",
        ),        
        parser.add_argument(
            "--condition",
            type=parse_condition,
            default="123",
            help="condtion value or path",
        )
        parser.add_argument(
            "--max_len",
            type=int,
            default=512,
            help="max length of tokens",
        )
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
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size
        self.condition = args.condition
        self.num_bins = args.num_bins
        self.condition_method = args.condition_method
        self.max_len = args.max_len
        
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
    
    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        if args.path is not None:
            print("load pretrain model weight from...", args.path)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.path,
                )
            keys = {k: v for k, v in state['model'].items() if  k.startswith('mol_cvae_model')}
            model.state_dict().update(keys)
            model.load_state_dict(keys, strict=False)
            model = model.mol_cvae_model
        return model