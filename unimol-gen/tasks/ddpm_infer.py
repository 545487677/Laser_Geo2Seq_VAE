# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import numpy as np
from unicore.data import Dictionary
from unicore.tasks import UnicoreTask, register_task
from unicore import checkpoint_utils


logger = logging.getLogger(__name__)

STRATEGY_DICT = {
    'TFM': 'teacher_forcing'
}


@register_task("ddpm_infer")
class UniMolFinetuneddpmInferTask(UnicoreTask):
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
            "--max_len",
            type=int,
            default=512,
            help="max length of tokens",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=1.,
            help="logvar scale",
        ),
        parser.add_argument(
            "--infer_mode",
            type=str,
            default="eval",
            choices=["eval","gen"],
            help="infer mode",
        ),        

    def __init__(self, args, dictionary, prior_dis):
        super().__init__(args)
        self.teacher_forcing = True if args.decoder == 'TFM' else False
        self.dictionary = dictionary
        self.prior_dis = prior_dis
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pad_idx = self.dictionary.pad()
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size
        self.max_len = args.max_len
        self.scale = args.scale
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        if args.encoder == 'unimol-oled':
            dictionary = Dictionary.load(os.path.join(args.data, args.task_name, "oled.dict.txt"))
        else:
            dictionary = Dictionary.load(os.path.join(args.data, args.task_name, "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        # prior dirstribution of atom lengths
        prior_dis = None
        if args.encoder == 'unimol-oled':
            prior_dis = np.load(os.path.join(args.data, args.task_name, 'atom_length.npy'), allow_pickle=True).tolist()
            logger.info("Atom prior distribution: {} types".format(len(prior_dis)))
        return cls(args, dictionary, prior_dis)
    
    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        if args.path is not None:
            print("load pretrain model weight from...", args.path)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.path,
                )
            model.load_state_dict(state['model'], strict=False)
        return model