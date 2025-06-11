#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import distributed_utils, options
from unicore import tasks
from tqdm import tqdm
import numpy as np
from utils import Metrics, draw_valid_smiles, set_random_seed

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


def main(args):

    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    task = tasks.setup_task(args)
    model = task.build_model(args)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(args)

    # Build loss
    loss = task.build_loss(args)
    loss.eval()

    sample = None
    with torch.no_grad():
        sample_size = args.sample_size
        log_outputs = []
        condition_list = []
        seed = args.seed
        
        pbar = tqdm(total=sample_size, desc="Generating samples", unit="sample")
        if args.infer_mode == 'eval':
            while len(log_outputs) < sample_size:
                set_random_seed(seed)
                _, _, log_output = task.valid_step(sample, model, loss, test=True)
                    
                log_outputs.extend(log_output['output_smi'])                
                if hasattr(args, 'condition'):
                    condition_list.extend(log_output['condition'].tolist())
                pbar.update(len(log_output['output_smi']))

                if len(log_outputs) >= sample_size:
                    log_outputs = log_outputs[:sample_size]
                    if hasattr(args, 'condition'):
                        condition_list = condition_list[:sample_size]
                    break
                seed += 1
            pbar.close()


            results = Metrics.evaluate(log_outputs, condition_list, args)

            logger.info("valid_smiles are as follows:\n {}".format(results['valid_smiles']))

            logger.info("validity: mean={:.2f}%, sd={:.2f}%, vals={}%".format(np.mean(results['valid_ratio']), np.std(results['valid_ratio']), results['valid_ratio']))

            logger.info("uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}%".format(np.mean(results['unique_ratio']), np.std(results['unique_ratio']), results['unique_ratio']))

            logger.info("novelty: mean={:.2f}%, sd={:.2f}%, vals={}%".format(np.mean(results['novel_ratio']), np.std(results['novel_ratio']), results['novel_ratio']))
                        
            fname = (args.path).split("/")[-2].replace('-','_')
            save_path = os.path.join(args.results_path, fname + ".out.pkl")
            with open(save_path, "wb") as f:
                    pickle.dump(results, f)
                    
        elif args.infer_mode == 'gen':
            unique_valid_smi = set()
            print('gen')
            while len(unique_valid_smi) < sample_size:
                set_random_seed(seed)
                _, _, log_output = task.valid_step(sample, model, loss, test=True)
                results = Metrics.evaluate(log_output['output_smi'], condition_list, args)
                del log_output
                unique_valid_smi.update(results['unique_smiles'])

                pbar.update(len(unique_valid_smi))
                print('len(unique_valid_smi):', len(unique_valid_smi))
                if len(unique_valid_smi) >= sample_size:
                    unique_valid_smi = list(unique_valid_smi)[:sample_size]
                    break
                seed += 1

            pbar.close()
            fname = (args.path).split("/")[-2].replace('-','_')
            save_path = os.path.join(args.results_path, fname + ".unique_out.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(unique_valid_smi, f)

        else:
            raise NotImplementedError


        logger.info("\n")
        logger.info("Done inference! ")



    return None

def cli_main():
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
