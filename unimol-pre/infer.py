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
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


def calculate_r2_scores_for_each_column(target_array, prediction_array):

    r2_scores = {}
    for column_index in range(target_array.shape[1]):
        r2_score_for_column = r2_score(target_array[:, column_index], prediction_array[:, column_index])
        r2_scores[f"Column {column_index}"] = r2_score_for_column
    return r2_scores

def calculate_mae_scores_for_each_column(target_array, pred_array):
    mae_scores = {}
    for column in range(target_array.shape[1]):
        mae_scores[column] = mean_absolute_error(target_array[:, column], pred_array[:, column])
    return mae_scores

def main(args):

    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if args.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

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

    for subset in args.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        # fname = (args.path).split("/")[-2]
        # save_path = os.path.join(args.results_path,  subset + ".out.pkl")
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        log_outputs = []
        pred_list = []
        target_list = []    
        name_list = []    
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if len(sample) == 0:
                continue
            _, _, log_output = task.valid_step(sample, model, loss, test=True)
            progress.log({}, step=i)
            log_outputs.append(log_output)
            pred_list.extend(log_output["predict"].cpu().detach().numpy())
            target_list.extend(log_output['target'].cpu().detach().numpy()) 
            name_list.extend(log_output['name'])


        pred_list = np.array(pred_list)
        target_list = np.array(target_list)
        name_list = np.array(name_list)

        if not np.all(target_list == 0):
            r2_scores = calculate_r2_scores_for_each_column(target_list, pred_list)

            for column, r2_score in r2_scores.items():
                print(f'R2 for {column}: {r2_score}')

            mae_scores = calculate_mae_scores_for_each_column(target_list, pred_list)

            for column, mae_score in mae_scores.items():
                print(f'MAE for column {column}: {mae_score}')        

        if args.task_name == "0702_plqy" or args.task_name == 'infer_plqy':
            column_names = ['PLQY']
        else:
            column_names = ['S1 (eV)', 'Sigma_opt (1e-16)', 'Sigma_ele (1e-16)', 'kr (10E8)']

        # Create DataFrame from pred_list with the specified column names
        pred_df = pd.DataFrame(pred_list, columns=column_names)

        # Insert name_list as the first column in the DataFrame
        pred_df.insert(0, 'Name', name_list)

        # Display the DataFrame
        print(pred_df.head())

        pred_df.to_csv(os.path.join(args.results_path, 'pred.csv'), index=False)


        logger.info("Done inference! ")

    return None


def cli_main():
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
