# Modified from https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/master/run_audio.py

import argparse
import torch
from almeval.datasets import build_dataset
from almeval.models import build_model
from tqdm import tqdm
from almeval.utils import *
from loguru import logger
import sys
import os
import warnings
import time
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def setup_logging(rank, model, data, work_dir):
    # Create log directory
    log_dir = os.path.join(work_dir, model, data, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'rank{rank}.log')
    
    # Remove default console output
    logger.remove()
    
    # Add file output
    log_file_handle = open(log_file, 'w', encoding='utf-8')
    logger.add(log_file_handle, 
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
               level="INFO")
    # Print rank=0 output to console as well
    if rank == 0:
        logger.add(sys.stdout, 
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                level="INFO")
    
    return log_file


def merge_one_dataset(args, dataset, result_file, eval_file):
    model_data_dir = osp.join(args.work_dir, args.model, dataset.DATASET_NAME)
    os.makedirs(model_data_dir, exist_ok=True)

    if args.reeval:
        perf = dataset.evaluate(result_file, method=args.eval_method)
        with open(eval_file, 'w') as f:
            json.dump(perf, f, indent=4)
        return

    tmp_files = [osp.join(model_data_dir, f'{rank}_{args.world_size}_{dataset.DATASET_NAME}.pkl') for rank in range(args.world_size)]

    # Merge if all tmp_files exist
    if all(osp.exists(tmpfile) for tmpfile in tmp_files):
        data_all = {}
        for tmpfile in tmp_files:
            data_all.update(load(tmpfile))
        raw_data = dataset.data
        for x in raw_data:
            idx = int(x['index'])
            if idx not in data_all:
                logger.warning(f'index {idx} not found in data_all, details: {x}')
                x['prediction'] = 'null'
                x['real_prompt'] = ''
                continue
            x['prediction'] = str(data_all[idx]['prediction'])
            x['real_prompt'] = str(data_all[idx]['prompt'])

        dump(raw_data, result_file)

        for tmpfile in tmp_files:
            os.remove(tmpfile)

        logger.info(f'model {args.model}, data {dataset.DATASET_NAME}, all {args.world_size} result merged to {result_file}.')

    if args.skip_eval:
        logger.info(f'skip eval for {dataset.DATASET_NAME}')
        return
    perf = dataset.evaluate(result_file, method=args.eval_method)
    with open(eval_file, 'w') as f:
        json.dump(perf, f, indent=4)
    logger.info(f'model {args.model}, data {dataset.DATASET_NAME} evaluated.')


def do_reeval(dataset_name, result_file='auto', method='default', subset=None):
    datasets = []
    for dataset_name in args.data:
        d = build_dataset(dataset_name, subset)
        if isinstance(d, list):
            datasets.extend(d)
        else:
            datasets.append(d)

    for dataset in datasets:
        if result_file == 'auto':
            benchmark_dir = osp.join(args.work_dir, args.model, dataset.DATASET_NAME)
            pred_result_file = osp.join(benchmark_dir, f'{args.model}_{dataset.DATASET_NAME}.jsonl')
        else:
            pred_result_file = result_file
        logger.info(f'evaluating {pred_result_file} with method {method}')
        perf = dataset.evaluate(pred_result_file, method=method)
        with open(pred_result_file.replace('.jsonl', f'_{method}_performance.json'), 'w') as f:
            json.dump(perf, f, indent=4)


def process_dataset(args, dataset, model):
    # Assign different subsets to each process
    model_data_dir = osp.join(args.work_dir, args.model, dataset.DATASET_NAME)
    result_file = osp.join(model_data_dir, f'{args.model}_{dataset.DATASET_NAME}.jsonl')
    eval_file = osp.join(model_data_dir, f'{args.model}_{dataset.DATASET_NAME}_{args.eval_method}_performance.json')
    os.makedirs(model_data_dir, exist_ok=True)
    rank = int(args.rank)
    
    if os.path.exists(result_file) and not args.force_reinfer:
        if args.reeval or not os.path.exists(eval_file):
            if rank==0:
                logger.info(f'file {result_file} exists, reevaluating...')
                merge_one_dataset(args, dataset, result_file, eval_file)
            return
        else:
            # Exit if not reeval and result file exists
            return 

    else:
        if args.debug:
            dataset.set_demo_mode()
        sample_indices = [i for i in range(len(dataset))]

        # Distribute data to each rank
        world_size = int(args.world_size)
        rank = int(args.rank)
        sample_indices_sub = sample_indices[rank::world_size]

        tmpl = osp.join(model_data_dir, f'{rank}_{args.world_size}_{dataset.DATASET_NAME}.pkl')
        out_file = tmpl.format(rank)
        res = load(out_file) if osp.exists(out_file) else {}
        
        processed_samples = 0
        for i in tqdm(sample_indices_sub, disable=args.rank != 0):
            if args.subset is not None:
                if dataset[i]["meta"]["subset"] != args.subset:
                    continue
            
            msg = dataset[i]
            idx = int(msg['index'])
            if not args.force_reinfer:
                if idx in res:
                    continue

            if processed_samples==0:
                logger.info(f'Msg example: {msg}')

            real_prompt, response = model(msg)
            torch.cuda.empty_cache()
            if response is None:
                continue
            
            # we need response and prompt, because model may change prompt
            res[idx] = {
                'prompt': real_prompt,
                'prediction': response,
            }
            processed_samples += 1
            if processed_samples % 20 == 0:
                dump(res, out_file)
        dump(res, out_file)

        # Write a file to indicate this rank is done
        with open(osp.join(model_data_dir, f'{rank}_{args.world_size}_{dataset.DATASET_NAME}.done'), 'w') as f:
            f.write('done')

        # Rank 0 needs to wait for other ranks to finish, then merge results
        time_elapsed = 0
        if rank == 0:
            while True:
                all_success_files = [osp.join(model_data_dir, f'{rank}_{args.world_size}_{dataset.DATASET_NAME}.done') 
                                for rank in range(args.world_size)]
                if len(all_success_files) == args.world_size and \
                    all(osp.exists(success_file) for success_file in all_success_files):
                    # Delete all done files
                    for success_file in all_success_files:
                        os.remove(success_file)
                    break
                else:
                    time.sleep(10)
                    time_elapsed += 10
                    logger.info(f'waiting for other ranks to finish, time elapsed: {time_elapsed}s')
            merge_one_dataset(args, dataset, result_file, eval_file)

def main(args):
    datasets = []
    for dataset_name in args.data:
        d = build_dataset(dataset_name, args.subset)
        if isinstance(d, list):
            datasets.extend(d)
        else:
            datasets.append(d)
    model = build_model(args.model, model_path=args.model_path)
    logger.info(f"Datasets: {datasets}")
    for dataset in datasets:
        setup_logging(args.rank, args.model, dataset.DATASET_NAME, args.work_dir)
        logger.info(f"Running {args.model} on dataset: {dataset.DATASET_NAME}")
        process_dataset(args, dataset, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='List of dataset names')
    parser.add_argument('--subset', type=str, default=None, help='subset name')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--model-path', type=str, help='Model Path')
    parser.add_argument('--work-dir', type=str, default='./eval_result', help='Working directory')
    parser.add_argument('--rank', type=int, default=0, help='Current GPU rank')
    parser.add_argument('--world-size', type=int, default=1, help='Total number of GPUs')
    parser.add_argument('--reeval', action='store_true', help='Whether to re-evaluate')
    parser.add_argument('--eval-file', type=str, default='auto', help='Evaluation file path, used when run_eval_only=True')
    parser.add_argument('--debug', action='store_true', help='Debug mode, only run 10 samples per dataset')
    parser.add_argument('--eval-method', type=str, default='default', help='Evaluation method')
    parser.add_argument('--force-reinfer', action='store_true', help='Whether to force re-inference')
    parser.add_argument('--skip-eval', action='store_true', help='Whether to skip evaluation')
    args = parser.parse_args()
    args.data = args.data.split(',')

    if args.reeval:
        do_reeval(args.data, args.eval_file, args.eval_method, args.subset)
    else:
        main(args)
