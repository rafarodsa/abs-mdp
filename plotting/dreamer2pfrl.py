'''
    preprocess log files
'''

import pandas as pd
from dreamerv2.common.plot import load_jsonl
from pathlib import Path
import os
import argparse
import tqdm

def dreamerv2pfrl(df):

    df = df[['step', 'eval_return']].dropna()
    df = df.groupby('step')
    eval_data = df.mean()
    eval_data['stdev'] = df.std()
    eval_data = eval_data.rename(columns={'step':'steps', 'eval_return': 'mean'})
    return eval_data.rename_axis('steps', axis=0)


def dreamerv3pfrl(df):
    df = df[['step', 'eval_episode/score']].dropna()
    df['eval_episode/score'] = (df['eval_episode/score'] > 0).astype(float)
    df = df.groupby('step')
    eval_data = df.mean()
    eval_data['stdev'] = df.std()
    eval_data = eval_data.rename(columns={'step':'steps', 'eval_episode/score': 'mean'})
    return eval_data.rename_axis('steps', axis=0)
                                 

def walk_first_level_dirs(base_dir):
    base_path = Path(base_dir)
    
    # Check if the base directory exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"The directory '{base_dir}' does not exist or is not a valid directory.")
        return
    
    # Get all the immediate subdirectories
    subdirectories = [item for item in base_path.iterdir() if item.is_dir()]

    return subdirectories

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', nargs='+', type=str, default=None)
    parser.add_argument('--evaluator-dirs', nargs='+', type=str, default=['ground'])
    parser.add_argument('--x-offset', nargs='+', type=float, default=None)
    parser.add_argument('--log-file', type=str, default='scores.txt')
    parser.add_argument('--v3', action='store_true', default=False)
    parser.add_argument('--subdir', type=str, default=None)
    args, unknown = parser.parse_known_args()

    if args.subdir:
        subdir = Path(args.subdir)
        eval_data = load_jsonl(subdir / 'metrics.jsonl')
        eval_data = dreamerv2pfrl(eval_data) if not args.v3 else dreamerv3pfrl(eval_data)
        os.makedirs(subdir / args.evaluator_dirs[0], exist_ok=True)
        eval_data.to_csv(subdir / args.evaluator_dirs[0] / args.log_file, sep='\t')
        print(f'Log parsed and saved at {subdir / args.evaluator_dirs[0] / args.log_file}')
    
    else:
        for subdir in walk_first_level_dirs(args.base_dir[0]):
            eval_data = load_jsonl(subdir / 'metrics.jsonl')
            eval_data = dreamerv2pfrl(eval_data) if not args.v3 else dreamerv3pfrl(eval_data)
            os.makedirs(subdir / args.evaluator_dirs[0], exist_ok=True)
            eval_data.to_csv(subdir / args.evaluator_dirs[0] / args.log_file, sep='\t')
            print(f'Log parsed and saved at {subdir / args.evaluator_dirs[0] / args.log_file}')

