'''
    Experiment data loading for PFRL logfile
'''

import pandas as pd
import os
import argparse
import re
from plotting.comparison_plotting import plot_comparison_learning_curves

def get_summary_data(csv_path, column_keys=('steps', 'episodes', 'mean')):
    assert len(column_keys) == 2, f'Must specify X and Y axis only, got {column_keys}'

    df = pd.read_csv(csv_path, sep='\t')
    data = dict()
    for key in column_keys:
        try:
            d = df[key]
            data[key] = d
        except:
            print(f'{key} is not in {csv_path}')
            data[key] = None
    return data[column_keys[0]].to_numpy(), data[column_keys[1]].to_numpy()


def gather_evaluator_csv_files_from_base_dir(base_dir=None, filter_fn=lambda path: True):
    """
    Here the base_dir is assumed to be ~/acme, and under the base_dir are the dirs
    named with `acme_id`, under which has `logs/evaluator/logs.csv`.

    Args:
        base_dir (_type_): acme results dir, if None use the default acme setting
        filter_fn (boolean function): 
    Returns:
        dict exp_dir -> csv_path
    """

    DEFAULT_BASE_DIR = './results'
    SCORE_CSV = 'scores.txt'
    if base_dir is None:
        base_dir = os.path.expanduser(DEFAULT_BASE_DIR)

    # find all subdir and get the acme_id of all
    id_to_csv = {}
    for exp_dir in os.listdir(base_dir):
        if filter_fn(exp_dir):
            csv_path = os.path.join(base_dir, exp_dir, SCORE_CSV)
            id_to_csv[exp_dir] = csv_path
    return id_to_csv




def filter_from_substring(substring):
    def filter_fn(exp_dir):
        return re.match(substring, exp_dir) is not None
    return filter_fn


def plot_pfrl_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--group-by', nargs='+', type=str, default=['lr', ])
    parser.add_argument('--x-axis', type=str, default='steps')
    parser.add_argument('--y-axis', type=str, default='mean')
    parser.add_argument('--xlabel', type=str, default='steps')
    parser.add_argument('--ylabel', type=str, default='success rate')
    parser.add_argument('--regex', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--window-size', type=int, default=10)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    if args.regex is not None:
        filter_fn = filter_from_substring(args.regex)


    plot_comparison_learning_curves(
        base_dir=args.base_dir,
        group_keys=args.group_by,
        summary_fn=lambda csv: get_summary_data(csv, column_keys=[args.x_axis, args.y_axis]),
        csv_path_fn=gather_evaluator_csv_files_from_base_dir,
        smoothen=args.window_size,
        save_path=args.save_path,
        ylabel=args.ylabel,
        xlabel=args.xlabel,
        show=args.show,
    )

if __name__ == '__main__':
   plot_pfrl_experiments()