'''
    Experiment data loading for PFRL logfile
'''

import pandas as pd
import os
import argparse
import re
from plotting.comparison_plotting import plot_comparison_learning_curves, generate_learning_curves
from matplotlib import pyplot as plt

from functools import partial

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


def gather_evaluator_csv_files_from_base_dir(base_dir=None, file_name='scores.txt', evaluator_dir='ground', filter_fn=lambda path: True):
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
    if base_dir is None:
        base_dir = os.path.expanduser(DEFAULT_BASE_DIR)

    # find all subdir and get the id of all
    id_to_csv = {}
    for exp_dir in os.listdir(base_dir):
        if filter_fn(exp_dir):
            csv_path = os.path.join(base_dir, exp_dir, evaluator_dir, file_name)
            if os.path.exists(csv_path):
                id_to_csv[exp_dir] = csv_path
    return id_to_csv


def filter_from_substring(substring):
    def filter_fn(exp_dir):
        return re.match(substring, exp_dir) is not None
    return filter_fn

def plot_curve(curve_data):
    plt.plot(curve_data["xs"], curve_data["mean"], linewidth=curve_data["linewidth"], label=curve_data["label"], alpha=0.9)
    plt.fill_between(curve_data["xs"], curve_data["top"], curve_data["bottom"], alpha=0.2)

def curves_to_csv(curves, filename):
    data = {}
    for curve in curves:
        label_prefix = f"{curve['label']}"
        data[f"{label_prefix}_xs"] = curve['xs']
        data[f"{label_prefix}_mean"] = curve['mean']
        data[f"{label_prefix}_top"] = curve['top']
        data[f"{label_prefix}_bottom"] = curve['bottom']

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def csv_to_curves(filename):
    df = pd.read_csv(filename)
    curves = []

    labels = set(col.split('_xs')[0] for col in df.columns if '_xs' in col)
    for label in labels:
        curve = {
            'xs': df[f"{label}_xs"].dropna().tolist(),
            'mean': df[f"{label}_mean"].dropna().tolist(),
            'top': df[f"{label}_top"].dropna().tolist(),
            'bottom': df[f"{label}_bottom"].dropna().tolist(),
            'label': label,
        }
        curves.append(curve)

    return curves

def plot_curve_groups(curve_groups, curve_group_names):
    for i, (group, group_name) in enumerate(zip(curve_groups, curve_group_names)):
        identifier = group_name if group_name is not None else f"group_{i}"
        for curve in group:
            label = f"{identifier}_{curve['label']}"
            plt.plot(curve["xs"], curve["mean"], linewidth=2, label=label, alpha=0.9)
            plt.fill_between(curve["xs"], curve["top"], curve["bottom"], alpha=0.2)

def save_curve_groups_to_csv(curve_groups, curve_group_names, filename):
    df = pd.DataFrame()

    for idx, (group, group_name) in enumerate(zip(curve_groups, curve_group_names)):
        identifier = group_name if group_name is not None else f"group{idx}"
        for curve in group:
            label = f"{identifier}__{curve['label']}"
            df[f"{label}_xs"] = pd.Series(curve["xs"])
            df[f"{label}_mean"] = pd.Series(curve["mean"])
            df[f"{label}_top"] = pd.Series(curve["top"])
            df[f"{label}_bottom"] = pd.Series(curve["bottom"])

    df.to_csv(filename, index=False)


def load_curve_groups_from_csv(filename):
    df = pd.read_csv(filename)
    curve_groups = {}
    for column in df.columns:
        # Splitting by the double underscore '__' to separate the group_name and curve label
        group_name, curve_info = column.split("__", 1)
        curve_property = curve_info.rsplit("_", 1)[-1]  # Getting 'xs', 'mean', 'top', or 'bottom'
        curve_label = curve_info.rsplit("_", 1)[0]     # Removing the trailing '_xs', '_mean', etc.
        
        if group_name not in curve_groups:
            curve_groups[group_name] = {}

        if curve_label not in curve_groups[group_name]:
            curve_groups[group_name][curve_label] = {}
        
        curve_groups[group_name][curve_label][curve_property] = df[column].dropna().tolist()

    # Convert the nested dict structure to a list of lists of dicts format
    result = []
    for group_name, curves_dict in curve_groups.items():
        group = []
        for curve_label, curve_data in curves_dict.items():
            group.append({
                'label': curve_label,
                'xs': curve_data['xs'],
                'mean': curve_data['mean'],
                'top': curve_data['top'],
                'bottom': curve_data['bottom'],
            })
        result.append(group)

    return result


def compare_pfrl_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dirs', nargs='+', type=str, default=None)
    parser.add_argument('--curve-names', nargs='+', type=str, default=None)
    parser.add_argument('--group-by', nargs='+', type=str, default=['',])
    parser.add_argument('--evaluator-dirs', nargs='+', type=str, default='ground')
    parser.add_argument('--log-file', type=str, default='scores.txt')
    parser.add_argument('--x-axis', type=str, default='steps')
    parser.add_argument('--y-axis', type=str, default='mean')
    parser.add_argument('--xlabel', type=str, default='steps')
    parser.add_argument('--ylabel', type=str, default='success rate')
    parser.add_argument('--regex', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--save-curves', action='store_true')
    parser.add_argument('--window-size', type=int, default=10)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()


    filter_fn = lambda path: True
    if args.regex is not None:
        filter_fn = filter_from_substring(args.regex)
    
    curves_groups = []
    if args.curve_names is None:
        args.curve_names = [None] * len(args.base_dirs)
    else:
        assert len(args.curve_names) == len(args.base_dirs)
        
    curve_names = []
    for i, base_dir in enumerate(args.base_dirs):
        for evaluator_dir in args.evaluator_dirs:
            csv_fn = partial(gather_evaluator_csv_files_from_base_dir, evaluator_dir=evaluator_dir, file_name=args.log_file, filter_fn=filter_fn)
            curves = generate_learning_curves(
                base_dir=base_dir,
                group_keys=args.group_by,
                summary_fn=lambda csv: get_summary_data(csv, column_keys=[args.x_axis, args.y_axis]),
                csv_path_fn=csv_fn,
                smoothen=args.window_size,
                save_path=args.save_path,
                ylabel=args.ylabel,
                xlabel=args.xlabel,
                show=args.show,
            )
            for c in curves:
                c['label'] = f'{evaluator_dir}_{c["label"]}'
                curve_names.append(args.curve_names[i])
            curves_groups.append(curves)

    

    plot_curve_groups(curves_groups, curve_names)
    
    plt.legend()

    if args.save_path is not None:
        plt.savefig(args.save_path)

    if args.save_curves:
        path = os.path.split(args.save_path)[0]
        save_curve_groups_to_csv(curves_groups, args.curve_names, os.path.join(path, 'curves.csv'))


    

def plot_pfrl_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--group-by', nargs='+', type=str, default=['', ])
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
#    plot_pfrl_experiments()
    compare_pfrl_experiments()