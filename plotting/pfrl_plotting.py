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
from itertools import product
from collections import defaultdict

import seaborn as sns


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

def gather_evaluator_csv_files_from_base_dirs(base_dir=[], file_name='scores.txt', evaluator_dir='ground', filter_fn=lambda path: True):
    """
    Here the base_dir is assumed to be ~/acme, and under the base_dir are the dirs
    named with `acme_id`, under which has `logs/evaluator/logs.csv`.

    Args:
        base_dir (_type_): acme results dir, if None use the default acme setting
        filter_fn (boolean function): 
    Returns:
        dict exp_dir -> csv_path
    """

    id_to_csv = {}
    for basedir in base_dir:
        id_to_csv.update(gather_evaluator_csv_files_from_base_dir(basedir, file_name, evaluator_dir, filter_fn))
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

def plot_curve_groups(curve_groups, curve_group_names, x_offsets=None):
    for i, (group, group_name, offset) in enumerate(zip(curve_groups, curve_group_names, x_offsets)):
        for curve, name, xoffset in zip(group, group_name, offset):
            identifier = name if name is not None else f"group_{i}"
            label = f"{identifier}_{curve['label']}"
            plt.plot(curve["xs"] + xoffset, curve["mean"], linewidth=2, label=label, alpha=0.9)
            plt.fill_between(curve["xs"] + xoffset, curve["top"], curve["bottom"], alpha=0.2)

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

def group_dirs_by_pattern(basedir, depth, identifiers, filter_func=None):
    """
    Group directories based on the pattern {identifier}_n or [+-]identifier, 
    where identifier is one of the possible identifiers.
    The identifier is matched as a substring of the directory name.
    Only directories that pass the filter_func will be included.
    """
    # Compile patterns for each identifier
    patterns_n = [re.compile(f"{re.escape(identifier)}_([\d]+(\.[\d]+)?)") for identifier in identifiers]
    patterns_bool = [re.compile(f"[+-]{re.escape(identifier)}") for identifier in identifiers]
    
    dir_dict = {}
    found_identifiers = set()
    for current_dir, subdirs, _ in os.walk(basedir):
        current_depth = current_dir.count(os.sep) - basedir.count(os.sep)
        # print(f'current_depth: {current_depth}, current_dir: {current_dir}, subdirs: {subdirs}, basedir: {basedir}')
        if current_depth < depth:
            for subdir in subdirs:
                full_path = os.path.join(current_dir, subdir)
                if len(identifiers) > 0:
                    for identifier, pattern_n, pattern_bool in zip(identifiers, patterns_n, patterns_bool):
                        match_n = pattern_n.search(subdir)
                        match_bool = pattern_bool.search(subdir)
                        if match_n:
                            found_identifiers.add(identifier)
                            key = f"{match_n.group(0)}"
                            _add_to_result(dir_dict, key, full_path, current_depth, depth, filter_func)
                        if match_bool:
                            found_identifiers.add(identifier)
                            key = f"{match_bool.group(0)}"
                            _add_to_result(dir_dict, key, full_path, current_depth, depth, filter_func)

            # remove found identifiers from the list of identifiers
            identifiers = list(set(identifiers) - found_identifiers)

        else:
            break
        

    # expand the paths to the last level allowed
    for k, v in dir_dict.items():
        if len(v) > 0:
            _expanded_paths = []
            for path in v:
                current_depth = path.count(os.sep) - basedir.count(os.sep)
                _expanded_paths.extend(get_subdirs(path, depth - current_depth, filter_func))
            dir_dict[k] = _expanded_paths

    return dir_dict, found_identifiers


def get_subdirs(base_path, depth, filter_fn=None):
    if depth < 0:
        return []

    # subdirs = [base_path] if (filter_fn is None or filter_fn(base_path)) else []
    subdirs = []
    start_level = base_path.rstrip(os.path.sep).count(os.path.sep)
    
    for root, dirs, _ in os.walk(base_path):
        if filter_fn is not None:
            dirs[:] = [d for d in dirs if filter_fn(os.path.join(root, d))]
        
        current_level = root.count(os.path.sep) - start_level
        if current_level == depth-1:
            for directory in dirs:
                subdirs.append(os.path.join(root, directory))
    return subdirs

def _add_to_result(dir_dict, key, full_path, current_depth, depth, filter_func):
    # Check for immediate subdirectories if we're at depth-1
    if current_depth == depth - 1:
        dplus1_subdirs = [os.path.join(full_path, d1_subdir) for d1_subdir in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d1_subdir))]
        # Apply the filter function, if provided
        if filter_func:
            dplus1_subdirs = [path for path in dplus1_subdirs if filter_func(path)]
        dir_dict.setdefault(key, []).extend(dplus1_subdirs)
    elif not filter_func or filter_func(full_path):
        dir_dict.setdefault(key, []).append(full_path)


def join_dirs(dir_dict, identifiers):
    
    identifier_values = defaultdict(set)
    for identifier in identifiers:
        for k in dir_dict.keys():
            if identifier in k:
                identifier_values[identifier].add(k)
    
    new_ids = product(*identifier_values.values())
    new_dir_dict = defaultdict(set)
    for new_id in new_ids:
        key  = "__".join(new_id)
        paths = [v for k, v in dir_dict.items() if k in new_id]
        if len(paths) > 0:
            print(paths)
            new_dir_dict[key] = set(paths[0]).intersection(*paths)
    return new_dir_dict

def multiple_level_exp(args):
    filter_fn = None
    if args.regex is not None:
        filter_fn = filter_from_substring(args.regex)

    paths, found_ids = group_dirs_by_pattern(args.base_dirs[0], args.depth, args.group_by, filter_func=filter_fn)
    paths = join_dirs(paths, found_ids)
    curves_groups = []
    if args.curve_names is None:
        args.curve_names = [None] * len(args.base_dirs)
    else:
        assert len(args.curve_names) == len(args.base_dirs)


    if args.x_offset is None:
        args.x_offset = [0] * len(args.base_dirs)
    elif len(args.x_offset) <= len(args.base_dirs):
        args.x_offset = args.x_offset + [0] * (len(args.base_dirs) - len(args.x_offset)) # pad with zeros
        
    curve_names = []
    offsets = []
    groupbys = set()

    group_keys = set(args.group_by) - set(found_ids)
    if len(group_keys) == 0:
        group_keys = ['']
    for i, (group_name, base_dir) in enumerate(paths.items()):
        for evaluator_dir in args.evaluator_dirs:
            csv_fn = partial(gather_evaluator_csv_files_from_base_dirs, evaluator_dir=evaluator_dir, file_name=args.log_file, filter_fn=lambda path: True)
            
            curves = generate_learning_curves(
                base_dir=base_dir,
                group_keys=group_keys,
                summary_fn=lambda csv: get_summary_data(csv, column_keys=[args.x_axis, args.y_axis]),
                csv_path_fn=csv_fn,
                smoothen=args.window_size,
                save_path=args.save_path,
                ylabel=args.ylabel,
                xlabel=args.xlabel,
                show=args.show,
            )
            for c in curves:
                groupbys.add(c['label'])
                c['label'] = f'{group_name}_{evaluator_dir}_{c["label"]}'
                curve_names.append(c['label'])
            curves_groups.append(curves)


    plot_curve_groups(curves_groups, curve_names, x_offsets=[0]*len(curves_groups))
    
    plt.legend()

    if args.save_path is not None:
        os.makedirs(os.path.split(args.save_path)[0], exist_ok=True)
        plt.savefig(args.save_path)

    if args.save_curves:
        path = os.path.split(args.save_path)[0]
        save_curve_groups_to_csv(curves_groups, args.curve_names, os.path.join(path, 'curves.csv'))

def compare_pfrl_experiments(args):
    curves_groups, curve_names, offsets, groupbys = get_curves(args)

    plot_curve_groups(curves_groups, curve_names, x_offsets=offsets)
    
    plt.legend()

    if args.save_path is not None:
        path = os.path.split(args.save_path)[0]
        os.makedirs(path, exist_ok=True)
        plt.savefig(args.save_path)

    if args.save_curves:
        path = os.path.split(args.save_path)[0]
        save_curve_groups_to_csv(curves_groups, args.curve_names, os.path.join(path, 'curves.csv'))


    # print final scores ordered from highest to lowest
    print_best_scores(curves_groups)

def get_curves(args):
    filter_fn = lambda path: True
    if args.regex is not None:
        filter_fn = filter_from_substring(args.regex)
    
    curves_groups = []
    if args.curve_names is None:
        args.curve_names = [None] * len(args.base_dirs)
    else:
        assert len(args.curve_names) == len(args.base_dirs)


    if args.x_offset is None:
        args.x_offset = [0] * len(args.base_dirs)
    elif len(args.x_offset) <= len(args.base_dirs):
        args.x_offset = args.x_offset + [0] * (len(args.base_dirs) - len(args.x_offset)) # pad with zeros
    
    if len(args.window_size) < len(args.base_dirs):
        args.window_size = args.window_size + [1] * (len(args.base_dirs) - len(args.window_size))
        

    curve_names = []
    offsets = []
    x_offsets = []
    names = []
    groupbys = set()
    for i, base_dir in enumerate(args.base_dirs):
        offsets = []
        curve_names = []
        print(f'Window size {args.window_size[i]}')
        for evaluator_dir in args.evaluator_dirs:
            csv_fn = partial(gather_evaluator_csv_files_from_base_dir, evaluator_dir=evaluator_dir, file_name=args.log_file, filter_fn=filter_fn)
            curves = generate_learning_curves(
                base_dir=base_dir,
                group_keys=args.group_by,
                summary_fn=lambda csv: get_summary_data(csv, column_keys=[args.x_axis, args.y_axis]),
                csv_path_fn=csv_fn,
                smoothen=args.window_size[i],
                save_path=args.save_path,
                ylabel=args.ylabel,
                xlabel=args.xlabel,
                show=args.show,
                truncate_max_frames=args.max_x_value-args.x_offset[i],
                min_len_curve=args.min_len_curve
            )
            for c in curves:
                groupbys.add(c['label'])
                c['label'] = f'{evaluator_dir}_{c["label"]}'
                curve_names.append(args.curve_names[i])
                offsets.append(args.x_offset[i])
            curves_groups.append(curves)
            names.append(curve_names)
            x_offsets.append(offsets)


    return curves_groups,names,x_offsets, groupbys

def print_best_scores(curves_groups):
    final_scores = []
    for curves in curves_groups:
        for curve in curves:
            final_scores.append((curve['label'], curve['mean'][-1]))
    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
    # print final scores in table format
    print("Final scores:")
    print("-------------")
    print("Experiment\t\t\t\t\tScore")  
    for score in final_scores:
        print(f"{score[0]}\t\t{score[1]}")


def plot_subplots(args):
    curves, curve_names, x_offsets, groupbys = get_curves(args)
    subplot_groups = list(groupbys)
    subplot_groups.sort()
    n_cols = args.n_cols
    layout = (len(subplot_groups) // n_cols + len(subplot_groups) % n_cols, n_cols)
    print(f'Subplot layout: {layout}')
    print(f'Subplot groups: {subplot_groups}')
    
    # sns.set_context("paper")
    sns.set_style("whitegrid")

    # Set color palette
    palette = sns.color_palette('husl', len(curves))

    fig, axs = plt.subplots(*layout, sharex=True, sharey=True, figsize=(24,12))


    for j, (group, group_name, offset) in enumerate(zip(curves, curve_names, x_offsets)):
        for curve, xoffset, name in zip(group, offset, group_name):
            identifier = name if name is not None else f"group_{j}"

            regex_pattern = '(.*{}_|.*{}$)'

            subplot_idx = list(filter(lambda x: re.match(regex_pattern.format(x[-1], x[-1]), curve['label']) is not None, enumerate(subplot_groups))) # TODO this fails when a tag 
            if len(subplot_idx) == 0:
                print(f"Curve {curve['label']} does not match any of the subplots")
                continue
            plt.subplot(*layout, subplot_idx[0][0] + 1)
            label = f"{identifier}_{curve['label']}"
            color = palette[j]  # Assigning color based on curve index
            curve['label'] = label
            plt.plot(curve["xs"] + xoffset, curve["mean"], color=color, linewidth=2, label=label)
            plt.fill_between(curve["xs"] + xoffset, curve["top"], curve["bottom"], color=color, alpha=0.2)
            plt.title(subplot_idx[0][1], fontsize=8)
    
    if args.save_path is not None:
        print(f'Saving figure to {args.save_path}')
        os.makedirs(os.path.split(args.save_path)[0], exist_ok=True)
        plt.savefig(args.save_path)

    print_best_scores(curves)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str, choices=['compare', 'multiple_level', 'subplot'], default='compare')
    parser.add_argument('--base-dirs', nargs='+', type=str, default=None)
    parser.add_argument('--curve-names', nargs='+', type=str, default=None)
    parser.add_argument('--group-by', nargs='+', type=str, default=['',])
    parser.add_argument('--evaluator-dirs', nargs='+', type=str, default=['ground'])
    parser.add_argument('--max-x-value', type=float, default=-1)
    parser.add_argument('--x-offset', nargs='+', type=float, default=None)
    parser.add_argument('--log-file', type=str, default='scores.txt')
    parser.add_argument('--x-axis', type=str, default='steps')
    parser.add_argument('--y-axis', type=str, default='mean')
    parser.add_argument('--xlabel', type=str, default='steps')
    parser.add_argument('--ylabel', type=str, default='success rate')
    parser.add_argument('--regex', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--save-curves', action='store_true')
    parser.add_argument('--window-size', nargs='+', type=int, default=[])
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--n-cols', type=int, default=4)
    parser.add_argument('--min-len-curve', type=int, default=-1)
    args, unknown = parser.parse_known_args()


    if args.script == 'compare':
        compare_pfrl_experiments(args)
    elif args.script == 'multiple_level':
        multiple_level_exp(args)
    elif args.script == 'subplot':
        plot_subplots(args)
    else:
        raise NotImplementedError(f"Unknown script {args.script}")

if __name__ == '__main__':
    parse_args()
