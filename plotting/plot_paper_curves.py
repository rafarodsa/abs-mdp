'''
    Experiment data loading for PFRL logfile
'''

import pandas as pd
import os
import argparse
import re
from plotting.comparison_plotting import plot_comparison_learning_curves, generate_learning_curves
from plotting.pfrl_plotting import get_curves, print_best_scores, save_curve_groups_to_csv 
from matplotlib import pyplot as plt

from functools import partial
from itertools import product
from collections import defaultdict
from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns



def format_ticks(value, _):
    if value >= 1e6:
        value = value / 1e6
        if value - int(value) < 0.1:
            return f'{int(value)}M'
        return f'{value:.1f}M'
    elif value >= 1e3:
        return f'{int(value / 1e3)}K'
    else:
        return str(int(value))


def plot_curve_groups(curve_groups, curve_group_names, x_offsets=None, add_baseline=True, max_x_value=1e30):

    for i, (group, group_name, offset) in enumerate(zip(curve_groups, curve_group_names, x_offsets)):
        for curve, name, xoffset in zip(group, group_name, offset):
            identifier = name if name is not None else f"group_{i}"
            # label = f"{identifier}_{curve['label']}"
            xs = curve["xs"] + xoffset
            _xs = xs[xs < max_x_value]
            ys = curve["mean"][xs < max_x_value]
            plt.plot(_xs, ys, linewidth=3, label=identifier, alpha=1)
            # plt.xscale('log')
            _top = curve["top"][xs <  max_x_value]
            _bottom = curve["bottom"][xs <  max_x_value]
            plt.fill_between(_xs, _top, _bottom, alpha=0.2)
            if xoffset > 0:
                plt.axvspan(0, xoffset, color='gray', alpha=0.2, linewidth=0)
    
    if add_baseline:
        curve = curve_groups[1][0]
        y_value = curve["mean"][-1]
        x_value = curve["xs"][-1] + x_offsets[-1][0]
        plt.gca().axhline(xmin=0., xmax=1., y=y_value, color='k', linewidth=2, alpha=0.7, linestyle='--')
        plt.gca().axvline(x=x_value, color='k', linewidth=2, alpha=0.7, linestyle='--')

    # plt.legend()
    # sns.move_legend(plt.gca(), "lower center", ncol=4, bbox_to_anchor=(0.5, -0.25), frameon=False)
    # plt.subplots_adjust(bottom=0.3) 
        

def compare_pfrl_experiments(args):
    curves_groups, curve_names, offsets, groupbys = get_curves(args)

    # plt.figure(figsize=(11, 7))
    # plt.figure(figsize=(8, 9))
    plt.figure(figsize=(10, 8))
    sns.set_context("talk", font_scale=1.2)
    sns.set_style("white")
    sns.color_palette()
    
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))
    plot_curve_groups(curves_groups, curve_names, x_offsets=offsets, max_x_value=args.max_x_value)
    
    
    if args.xlabel is not None:
        plt.xlabel(args.xlabel)
    if args.ylabel is not None:
        plt.ylabel(args.ylabel)
    if args.title is not None:
        plt.title(args.title)
    plt.tight_layout()
    if args.save_path is not None:
        path, fname = os.path.split(args.save_path)
        os.makedirs(path, exist_ok=True)
        plt.savefig(args.save_path, dpi=300)
        plt.savefig(os.path.join(path, f'{fname}_transparent.png'), dpi=300, transparent=True)
        print(f'Saving figure to {args.save_path}')

    if args.save_curves:
        path = os.path.split(args.save_path)[0]
        save_curve_groups_to_csv(curves_groups, args.curve_names, os.path.join(path, 'curves.csv'))


    # print final scores ordered from highest to lowest
    print_best_scores(curves_groups)


def plot_sequential_goals(args):
    curves, curve_names, x_offsets, groupbys = get_curves(args)
    subplot_groups = list(groupbys)
    subplot_groups.sort()

    fig = plt.figure(figsize=(8,9))
    sns.set_context("talk", font_scale=2)
    sns.set_style("white")

    # Set color palette
    palette = sns.color_palette()
    
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))

    handles = []
    names = []

    for j, (group, group_name, offset) in enumerate(zip(curves, curve_names, x_offsets)):
        xs = []
        ys = []
        top, bottom = [], []
        labels = []
        last_x = offset[0]
        # last_x = 0
        for curve, xoffset, name in zip(group, offset, group_name):
            identifier = name if name is not None else f"group_{j}"

            xs.append(curve["xs"] + last_x)
            ys.append(curve["mean"])
            top.append(curve["top"])
            bottom.append(curve["bottom"])
            labels.append(curve['label'])
            last_x = xs[-1][-1]

        # for curve, xoffset, name in zip(group, offset, group_name):
        identifier = name if name is not None else f"group_{j}"
        
        label=identifier
        if j == 1:
            ax = plt.gca().twiny()
        elif j == 0:
            ax = plt.gca()
            ax.set_xlabel(args.xlabel)
        else:
            raise ValueError("Only two experiments supported")

        
            
        ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
        color = palette[j]  # Assigning color based on curve index
        for k, (_xs, _ys, _top, _bottom, _label) in enumerate(zip(xs, ys, top, bottom, labels)):
            ax.plot(_xs, _ys, color=color, linewidth=3, label=label)
            ax.fill_between(_xs, _top, _bottom, color=color, alpha=0.2)
            if j == 0 and k < len(xs) - 1:
                ax.axvline(x=_xs[-1], color='k', linewidth=2, alpha=0.9, linestyle='--')
        print(labels)
        handles.append(ax.plot([], [], color=color, linewidth=3, label=label)[0])
        names.append(label)
        ax.set_ylabel(args.ylabel)
    # ax.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.subplots_adjust(bottom=0.15)
    # fig.legend(handles, names, loc='lower center', ncol=2, fancybox=False, shadow=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()

    if args.save_path is not None:
        print(f'Saving figure to {args.save_path}.png')
        path, fname = os.path.split(args.save_path)
        os.makedirs(path, exist_ok=True)
        plt.savefig(args.save_path, dpi=300)
        plt.savefig(os.path.join(path, f'{fname}_transparent.png'), dpi=300, transparent=True)

def plot_subplots(args):
    curves, curve_names, x_offsets, groupbys = get_curves(args)
    subplot_groups = list(groupbys)
    subplot_groups.sort()
    n_cols = args.n_cols
    layout = (len(subplot_groups) // n_cols + len(subplot_groups) % n_cols, n_cols)
    print(f'Subplot layout: {layout}')
    print(f'Subplot groups: {subplot_groups}')
    
    sns.set_context("talk", font_scale=1.2)
    sns.set_style("white")

    # Set color palette
    palette = sns.color_palette()

    fig, axes = plt.subplots(*layout, sharex=True, sharey=True, figsize=(12, 16))
    for j, (group, group_name, offset) in enumerate(zip(curves, curve_names, x_offsets)):
        for curve, xoffset, name in zip(group, offset, group_name):
            identifier = name if name is not None else f"group_{j}"

            regex_pattern = '(.*{}_|.*{}$)'

            subplot_idx = list(filter(lambda x: re.match(regex_pattern.format(x[-1], x[-1]), curve['label']) is not None, enumerate(subplot_groups)))
            if len(subplot_idx) == 0:
                print(f"Curve {curve['label']} does not match any of the subplots")
                continue
            ax = axes[subplot_idx[0][0] // n_cols, subplot_idx[0][0] % n_cols]
            if subplot_idx[0][0] % n_cols == 0:
                ax.set_ylabel(args.ylabel)
            if subplot_idx[0][0] // n_cols == layout[0] - 1:
                ax.set_xlabel(args.xlabel)
            ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
            # label = f"{identifier}_{curve['label']}"
            label=identifier
            color = palette[j]  # Assigning color based on curve index
            curve['label'] = label
            ax.plot(curve["xs"] + xoffset, curve["mean"], color=color, linewidth=2, label=label)
            ax.fill_between(curve["xs"] + xoffset, curve["top"], curve["bottom"], color=color, alpha=0.2)
            if xoffset > 0:
                ax.axvspan(0, xoffset, color='gray', alpha=0.2, linewidth=0)
            ax.set_title(subplot_idx[0][1])

    h, l = fig.get_axes()[0].get_legend_handles_labels()
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1)
    legend = fig.legend(h, l, loc='lower center', ncol=3, fancybox=True, shadow=False)
    legend.get_frame().set_alpha(0.7)
    
    if args.save_path is not None:
        
        print(f'Saving figure to {args.save_path}.png')
        path, fname = os.path.split(args.save_path)
        os.makedirs(path, exist_ok=True)
        plt.savefig(args.save_path, dpi=300)
        plt.savefig(os.path.join(path, f'{fname}_transparent.png'), dpi=300, transparent=True)


def _get_x_value(xs, ys, value):
    x_value_a = None
    for x, y in zip(xs, ys):
        if y >= value:
            x_value_a = x
            break
    return x_value_a

def plot_sample_barplot(args):
    assert len(args.base_dirs) == 2
    curves, curve_names, x_offsets, groupbys = get_curves(args)
    subplot_groups = list(groupbys)
    subplot_groups.sort()
    n_cols = args.n_cols
    layout = (len(subplot_groups) // n_cols + len(subplot_groups) % n_cols, n_cols)
    print(f'Subplot layout: {layout}')
    print(f'Subplot groups: {subplot_groups}')

    sns.set_context("talk", font_scale=2)
    sns.set_style("white")
    sns.color_palette('Set2')

    baseline_group = 0  # Assuming the baseline is the first group
    # import ipdb; ipdb.set_trace()
    # Get the final value of the baseline curve for each curve in the baseline group
    baseline_final_values = [c['mean'][-1] for c in curves[baseline_group]]
    baseline_x_values = [c['xs'][-1] for c in curves[baseline_group]]

    # Create lists to store the x-values of other curves when they reach the same value
    x_values_of_other_curves = [[] for _ in range(len(curves)-1)]

    # Loop through the other groups and calculate x-values when they reach the same value as baseline
    for curve_idx, curves_d in enumerate(curves[1:]):
        for curve_data, value in zip(curves_d, baseline_final_values):
            x_value = _get_x_value(curve_data['xs'], curve_data['mean'], value)
            x_values_of_other_curves[curve_idx].append(x_value)

    print(f'Baseline final values: {baseline_final_values}')
    print(f'Baseline x values: {baseline_x_values}')
    print(f'X values of other curves: {x_values_of_other_curves}')

    print(x_offsets)
    avg_x_values = np.mean(x_values_of_other_curves, axis=-1) + x_offsets[1][0]
    std_x_values = np.std(x_values_of_other_curves, axis=-1)
    
    avg_baseline_x_value = np.mean(baseline_x_values)
    std_baseline_x_value = np.std(baseline_x_values)

    print(f'Average x values: {avg_x_values}')
    print(f'Std x values: {std_x_values}')
    print(curve_names)
    print(f'Average baseline x value: {avg_baseline_x_value}')
    print(f'Std baseline x value: {std_baseline_x_value}')

    # Plot the difference in a barplot
    plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))
    categories = np.array(curve_names[0][0:1] + curve_names[1][0:1] )
    print(categories)
    y = np.concatenate([[avg_baseline_x_value], avg_x_values])
    y_std = np.concatenate([[std_baseline_x_value], std_x_values])
    print(y)

    sns.barplot(x=categories, y=y, yerr=y_std, palette='Set2')
    plt.ylabel('env steps')
    # plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels if needed

    if args.save_path is not None:
        print(f'Saving figure to {args.save_path}.png')
        path, fname = os.path.split(args.save_path)
        os.makedirs(path, exist_ok=True)
        plt.savefig(args.save_path, dpi=300)
        plt.savefig(os.path.join(path, f'{fname}_transparent.png'), dpi=300, transparent=True)

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str, choices=['compare', 'sequential', 'subplot', 'barplot'], default='compare')
    parser.add_argument('--base-dirs', nargs='+', type=str, default=None)
    parser.add_argument('--curve-names', nargs='+', type=str, default=None)
    parser.add_argument('--group-by', nargs='+', type=str, default=['',])
    parser.add_argument('--evaluator-dirs', nargs='+', type=str, default=['ground'])
    parser.add_argument('--max-x-value', type=float, default=-1)
    parser.add_argument('--x-offset', nargs='+', type=float, default=None)
    parser.add_argument('--log-file', type=str, default='scores.txt')
    parser.add_argument('--x-axis', type=str, default='steps')
    parser.add_argument('--y-axis', type=str, default='mean')
    parser.add_argument('--xlabel', type=str, default=None)
    parser.add_argument('--ylabel', type=str, default=None)
    parser.add_argument('--regex', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--save-curves', action='store_true')
    parser.add_argument('--window-size', nargs='+', type=int, default=[])
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--n-cols', type=int, default=4)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--min-len-curve', type=int, default=-1)
    args, unknown = parser.parse_known_args()


    if args.script == 'compare':
        compare_pfrl_experiments(args)
    elif args.script == 'sequential':
        plot_sequential_goals(args)
    elif args.script == 'subplot':
        plot_subplots(args)
    elif args.script == 'barplot':
        plot_sample_barplot(args)
    else:
        raise NotImplementedError(f"Unknown script {args.script}")

if __name__ == '__main__':
    parse_args()
