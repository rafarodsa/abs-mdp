import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_scores(base_dir, subdir, window_size, label=None):
    file_path = os.path.join(base_dir, subdir, 'scores.txt')
    if not os.path.exists(file_path):
        print(f"{file_path} not found!")
        return None

    df = pd.read_csv(file_path, sep='\t')
    # Apply smoothing filter
    df['mean_smooth'] = df['mean'].rolling(window=window_size).mean()
    df['stdev_smooth'] = df['stdev'].rolling(window=window_size).mean()

    plt.fill_between(df['steps'], df['mean_smooth'] - df['stdev_smooth'], df['mean_smooth'] + df['stdev_smooth'], alpha=0.2)
    plt.plot(df['steps'], df['mean_smooth'], label=label if label else subdir)

def main(base_dir, window_size, baseline_dir=None):
    sns.set(style="whitegrid")  # Set Seaborn style
    plt.figure(figsize=(10, 6))
    
    # Plot for ground_env
    plot_scores(base_dir, 'ground_env', window_size)

    # Plot for sim_env
    plot_scores(base_dir, 'sim_env', window_size)

    # Plot for baseline if provided
    if baseline_dir:
        plot_scores(baseline_dir, '', window_size, label='baseline')

    plt.xlabel('Steps')
    plt.ylabel('Smoothed Mean Score')
    plt.legend()
    plt.title('Smoothed Mean Scores vs. Steps with Confidence Intervals')
    plt.tight_layout()
    plt.savefig('scores.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot smoothed mean scores vs. steps from scores.txt files in subdirectories and compare with a baseline.')
    parser.add_argument('--base_dir', type=str, help='Path to the base directory containing ground_env and sim_env subdirectories.')
    parser.add_argument('--window_size', type=int, default=100, help='Window size for the smoothing filter.')
    parser.add_argument('--baseline_dir', type=str, default=None, help='Path to the directory containing the baseline scores.txt file.')
    args = parser.parse_args()

    main(args.base_dir, args.window_size, args.baseline_dir)
