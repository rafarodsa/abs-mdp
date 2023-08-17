'''
    Experiment data loading for PFRL logfile
'''

import pandas as pd
import os


def get_summary_data(csv_path, column_keys=('steps', 'episodes', 'mean')):

    df = pd.read_csv(csv_path, sep='\t')
    data = dict()
    for key in column_keys:
        try:
            d = df[key]
            data[key] = d
        except:
            print(f'{key} is not in {csv_path}')
            data[key] = None
    return data    


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


if __name__ == '__main__':
    csvs = gather_evaluator_csv_files_from_base_dir()
    for exp_id, csv_path in csvs.items():
        print(f'Loading data for {exp_id}')
        data = get_summary_data(csv_path=csv_path)
        print(data)