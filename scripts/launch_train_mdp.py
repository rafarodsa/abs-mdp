import argparse
import json, os
import re 

def tasklist(task_list_str):
    tasks = task_list_str.split(',')
    t = []
    for task in tasks:
        if '-' in task:
            min_t, max_t = task.split('-')
            _t = list(range(int(min_t), int(max_t)+1))
        else:
            _t = [int(task)]
        t = t + _t
    return t

def contains(string, substrings):
    r = True
    for s in substrings:
        r = r and (s in string)
    return r

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--filter', nargs='*')
    parser.add_argument('--dryrun', action='store_true', default=False)
    parser.add_argument('--jobname', type=str)
    parser.add_argument('--command', type=str)
    args = parser.parse_args()

    substrings = [s.strip(' ,') for s in args.filter]
    dirpath, dirs, files = next(os.walk(args.dir)) # first level.
    dirs = filter(lambda s: contains(s.strip('/'), substrings), dirs)
    config_path_format = '{}/{}/phi_train/csv_logs/infomax-pb/version_0/hparams.yaml'
    config_paths = [config_path_format.format(args.dir.strip('/'), tag) for tag in dirs]
    onager_prelaunch = f'onager prelaunch +jobname {args.jobname} +command "{args.command}" +arg --config {" ".join(config_paths)}'

    os.system(onager_prelaunch)
    
    
