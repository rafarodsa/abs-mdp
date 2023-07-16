import argparse
import json, os

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=str)
    parser.add_argument('--header', type=str)
    parser.add_argument('--tasklist', type=str)
    parser.add_argument('--dryrun', action='store_true', default=False)
    args = parser.parse_args()

    with open(args.jobs, 'r') as f:
        jobs = json.load(f)
    with open(args.header, 'r') as f:
        header = f.readlines()
    header = ''.join(header)

    if args.tasklist is not None:
        t_ = tasklist(args.tasklist)
        jobs = [jobs[str(t)] for t in t_]
    else:
        jobs = [j for j in jobs.values()]
    

    cmd_str = '{}\n\nsrun {}'
    for job in jobs:
        job_cmd, job_tag = job
        sbatch_script = cmd_str.format(header, job_cmd)

        with open(f'{job_tag}.sh', 'w') as f:
            f.writelines(sbatch_script)
        if not args.dryrun:
            os.system(f'sbatch {job_tag}.sh')
            os.system(f'rm {job_tag}.sh')
    
    
