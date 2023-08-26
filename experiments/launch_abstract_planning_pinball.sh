#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 -j <job_name> <search_directory> <string_prefix>"
    exit 1
fi
dryrun=''
finetune='false'
partition='3090-gcondo'
substring=''
outdir='planning_ddqn'

# Parse the command line options and arguments
while getopts j:dfp:s:o: opt; do
    case "$opt" in
        j) job_name="$OPTARG";;
        d) dryrun='-d';;
        f) finetune='true';;
        p) partition="$OPTARG";;
        s) substring="$OPTARG";;
        o) outdir="$OPTARG";;
        \?) echo "Usage: $0 -j <job_name> <search_directory> <string_prefix>"
            exit 1;;
    esac
done

echo "$dryrun"
echo "$finetune"

# Shift the arguments to the remaining ones after the options
shift $((OPTIND - 1))

search_directory="$1"
string_prefix="$2"

if [ -z "$substring" ]; then
    matching_subdirs=$(find "$search_directory" -maxdepth 1 -type d -name "$string_prefix*" -exec basename {} \;)
else
    matching_subdirs=$(find "$search_directory" -maxdepth 1 -type d -name "$string_prefix*" -exec basename {} \; | grep "$substring")
fi

# echo "$matching_subdirs"

echo $job_name

count=1
jobnames=()
for path in $matching_subdirs; do

    onager prelaunch +command "python experiments/pb_obstacles/fullstate/plan.py --config experiments/pb_obstacles/fullstate/config/ddqn_sim.yaml --experiment_cwd ${search_directory} --experiment_name ${path} --experiment.finetune ${finetune} --experiment.outdir ${outdir} " \
    +jobname "${job_name}_${count}" \
    +arg --use-ground-init false \
    +arg --env.goal 0 1 2 3 4 5 6 7 8 9 \
    +arg --experiment.seed 31 56 43 64 24 \
    +tag --exp-id

    jobnames+=("${job_name}_${count}")
    count=$((count + 1))
    
done

for jobname in "${jobnames[@]}"; do
    onager launch --jobname "$jobname" --backend slurm -p $partition --cpus 4 --mem 8 --tasks-per-node 4 --time 6:00:00 $dryrun
done 
