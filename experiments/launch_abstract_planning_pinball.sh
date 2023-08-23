#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 -j <job_name> <search_directory> <string_prefix>"
    exit 1
fi
dryrun=''
finetune='false'


# Parse the command line options and arguments
while getopts j:df opt; do
    case "$opt" in
        j) job_name="$OPTARG";;
        d) dryrun='-d';;
        f) finetune='true';;
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

matching_subdirs=$(find "$search_directory" -maxdepth 1 -type d -name "$string_prefix*" -exec basename {} \;)

# echo "$matching_subdirs"

echo $job_name

count=1
jobnames=()
for path in $matching_subdirs; do

    onager prelaunch +command "python experiments/pb_obstacles/fullstate/plan.py --config experiments/pb_obstacles/fullstate/config/ddqn.yaml --experiment_cwd ${search_directory} --experiment_name ${path} --experiment.finetune ${finetune} " \
    +jobname "${job_name}_${count}" \
    +arg --use-ground-init true \
    +arg --env.goal 0 1 2 3 4 5 6 7 8 9 \
    +arg --experiment.seed 31 56 43 64 24 \
    +tag --exp-id

    jobnames+=("${job_name}_${count}")
    count=$((count + 1))
    
done


for jobname in "${jobnames[@]}"; do
    onager launch --jobname "$jobname" --backend slurm -p 3090-gcondo --cpus 10 --mem 16 --tasks-per-node 10 --time 1-00:00:00 $dryrun
done 

