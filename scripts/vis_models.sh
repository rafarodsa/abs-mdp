

# input: directory base name (e.g. /users/rrodri19/abs-mdp/experiments/pb_obstacles/fullstate/mdps/pinball_rssm_small*)

VIS=''
MAKE_MDP=''
RSSM=''
OUTDIR=''
BASEDIR=''
DEVICE='cpu'

while getopts d:vmro:g: flag
do
    case "${flag}" in
        d) BASEDIR=${OPTARG};;
        v) VIS='TRUE';;
        m) MAKE_MDP='TRUE';;
        r) RSSM='--rssm';;
        o) OUT_DIR=$OPTARG;;
        g) DEVICE=$OPTARG
    esac
done

if [[ -z $BASEDIR ]]; then
    echo 'ERROR: Missing Experiment dir'
    exit -1
fi


for i in $(echo "${BASEDIR}*"); do
    echo "$i"
    if [[ -n $MAKE_MDP ]]; then
        python scripts/make_absmdp_experiments.py --experiment $i $RSSM
    fi

    if [[ -n $VIS ]]; then
        python scripts/visualize_experiment.py --experiment $i --device $DEVICE $RSSM 
    fi
done


# for each one
    # run vis generation
    # create mdp model
    # plan?


