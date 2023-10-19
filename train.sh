#!/usr/bin/env bash
#
# Script that launches policy training
#

if [ "$#" -lt 2 ]; then
    echo "Illegal number of parameters. Usage: <name of a directory for output files> <policy type> <OPTIONAL extra train args>"
    exit 1
fi

OUTPUT_DIR="$1"
POLICY="$2"
EXTRA_ARGS=""

if [ "$#" -eq 3 ]; then
    EXTRA_ARGS="$3"
    echo $EXTRA_ARGS
fi

SCRIPT_DIR=$(realpath $(dirname $0))
cd $SCRIPT_DIR
source .venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR/Python-RVO2:$SCRIPT_DIR/sarl_star_ros/CrowdNav

python sarl_star_ros/CrowdNav/crowd_nav/train.py \
    --env_config "sarl_star_ros/CrowdNav/crowd_nav/configs/env.config" \
    --policy "${POLICY}" \
    --policy_config "sarl_star_ros/CrowdNav/crowd_nav/configs/policy.config" \
    --train_config "sarl_star_ros/CrowdNav/crowd_nav/configs/train.config" \
    --output_dir "custom_training/data/${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

deactivate
