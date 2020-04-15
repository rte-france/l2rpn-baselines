#!/bin/bash

export RDQN_NAME=rdqn-X.0.0.x
export RDQN_DATA=~/data_grid2op/rte_case14_realistic

./inspect_action_space.py --path_data $RDQN_DATA

rm -rf ./logs-train/$RDQN_NAME
./train.py \
    --name $RDQN_NAME \
    --data_dir $RDQN_DATA \
    --num_pre_steps 256 \
    --num_train_steps 131072 \
    --trace_length 12

rm -rf ./logs-eval/$RDQN_NAME
./evaluate.py \
    --data_dir $RDQN_DATA \
    --load_file ./models/$RDQN_NAME.h5 \
    --logs_dir ./logs-eval/$RDQN_NAME \
    --nb_episode 10
