#!/bin/bash

export DQN_NAME=dqn-X.0.0.x
export DQN_DATA=~/data_grid2op/rte_case14_realistic

./inspect_action_space.py --path_data $DQN_DATA

rm -rf ./logs/$DQN_NAME
./train.py \
    --name $DQN_NAME \
    --data_dir $DQN_DATA \
    --num_pre_steps 256 \
    --num_train_steps 131072 \
    --num_frames 4

rm -rf ./logs-$DQN_NAME
./evaluate.py \
    --data_dir $DQN_DATA \
    --load_file ./models/$DQN_NAME.h5 \
    --logs_dir ./logs-eval/$DQN_NAME \
    --nb_episode 10
