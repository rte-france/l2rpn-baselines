#!/bin/bash

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

export DQN_NAME=dqn-X.0.0.x
export DQN_DATA=~/data_grid2op/rte_case14_realistic

./inspect_action_space.py --path_data $DQN_DATA

rm -rf ./logs-train/$DQN_NAME
./train.py\
    --name $DQN_NAME \
    --data_dir $DQN_DATA \
    --num_pre_steps 256 \
    --num_train_steps 131072 \
    --num_frames 4

rm -rf ./logs-eval/$DQN_NAME
./evaluate.py \
    --data_dir $DQN_DATA \
    --load_file ./models/$DQN_NAME.h5 \
    --logs_dir ./logs-eval/$DQN_NAME \
    --nb_episode 10
