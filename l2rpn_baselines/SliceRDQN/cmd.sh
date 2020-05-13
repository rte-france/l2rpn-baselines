#!/bin/bash

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

export SRDQN_NAME=slice-rdqn-X.0.0.x
export SRDQN_DATA=~/data_grid2op/rte_case14_realistic

./inspect_action_space.py --path_data $SRDQN_DATA

rm -rf ./logs-train/$SRDQN_NAME
./train.py\
    --name $SRDQN_NAME \
    --data_dir $SRDQN_DATA \
    --num_pre_steps 256 \
    --num_train_steps 131072 \
    --trace_length 12

rm -rf ./logs-eval/$SRDQN_NAME
./evaluate.py \
    --data_dir $SRDQN_DATA \
    --load_file ./models/$SRDQN_NAME.tf \
    --logs_dir ./logs-eval/$SRDQN_NAME \
    --nb_episode 10
