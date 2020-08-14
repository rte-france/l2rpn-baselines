#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import tensorflow as tf
import warnings
import numpy as np
import argparse

from grid2op.Parameters import Parameters
from grid2op import make
from grid2op.Reward import L2RPNReward

from l2rpn_baselines.SAC.SAC_NN import SAC_NN
from l2rpn_baselines.SAC.SAC_Config_NN import SAC_Config_NN
from l2rpn_baselines.SAC.SAC_Config_Train import SAC_Config_Train
from l2rpn_baselines.SAC.SAC_Agent import SAC_Agent

from l2rpn_baselines.utils import cli_train
from l2rpn_baselines.utils.waring_msgs import _WARN_GPU_MEMORY
from l2rpn_baselines.utils import make_multi_env

def cli():
    parser = argparse.ArgumentParser(description="Train SAC baseline")
    parser.add_argument("--name", required=True,
                        help="The name of the model")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name or path to a dataset root dir")
    parser.add_argument("--nb_env", required=False, default=1,
                        help="Enable MultiProcess training")
    parser.add_argument("--save_path", required=True,
                        help="Directory where to save the model")
    parser.add_argument("--logs_dir", required=True,
                        help="Directory where to save the logs")
    parser.add_argument("--load_path", required=False,
                        help="Resume training from model path")
    parser.add_argument("--num_iterations", required=True,
                        type=int, help="Number of training iterations")
    parser.add_argument("--nn_config", required=True,
                        help="Path to NN json config file")
    parser.add_argument("--train_config", required=True,
                        help="Path to training config json file")
    return parser.parse_args()

def train(env,
          name="SAC",
          iterations=1,
          save_path=None,
          logs_dir=None,
          load_path=None,
          verbose=True,
          nn_param=None,
          train_param=None):

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    baseline = SAC_Agent(observation_space=env.observation_space,
                         action_space=env.action_space,
                         name=name,
                         nn_config=nn_param,
                         verbose=verbose,
                         training=True)

    if load_path is not None:
        baseline.load(load_path)

    baseline.train(env,
                   iterations,
                   save_path,
                   logs_dir,
                   train_param)


if __name__ == "__main__":
    args = cli()

    # Load configs
    nn_conf = SAC_Config_NN.from_json_file(args.nn_config)
    train_conf = SAC_Config_Train.from_json_file(args.train_config)
    
    # Use custom grid2op params
    game_param = Parameters()

    # Get fast backend if available
    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()

    # Create grid2op game environement
    env = make(args.dataset,
               param=game_param,
               reward_class=L2RPNReward,
               backend=backend)
    # Handle MultiProcessing
    env_init = env
    if args.nb_env > 1:
        env = make_multi_env(env_init=env_init, nb_env=int(args.nb_env))

    # Call training interface
    train(env,
          name=args.name,
          iterations=args.num_iterations,
          save_path=args.save_path,
          logs_dir=args.logs_dir,
          load_path=args.load_path,
          nn_param=nn_conf,
          train_param=train_conf)

    env_init.close()
