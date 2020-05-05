#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import argparse
import tensorflow as tf

from grid2op.MakeEnv import make
from grid2op.Reward import *
from grid2op.Action import *
from grid2op.Parameters import Parameters

from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN import DoubleDuelingDQN as DDDQNAgent

DEFAULT_NAME = "DoubleDuelingDQN"
DEFAULT_SAVE_DIR = "./models"
DEFAULT_LOG_DIR = "./logs-train"
DEFAULT_PRE_STEPS = 256
DEFAULT_TRAIN_STEPS = 1024
DEFAULT_N_FRAMES = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-5

def cli():
    parser = argparse.ArgumentParser(description="Train baseline DDQN")

    # Paths
    parser.add_argument("--name", required=True,
                        help="The name of the model")
    parser.add_argument("--data_dir", required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--save_dir", required=False,
                        default=DEFAULT_SAVE_DIR, type=str,
                        help="Directory where to save the model")
    parser.add_argument("--load_file", required=False,
                        help="Path to model.h5 to resume training with")
    parser.add_argument("--logs_dir", required=False,
                        default=DEFAULT_LOG_DIR, type=str,
                        help="Directory to save the logs")
    # Params
    parser.add_argument("--num_pre_steps", required=False,
                        default=DEFAULT_PRE_STEPS, type=int,
                        help="Number of random steps before training")
    parser.add_argument("--num_train_steps", required=False,
                        default=DEFAULT_TRAIN_STEPS, type=int,
                        help="Number of training iterations")
    parser.add_argument("--num_frames", required=False,
                        default=DEFAULT_N_FRAMES, type=int,
                        help="Number of stacked states to use during training")
    parser.add_argument("--batch_size", required=False,
                        default=DEFAULT_BATCH_SIZE, type=int,
                        help="Mini batch size (defaults to 1)")
    parser.add_argument("--learning_rate", required=False,
                        default=DEFAULT_LR, type=float,
                        help="Learning rate for the Adam optimizer")

    return parser.parse_args()


def train(env,
          name = DEFAULT_NAME,
          iterations = DEFAULT_TRAIN_STEPS,
          save_path = DEFAULT_SAVE_DIR,
          load_path = None,
          logs_path = DEFAULT_LOG_DIR,
          num_pre_training_steps = DEFAULT_PRE_STEPS,
          num_frames = DEFAULT_N_FRAMES,
          batch_size= DEFAULT_BATCH_SIZE,
          learning_rate= DEFAULT_LR):

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    agent = DDDQNAgent(env.observation_space,
                       env.action_space,
                       name=name,
                       is_training=True,
                       batch_size=batch_size,
                       num_frames=num_frames,
                       lr=learning_rate)

    if load_path is not None:
        agent.load(load_path)

    agent.train(env,
                iterations,
                save_path,
                num_pre_training_steps,
                logs_path)


if __name__ == "__main__":
    args = cli()

    # Use custom params
    params = Parameters()
    params.MAX_SUB_CHANGED = 2

    # Create grid2op game environement
    env = make(args.data_dir,
               param=params,
               action_class=TopologyChangeAndDispatchAction,
               reward_class=CombinedScaledReward)

    # Only load 128 steps in ram
    env.chronics_handler.set_chunk_size(128)

    # Register custom reward for training
    cr = env.reward_helper.template_reward
    cr.addReward("overflow", CloseToOverflowReward(), 50.0)
    cr.addReward("game", GameplayReward(), 200.0)
    cr.addReward("recolines", LinesReconnectedReward(), 50.0)
    # Initialize custom rewards
    cr.initialize(env)
    # Set reward range to something managable
    cr.set_range(-10.0, 10.0)

    train(env,
          name = args.name,
          iterations = args.num_train_steps,
          num_pre_training_steps = args.num_pre_steps,
          save_path = args.save_dir,
          load_path = args.load_file,
          logs_path = args.logs_dir,
          num_frames = args.num_frames,
          batch_size = args.batch_size,
          learning_rate = args.learning_rate)
