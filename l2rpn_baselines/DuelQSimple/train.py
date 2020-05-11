#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import tensorflow as tf
from l2rpn_baselines.utils import cli_train
from l2rpn_baselines.DeepQSimple.DeepQSimple import DeepQSimple, DEFAULT_NAME


def train(env,
          name=DEFAULT_NAME,
          iterations=1,
          save_path=None,
          load_path=None,
          logs_dir=None):

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    baseline = DeepQSimple(env.action_space,
                           name=name,
                           istraining=True)

    if load_path is not None:
        baseline.load(load_path)

    baseline.train(env,
                   iterations,
                   save_path=save_path,
                   logdir=logs_dir)
    # as in our example (and in our explanation) we recommend to save the mode regurlarly in the "train" function
    # it is not necessary to save it again here. But if you chose not to follow these advice, it is more than
    # recommended to save the "baseline" at the end of this function with:
    # baseline.save(path_save)


if __name__ == "__main__":
    from grid2op.Parameters import Parameters
    from grid2op import make
    from grid2op.Reward import L2RPNReward
    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()

    args = cli_train().parse_args()

    # Use custom params
    params = Parameters()

    # Create grid2op game environement
    env = make(args.env_name,
               param=params,
               reward_class=L2RPNReward,
               backend=backend)

    nm_ = args.name if args.name is not None else DEFAULT_NAME
    train(env,
          name=nm_,
          iterations=args.num_train_steps,
          save_path=args.save_path,
          load_path=args.load_path,
          logs_dir=args.logs_dir)
