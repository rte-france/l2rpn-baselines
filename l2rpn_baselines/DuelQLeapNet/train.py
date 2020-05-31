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
from l2rpn_baselines.DuelQLeapNet.DuelQLeapNet import DuelQLeapNet, DEFAULT_NAME
from l2rpn_baselines.utils import TrainingParam

import pdb


def train(env,
          name=DEFAULT_NAME,
          iterations=1,
          save_path=None,
          load_path=None,
          logs_dir=None,
          nb_env=1,
          training_param=None,
          **kwargs_converters):

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if training_param is None:
        training_param = TrainingParam()

    baseline = DuelQLeapNet(env.action_space,
                            name=name,
                            istraining=True,
                            nb_env=nb_env,
                            lr=training_param.lr,
                            learning_rate_decay_steps=training_param.lr_decay_steps,
                            learning_rate_decay_rate=training_param.lr_decay_rate,
                            **kwargs_converters
                            )

    if load_path is not None:
        baseline.load(load_path)

    baseline.train(env,
                   iterations,
                   save_path=save_path,
                   logdir=logs_dir,
                   training_param=training_param)
    # as in our example (and in our explanation) we recommend to save the mode regurlarly in the "train" function
    # it is not necessary to save it again here. But if you chose not to follow these advice, it is more than
    # recommended to save the "baseline" at the end of this function with:
    # baseline.save(path_save)


if __name__ == "__main__":
    # import grid2op
    import numpy as np
    from grid2op.Parameters import Parameters
    from grid2op import make
    from grid2op.Reward import L2RPNReward
    import re
    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()

    args = cli_train().parse_args()

    # is it highly recommended to modify the reward depening on the algorithm.
    # for example here i will push my algorithm to learn that plyaing illegal or ambiguous action is bad
    class MyReward(L2RPNReward):
        def initialize(self, env):
            self.reward_min = 0.0
            self.reward_max = 1.0

        def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
            if has_error or is_illegal or is_ambiguous:
                # previous action was bad
                res = self.reward_min
            elif is_done:
                # really strong reward if an episode is over without game over
                res = self.reward_max
            else:
                res = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
                res /= env.n_line
                if not np.isfinite(res):
                    res = self.reward_min
            return res

    # Use custom params

    # Create grid2op game environement
    env_init = None
    from grid2op.Chronics import MultifolderWithCache
    game_param = Parameters()
    game_param.NB_TIMESTEP_COOLDOWN_SUB = 2
    game_param.NB_TIMESTEP_COOLDOWN_LINE = 2
    env = make(args.env_name,
               param=game_param,
               reward_class=MyReward,
               backend=backend,
               chronics_class=MultifolderWithCache
               )
    # env.chronics_handler.set_max_iter(7*288)
    env.chronics_handler.real_data.set_filter(lambda x: re.match(".*0[0-6][0-9]{2}$", x) is not None)
    env.chronics_handler.real_data.reset_cache()

    # env.chronics_handler.real_data.
    if args.nb_env > 1:
        env_init = env
        from grid2op.Environment import MultiEnvironment
        env = MultiEnvironment(int(args.nb_env), env)
        # TODO hack i'll fix in 0.9.0
        env.action_space = env_init.action_space
        env.observation_space = env_init.observation_space
        env.fast_forward_chronics = lambda x: None
        env.chronics_handler = env_init.chronics_handler
        env.current_obs = env_init.current_obs
        env.set_ff()

    tp = TrainingParam()
    tp.lr = 1e-3
    tp.lr_decay_steps = 30000
    tp.minibatch_size = 256
    tp.update_freq = 128
    tp.min_iter = 10
    tp.buffer_size = 1000000
    tp.min_observation = 10000
    tp.initial_epsilon = 0.4
    tp.final_epsilon = 1./(2*7*288.)
    tp.step_for_final_epsilon = int(1e5)
    kwargs_converters = {"all_actions": None,
                         "set_line_status": False,
                         "change_bus_vect": True,
                         "set_topo_vect": False
                         }
    nm_ = args.name if args.name is not None else DEFAULT_NAME
    try:
        train(env,
              name=nm_,
              iterations=args.num_train_steps,
              save_path=args.save_path,
              load_path=args.load_path,
              logs_dir=args.logs_dir,
              nb_env=args.nb_env,
              training_param=tp,
              **kwargs_converters)
    finally:
        env.close()
        if args.nb_env > 1:
            env_init.close()
