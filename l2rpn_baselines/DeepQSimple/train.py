#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import tensorflow as tf
from l2rpn_baselines.utils import cli_train as cli
from l2rpn_baselines.DeepQSimple.DeepQSimple import DeepQSimple, DEFAULT_NAME


def train(env,
          name=DEFAULT_NAME,
          iterations=1,
          save_path=None,
          load_path=None,
          logs_dir=None,
          nb_env=1):

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    baseline = DeepQSimple(env.action_space,
                           name=name,
                           istraining=True,
                           nb_env=nb_env)

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
    args = cli().parse_args()

    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()

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
            return res

    # Use custom params
    params = Parameters()

    # Create grid2op game environement
    env_init = None
    env = make(args.env_name,
               param=params,
               reward_class=MyReward,
               backend=backend)

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

    nm_ = args.name if args.name is not None else DEFAULT_NAME
    try:
        train(env,
              name=nm_,
              iterations=args.num_train_steps,
              save_path=args.save_path,
              load_path=args.load_path,
              logs_dir=args.logs_dir,
              nb_env=args.nb_env)
    finally:
        env.close()
        if args.nb_env > 1:
            env_init.close()

