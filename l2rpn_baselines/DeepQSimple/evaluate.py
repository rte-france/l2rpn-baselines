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

from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from grid2op.Reward import *
from grid2op.Action import *

from l2rpn_baselines.utils.save_log_gif import save_log_gif
from l2rpn_baselines.DeepQSimple.DeepQSimple import DeepQSimple, DEFAULT_NAME

DEFAULT_LOGS_DIR = "./logs-eval/do-nothing-baseline"
DEFAULT_NB_EPISODE = 1
DEFAULT_NB_PROCESS = 1
DEFAULT_MAX_STEPS = -1


def evaluate(env,
             name=DEFAULT_NAME,
             load_path=None,
             logs_path=DEFAULT_LOGS_DIR,
             nb_episode=DEFAULT_NB_EPISODE,
             nb_process=DEFAULT_NB_PROCESS,
             max_steps=DEFAULT_MAX_STEPS,
             verbose=False,
             save_gif=False):

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = args.verbose

    # Run
    # Create agent
    agent = DeepQSimple(action_space=env.action_space,
                        name=name,
                        store_action=nb_process == 1)
    # force creation of the neural networks
    obs = env.reset()
    _ = agent.act(obs, 0., False)

    # Load weights from file
    agent.load(load_path)

    # Build runner
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)

    # Print model summary
    stringlist = []
    agent.deep_q.model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)

    # Run
    os.makedirs(logs_path, exist_ok=True)
    res = runner.run(path_save=logs_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     pbar=True)

    # Print summary
    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

    if len(agent.dict_action):
        # I output some of the actions played
        print("The agent played {} different action".format(len(agent.dict_action)))
        for id_, (nb, act) in agent.dict_action.items():
            print("Action with ID {} was played {} times".format(id_, nb))
            print("{}".format(act))
            print("-----------")

    if save_gif:
        print("Saving the gif of the episodes")
        save_log_gif(logs_path, res)


if __name__ == "__main__":
    from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward
    from l2rpn_baselines.utils import cli_eval

    # Parse command line
    args = cli_eval().parse_args()

    # Create dataset env
    env = make(args.env_name,
               reward_class=L2RPNSandBoxScore,
               other_rewards={
                   "reward": L2RPNReward
               })

    # Call evaluation interface
    evaluate(env,
             name=args.name,
             load_path=os.path.abspath(args.load_path),
             logs_path=args.logs_dir,
             nb_episode=args.nb_episode,
             nb_process=args.nb_process,
             max_steps=args.max_steps,
             verbose=args.verbose,
             save_gif=args.save_gif)
