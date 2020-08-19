#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import argparse

from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from grid2op.Reward import *
from grid2op.Action import *

from l2rpn_baselines.utils.save_log_gif import save_log_gif
from l2rpn_baselines.SAC.SAC_Agent import SAC_Agent
from l2rpn_baselines.SAC.SAC_Config_NN import SAC_Config_NN
from l2rpn_baselines.SAC.SAC_NN import SAC_NN
from l2rpn_baselines.utils import tf_limit_gpu_usage

DEFAULT_LOGS_DIR = "./logs-eval/SAC"
DEFAULT_NB_EPISODE = 1
DEFAULT_NB_PROCESS = 1
DEFAULT_MAX_STEPS = -1

def cli():
    parser = argparse.ArgumentParser(description="Evaluate SAC baseline")

    parser.add_argument("--dataset", required=True,
                        help="Dataset name or path to a dataset root dir")
    parser.add_argument("--logs_dir",
                        required=False, default=DEFAULT_LOGS_DIR,
                        help="Directory where to save the logs")
    parser.add_argument("--load_path", required=True,
                        help="Resume training from model path")
    parser.add_argument("--nn_config", required=True,
                        help="Path to NN json config file")
    parser.add_argument("--nb_episode", required=False,
                        type=int, default=DEFAULT_NB_EPISODE,
                        help="Number of episodes to run")
    parser.add_argument("--max_steps", required=False,
                        type=int, default=DEFAULT_MAX_STEPS,
                        help="Number of episodes to run")
    parser.add_argument("--nb_process", required=False,
                        type=int, default=DEFAULT_NB_PROCESS,
                        help="Number of processes to use")
    parser.add_argument("--quiet", action="store_false",
                        help="Disable verbose logging")
    return parser.parse_args()

def evaluate(env,
             load_path=None,
             logs_path=DEFAULT_LOGS_DIR,
             nb_episode=DEFAULT_NB_EPISODE,
             nb_process=DEFAULT_NB_PROCESS,
             max_steps=DEFAULT_MAX_STEPS,
             nn_param=None,
             verbose=True):

    tf_limit_gpu_usage()

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    if load_path is None:
        err_msg = "Cannot evaluate a model if there is nothing to be loaded."
        raise RuntimeError(err_msg)

    # Run
    # Create agent
    agent = SAC_Agent(observation_space=env.observation_space,
                      action_space=env.action_space,
                      nn_config=nn_param,
                      training=False)

    # Load weights from file
    agent.nn.load_network(load_path)

    # Build runner
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)

    # Run
    os.makedirs(logs_path, exist_ok=True)
    res = runner.run(path_save=logs_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     pbar=verbose)

    # Print summary

    if verbose:
        print("Evaluation summary:")
        for _, chron_name, cum_reward, nb_time_step, max_ts in res:
            msg_tmp = "chronics at: {}".format(chron_name)
            msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
            msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step,
                                                            max_ts)
            print(msg_tmp)

    return agent, res


if __name__ == "__main__":
    from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward
    from l2rpn_baselines.utils import cli_eval

    # Parse command line
    args = cli()

    nn_conf = SAC_Config_NN.from_json_file(args.nn_config)

    # Create dataset env
    env = make(args.dataset,
               other_rewards={
                   "capacity": L2RPNReward,
                   "codalab": L2RPNSandBoxScore
               })

    # Call evaluation interface
    evaluate(env,
             load_path=args.load_path,
             nn_param=nn_conf,
             logs_path=args.logs_dir,
             nb_episode=args.nb_episode,
             nb_process=args.nb_process,
             max_steps=args.max_steps,
             verbose=args.quiet)
