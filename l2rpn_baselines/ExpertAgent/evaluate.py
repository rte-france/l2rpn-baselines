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
import numpy as np
import logging
from grid2op.dtypes import dt_int
from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from grid2op.Reward import *
from grid2op.Action import *

try:
    from l2rpn_baselines.ExpertAgent import ExpertAgent
    from l2rpn_baselines.utils.save_log_gif import save_log_gif
    from l2rpn_baselines.ExpertAgent.ExpertAgent import other_rewards
except ImportError as exc_:
    raise ImportError("ExpertAgent baseline impossible to load the required dependencies for using the model. "
                      "The error was: \n {}".format(exc_))


DEFAULT_LOGS_DIR = "./logs-eval/expert-agent-baseline"
DEFAULT_NB_EPISODE = 1
DEFAULT_NB_PROCESS = 1
DEFAULT_MAX_STEPS = -1


def cli():
    parser = argparse.ArgumentParser(description="Eval baseline DDDQN")
    parser.add_argument("--data_dir", required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--logs_dir", required=False,
                        default=DEFAULT_LOGS_DIR, type=str,
                        help="Path to output logs directory")
    parser.add_argument("--nb_episode", required=False,
                        default=DEFAULT_NB_EPISODE, type=int,
                        help="Number of episodes to evaluate")
    parser.add_argument("--nb_process", required=False,
                        default=DEFAULT_NB_PROCESS, type=int,
                        help="Number of cores to use")
    parser.add_argument("--max_steps", required=False,
                        default=DEFAULT_MAX_STEPS, type=int,
                        help="Maximum number of steps per scenario")
    parser.add_argument("--gif", action='store_true',
                        help="Enable GIF Output")
    parser.add_argument("--verbose", action='store_true',
                        help="Verbose runner output")
    parser.add_argument("--grid", required=True,
                        help="Grid id to use")
    parser.add_argument("--test", required=False, default=False,
                        help="Whether the data set is test or not")
    return parser.parse_args()


def evaluate(env,
             load_path=None,
             logs_path=DEFAULT_LOGS_DIR,
             nb_episode=DEFAULT_NB_EPISODE,
             nb_process=DEFAULT_NB_PROCESS,
             max_steps=DEFAULT_MAX_STEPS,
             grid="IEEE14", #IEEE14,IEEE118_3 (WCCI or Neurips Track Robustness), IEEE118
             seed=None,
             verbose=False,
             save_gif=False):
    """
        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The environment on which the baseline will be evaluated.

        load_path: ``str``
            The path where the model is stored. This is used by the agent when calling "agent.load)

        logs_path: ``str``
            The path where the agents results will be stored.

        nb_episode: ``int``
            Number of episodes to run for the assessment of the performance.
            By default it's 1.

        nb_process: ``int``
            Number of process to be used for the assessment of the performance.
            Should be an integer greater than 1. By defaults it's 1.

        max_steps: ``int``
            Maximum number of timestep each episode can last. It should be a positive integer or -1.
            -1 means that the entire episode is run (until the chronics is out of data or until a game over).
            By default it's -1.

        grid: ``string``
            Name identifier of the environment grid. Used for local optimisation of choices

        seed: ``int``
            seed info for reproducibility purposes

        verbose: ``bool``
            verbosity of the output

        save_gif: ``bool``
            Whether or not to save a gif into each episode folder corresponding to the representation of the said
            episode.

        Returns
        -------
        ``None``
    """
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    # Build runner
    agent = ExpertAgent(env.action_space, env.observation_space, "Template", grid)
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent
                    )
    env_seeds = None
    agent_seeds = None
    if seed is not None:
        np.random.seed(seed)
        max_int = np.iinfo(dt_int).max
        env_seeds = list(np.random.randint(max_int, size=int(nb_episode)))
        agent_seeds = list(np.random.randint(max_int, size=int(nb_episode)))

    # Run
    os.makedirs(logs_path, exist_ok=True)
    res = runner.run(path_save=logs_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     env_seeds=env_seeds,
                     agent_seeds=agent_seeds,
                     pbar=True)

    # Print summary
    logging.info("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "chronics at: {}".format(chron_name)
        msg_tmp += "\ttotal reward: {:.6f}".format(cum_reward)
        msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
        logging.info(msg_tmp)

    if save_gif:
        save_log_gif(logs_path, res)
    return res

if __name__ == "__main__":
    # Parse command line
    args = cli()
    # Create dataset env
    env = make(args.data_dir,
               test=args.test,
               reward_class=RedispReward,
               action_class=TopologyChangeAction,
               other_rewards=other_rewards
               )
    # Call evaluation interface
    evaluate(env,
             logs_path=args.logs_dir,
             nb_episode=args.nb_episode,
             nb_process=args.nb_process,
             max_steps=args.max_steps,
             verbose=args.verbose,
             save_gif=args.gif,
             grid=args.grid)
