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
from l2rpn_baselines.SAC.SAC import SAC, DEFAULT_NAME
from l2rpn_baselines.SAC.SAC_NNParam import SAC_NNParam
from l2rpn_baselines.SAC.SAC_NN import SAC_NN

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
    """
    How to evaluate the performances of the trained SAC agent.

    Parameters
    ----------
    env: :class:`grid2op.Environment`
        The environment on which you evaluate your agent.

    name: ``str``
        The name of the trained baseline

    load_path: ``str``
        Path where the agent has been stored

    logs_path: ``str``
        Where to write the results of the assessment

    nb_episode: ``str``
        How many episodes to run during the assessment of the performances

    nb_process: ``int``
        On how many process the assessment will be made. (setting this > 1 can lead to some speed ups but can be
        unstable on some plaform)

    max_steps: ``int``
        How many steps at maximum your agent will be assessed

    verbose: ``bool``
        Currently un used

    save_gif: ``bool``
        Whether or not you want to save, as a gif, the performance of your agent. It might cause memory issues (might
        take a lot of ram) and drastically increase computation time.

    Returns
    -------
    agent: :class:`l2rpn_baselines.utils.DeepQAgent`
        The loaded agent that has been evaluated thanks to the runner.

    res: ``list``
        The results of the Runner on which the agent was tested.


    Examples
    -------
    You can evaluate a DeepQSimple this way:

    .. code-block:: python

        from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward
        from l2rpn_baselines.SAC import eval

        # Create dataset env
        env = make("l2rpn_case14_sandbox",
                   reward_class=L2RPNSandBoxScore,
                   other_rewards={
                       "reward": L2RPNReward
                   })

        # Call evaluation interface
        evaluate(env,
                 name="MyAwesomeAgent",
                 load_path="/WHERE/I/SAVED/THE/MODEL",
                 logs_path=None,
                 nb_episode=10,
                 nb_process=1,
                 max_steps=-1,
                 verbose=False,
                 save_gif=False)
    """

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    if load_path is None:
        raise RuntimeError("Cannot evaluate a model if there is nothing to be loaded.")
    path_model, path_target_model = SAC_NN.get_path_model(load_path, name)
    nn_archi = SAC_NNParam.from_json(os.path.join(path_model, "nn_architecture.json"))

    # Run
    # Create agent
    agent = SAC(action_space=env.action_space,
                name=name,
                store_action=nb_process == 1,
                nn_archi=nn_archi,
                observation_space=env.observation_space)

    # Load weights from file
    agent.load(load_path)

    # Print model summary
    stringlist = []
    agent.deep_q.model_value.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    if verbose:
        print("Value model: {}".format(short_model_summary))

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
            msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)

        if len(agent.dict_action):
            # I output some of the actions played
            print("The agent played {} different action".format(len(agent.dict_action)))
            for id_, (nb, act, types) in agent.dict_action.items():
                print("Action with ID {} was played {} times".format(id_, nb))
                print("{}".format(act))
                print("-----------")

    if save_gif:
        if verbose:
            print("Saving the gif of the episodes")
        save_log_gif(logs_path, res)

    return agent, res


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
