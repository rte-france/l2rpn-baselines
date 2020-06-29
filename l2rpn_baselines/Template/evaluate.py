#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
from grid2op.Runner import Runner

from l2rpn_baselines.Template.Template import Template
from l2rpn_baselines.utils.save_log_gif import save_log_gif


def evaluate(env,
             load_path=".",
             logs_path=None,
             nb_episode=1,
             nb_process=1,
             max_steps=-1,
             verbose=False,
             save_gif=False,
             **kwargs):
    """
    In order to submit a valid basline, it is mandatory to provide a "evaluate" function with the same signature as this one.

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

    verbose: ``bool``
        verbosity of the output

    save_gif: ``bool``
        Whether or not to save a gif into each episode folder corresponding to the representation of the said episode.

    kwargs:
        Other key words arguments that you are free to use for either building the agent save it etc.

    Returns
    -------
    ``None``
    """
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    # Create the agent (this piece of code can change)
    agent = Template(env.action_space, env.observation_space, "Template")

    # Load weights from file (for example)
    agent.load(load_path)

    # Build runner
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)

    # you can do stuff with your model here

    # start the runner
    res = runner.run(path_save=logs_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     pbar=False)

    # Print summary
    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

    if save_gif:
        save_log_gif(logs_path, res)


if __name__ == "__main__":
    """
    This is a possible implementation of the eval script.
    """
    import grid2op
    from l2rpn_baselines.utils import cli_eval
    args_cli = cli_eval().parse_args()
    env = grid2op.make()
    evaluate(env,
             load_path=args_cli.load_path,
             logs_path=args_cli.logs_path,
             nb_episode=args_cli.nb_episode,
             nb_process=args_cli.nb_process,
             max_steps=args_cli.max_steps,
             verbose=args_cli.verbose,
             save_gif=args_cli.save_gif)
