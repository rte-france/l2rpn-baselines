#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import logging
from pathlib import Path
from typing import Union, Optional

import grid2op
from grid2op.Runner import Runner
from l2rpn_baselines.utils.save_log_gif import save_log_gif

from curriculumagent.baseline.baseline import CurriculumAgent


def evaluate(
        env: grid2op.Environment.BaseEnv,
        load_path: Union[str, Path] = ".",
        logs_path: Optional[Union[str, Path]] = None,
        nb_episode: int = 1,
        nb_process: int = 1,
        max_steps: int = -1,
        verbose: Union[bool, int] = False,
        save_gif: bool = False,
        **kwargs,
) -> Runner:
    """This is the evaluate method for the Curriculum Agent.

    Args:
        env: The environment on which the baseline will be evaluated. The default is the IEEE14 Case. For other
        environments please retrain the agent in advance.
        load_path: The path where the model is stored. This is used by the agent when calling "agent.load()"
        logs_path: The path where the agents results will be stored.
        nb_episode: Number of episodes to run for the assessment of the performance. By default, it equals 1.
        nb_process: Number of process to be used for the assessment of the performance. Should be an integer greater
        than 1. By default, it's equals 1.
        max_steps: Maximum number of timesteps each episode can last. It should be a positive integer or -1.
        -1 means that the entire episode is run (until the chronics is out of data or until a game over).
        By default,it equals -1.
        verbose: Verbosity of the output.
        save_gif:  Whether to save a gif into each episode folder corresponding to the representation of the said
        episode. Note, that depending on the environment (and the performance of your agent) this creation of the gif
        might take quite a lot of time!
        **kwargs:

    Returns:
        The experiment file consisting of the data.

    """
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    # Create the agent (this piece of code can change)
    agent = CurriculumAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        name="Evaluation"
    )
    # Load weights from file (for example)
    agent.load(load_path)

    # Build runner
    runner = Runner(**runner_params, agentClass=None, agentInstance=agent)

    # you can do stuff with your model here

    # start the runner

    if nb_process > 1:
        logging.warning(
            f"Parallel execution is not yet available for keras model. Therefore, the number of processes is comuted with "
            f"only one process."
        )
        nb_process = 1

    res = runner.run(path_save=logs_path, nb_episode=nb_episode, nb_process=nb_process, max_iter=max_steps, pbar=False)

    # Print summary
    logging.info("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        logging.info(msg_tmp)

    if save_gif:
        save_log_gif(logs_path, res)
    return res


if __name__ == "__main__":
    """
    This is a possible implementation of the eval script.
    """
    from lightsim2grid import LightSimBackend
    import grid2op

    logging.basicConfig(level=logging.INFO)
    env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
    obs = env.reset()
    path_of_model = Path(__file__).parent / "model_IEEE14"
    myagent = CurriculumAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        model_path=path_of_model,
        path_to_data=path_of_model,
        name="Test",
    )
    env = grid2op.make("l2rpn_case14_sandbox")
    out = evaluate(
        env,
        load_path=path_of_model,
        logs_path=Path(__file__).parent / "logs",
        nb_episode=10,
        nb_process=1,
        max_steps=-1,
        verbose=0,
        save_gif=True,
    )
