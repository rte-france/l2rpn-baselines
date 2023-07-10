#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
from pathlib import Path
from typing import Union, Optional

import grid2op
import ray

from curriculumagent.baseline import CurriculumAgent


def train(
        env: grid2op.Environment.BaseEnv,
        name: str = "Training_Pipeline",
        iterations: int = 1,
        save_path=Optional[Union[Path, str]],
        **kwargs
):
    """
    This is the method to train the full pipeline of the CurriculumAgent.
    For a quick training the CurriculumAgent we recommend to use the agent.train() method to only
    train the RL agent. This, however, requires that you already have an model and an action set.

    If you want/have to start at zero and need to find actions as well as the model, feel free to run
    this pipeline. This pipeline will create for each step (agent) in the pipeline a directory.

    Note two things:
    1. This pipeline does not require a loading of the CurriculumAgent and thus does not need an action
    set and a pretrained model. You start from scratch.
    2. This execution is computationally expensive. Further, your machine needs to be able to run RLlib.

    Args:
        env: Grid2op Environment.
        name: Name of the Training agent.
        iterations: Number of iterations to train.
        save_path: Optional Save path where to save the model.
        **kwargs: Additional arguments you want to pass to the train_full_pipeline.

    Returns:
        None.

    """
    baseline = CurriculumAgent(action_space=env.action_space, observation_space=env.observation_space, name=name)

    baseline.train_full_pipeline(env=env, name=name, iterations=iterations, save_path=save_path,**kwargs)
    ray.shutdown()



if __name__ == "__main__":
    """
    This is a possible implementation of the train script.
    """
    import grid2op
    from l2rpn_baselines.utils import cli_train

    args_cli = cli_train().parse_args()
    env = grid2op.make()
    train(
        env=env,
        name=args_cli.name,
        iterations=args_cli.num_train_steps,
        save_path=args_cli.save_path,
    )
