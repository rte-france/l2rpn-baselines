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

from curriculumagent.baseline import CurriculumAgent


def train(env,
          name="Template",
          iterations=1,
          model_path=Union[Path, str],
          save_path=Optional[Union[Path, str]],
          load_path=None,
          **kwargs):
    """ Train method of the agent. Generally try training with the baseline agent.

    Args:
        env: Grid2op Environment
        name: Name of the Training agent
        iterations: Number of iterations to train
        model_path: Where to find the initial model. This is required in order to initialize the model.
        save_path: Optional Savepath where to save the model
        load_path: Optional Load Path, if a model should be loaded.
        **kwargs:

    Returns: None

    """
    baseline = CurriculumAgent(action_space=env.action_space,
                               model_path=model_path,
                               name=name)

    if load_path is not None:
        baseline.load(load_path)

    baseline.train(env=env, name=name, iterations=iterations, save_path=save_path)


if __name__ == "__main__":
    """
    This is a possible implementation of the train script.
    """
    import grid2op
    from l2rpn_baselines.utils import cli_train

    args_cli = cli_train().parse_args()
    env = grid2op.make()
    train(env=env,
          name=args_cli.name,
          iterations=args_cli.num_train_steps,
          save_path=args_cli.save_path,
          load_path=args_cli.load_path)
