# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.


def train_generic(agent,
                   env,
                   name="Template",
                   iterations=1,
                   save_path=None,
                   load_path=None,
                  **kwargs_train):
    """
    This function is a helper to train more easily some agent using their default "train" method.

    Parameters
    ----------
    agent: :class:`grid2op.Agent`
        A grid2op agent that must implement all the baseline attributes and the train method.

    env: :class:`grid2op.Environment`
        The environment on which to train your baseline. It must be compatible with the agent created.

    name: ``str``
        Here for compatibility with the baseline "train" method. Currently unused (define the name when you create
        your baseline)

    iterations: ``int``
        Number of iterations on which to train your agent.

    save_path: ``str``
        Where to save your results (put None do deactivate saving)

    load_path: ``str``
        Path to load the agent from.

    kwargs_train: ``dict``
        Other argument that will be passed to `agent.train(...)`

    Returns
    -------
    agent: :class:`grid2op.Agent`
        The trained agent.

    """

    if load_path is not None:
        agent.load(load_path)

    agent.train(env,
                iterations,
                save_path,
                **kwargs_train)

    return agent


if __name__ == "__main__":
    pass
