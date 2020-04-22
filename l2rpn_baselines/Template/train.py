# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from l2rpn_baselines.Template.TemplateBaseline import TemplateBaseline


def train(env,
          num_training_steps,
          path_save=None,
          name="baseline",
          path_loading=None,
          **kwargs):
    """
    This an example function to train a baseline.

    In order to be valid, if you chose (which is recommended) to provide a training script to help other retrain your
    baseline in different environments, or for longer period of time etc. This script should be contain the "train"
    function with at least the following arguments.

    Parameters
    ----------
    env: :class:`grid2op.Environment.Environment`
        The environmnent on which the baseline will be trained

    num_training_steps: ``int``
        Number of training step to perform

    path_save: ``str``
        The path where the baseline will be saved at the end of the training procedure.

    name: ``str``
        Fancy name you give to this baseline.

    path_loading: ``str``
        Path where to look for reloading the model. Use ``None`` if no model should be loaded.

    kwargs:
        Other key-word arguments that you might use for training.

    """

    baseline = TemplateBaseline(env.action_space,
                                env.observation_space,
                                name=name)

    if path_loading is not None:
        baseline.load_network(path_loading)

    baseline.train(env, num_training_steps, path_save=path_save)
    # as in our example (and in our explanation) we recommend to save the mode regurlarly in the "train" function
    # it is not necessary to save it again here. But if you chose not to follow these advice, it is more than
    # recommended to save the "baseline" at the end of this function with:
    # baseline.save(path_save)


if __name__ == "__main__":
    """
    This is a possible implementation of the eval script.
    """
    import grid2op
    from l2rpn_baselines.utils import cli_train
    args_cli = cli_train().parse_args()
    env = grid2op.make()
    train(env=env,
          num_training_steps=args_cli.num_train_steps,
          path_save=args_cli.path_save,
          name=args_cli.name,
          path_loading=args_cli.path_loading)