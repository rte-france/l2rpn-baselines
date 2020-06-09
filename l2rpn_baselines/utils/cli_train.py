# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import argparse


def cli_train():
    """some default command line arguments (cli) for training the baselines. Can be reused in some baselines here."""
    parser = argparse.ArgumentParser(description="Train baseline DDQN")
    parser.add_argument("--num_train_steps", required=False,
                        default=1024, type=int,
                        help="Number of training iterations")
    parser.add_argument("--save_path", required=False,
                        help="Path where the model should be saved.")
    parser.add_argument("--name", required=False,
                        help="Name given to your model.")
    parser.add_argument("--nb_env", required=False, default=1, type=int,
                        help="Number of process to use when training your Agent. If > 1 then MultiEnv will be used. "
                             "NB: not all models are compatible. NB it does not work on windows at the moment. "
                             "NB: experimental at the moment.")
    parser.add_argument("--load_path", required=False,
                        help="Path from which to reload your model from (by default ``None`` to NOT reload anything)")
    parser.add_argument("--env_name", required=False, default="l2rpn_case14_sandbox",
                        help="Name of the environment to load (default \"l2rpn_case14_sandbox\"")
    parser.add_argument("--logs_dir", required=False, default=None,
                        help="Where to output the training logs (usually tensorboard logs)")
    return parser
