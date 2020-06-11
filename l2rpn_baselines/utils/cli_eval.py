# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import argparse
from l2rpn_baselines.utils.str2bool import str2bool


# TODO add possibilitiy to add new CLI when calling the function
def cli_eval():
    """ some usefull command line arguments (CLI) for the evaluation of the baseline."""
    parser = argparse.ArgumentParser(description="Train baseline DDQN")
    parser.add_argument("--env_name", default="l2rpn_case14_sandbox",
                        help="Name of the l2rpn dataset environment")
    parser.add_argument("--load_path", default=".",
                        help="The path where the trained model has been stored")
    parser.add_argument("--name", default="baseline",
                        help="The name of the trained baseline")
    parser.add_argument("--logs_dir", required=False,
                        default="./logs-eval", type=str,
                        help="Path to output logs directory")
    parser.add_argument("--nb_episode", required=False,
                        default=1, type=int,
                        help="Number of episodes to evaluate")
    parser.add_argument("--nb_process", required=False,
                        default=1, type=int,
                        help="Number of cores to use")
    parser.add_argument("--max_steps", required=False,
                        default=8000, type=int,
                        help="Maximum number of steps per scenario")
    parser.add_argument("--verbose", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Enable verbose runner mode..")
    parser.add_argument("--save_gif", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Save the gif as \"epidose.gif\" in the episode path module.")
    return parser
