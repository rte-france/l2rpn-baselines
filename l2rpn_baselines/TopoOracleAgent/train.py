#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os.path
import argparse

from oracle4grid.core.utils.launch_utils import load
from oracle4grid.core.oracle import oracle, save_oracle_data_for_replay,OracleParser
from ressources.constants import EnvConstants,BACKEND #where to change chosen reward


DEFAULT_NAME = "TopoOracleAgent"
DEFAULT_DATA_DIR= "rte_case14_realistic"#"l2rpn_wcci_2022"#"l2rpn_neurips_2020_track1_small"#
DEFAULT_SAVE_DIR = "./output_best_action_paths"
DEFAULT_LOG_DIR = "./logs"
DEFAULT_ACTION_FILE="./ressources/actions/rte_case14_realistic/test_unitary_actions_5.json"#"./ressources/actions/l2rpn_wcci_2022/LIPS_actions.json"#"./ressources/actions/neurips_track1/LIPS_actions.json"#./ressources/actions/Ieee14_Sandbox_test/unitary_actions_l2rpn_2019.json"#"./ressources/actions/neurips_track1/ExpertActions_Track1_action_list_score4.json"
DEFAULT_SCENARIO="000"#"2050-12-26_31"#"Scenario_april_000#"000"
DEFAULT_MAX_DEPTH = 2
DEFAULT_AGENT_SEED = 0
DEFAULT_ENV_SEED = 0
DEFAULT_MAX_TIMESTEPS = 10#288 #one day. 2016 if you want a full week
DEFAULT_SIGNIFICANT_DIGIT=2
DEFAULT_N_CORES=8
DEFAULT_BEST_ACTION_PATH_TYPE="longest" #shortest if you want to minimize given the chosen reward inEnvConstants. Longest if you rather want to maximize the reward in


def cli():
    parser = argparse.ArgumentParser(description="Train baseline TopoOracleAgent")
    # Paths
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help="The name of the model")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Path to the dataset root directory")
    parser.add_argument("--save_dir", required=False,
                        default=DEFAULT_SAVE_DIR, type=str,
                        help="Directory where to save the ouput best action path")
    parser.add_argument("--action_file",default=DEFAULT_ACTION_FILE, type=str,
                        help="Path to file where unitary actions of interest are describedh")
    parser.add_argument("--logs_dir", required=False,
                        default=None, type=str,
                        help="Directory to save the logs if desired - None by default")
    parser.add_argument("--gif", action='store_true',
                        help="Enable GIF Output")
    parser.add_argument("--verbose", action='store_true',
                        help="Verbose runner output")

    # Params
    parser.add_argument("--scenario",  type=str,default=DEFAULT_SCENARIO,
                        help="Scenario name of interest")
    parser.add_argument("--agent_seed", type=int,default=DEFAULT_AGENT_SEED,
                        help="Agent seed for reproducible scenario")
    parser.add_argument("--env_seed", type=int,default=DEFAULT_ENV_SEED,
                        help="Env seed for reproducible scenario")
    parser.add_argument("--max_timesteps", required=False,
                        default=DEFAULT_MAX_TIMESTEPS, type=int,
                        help="Number of timestep to run in scenario")
    parser.add_argument("--max_depth", required=False,
                        default=DEFAULT_MAX_DEPTH, type=int,
                        help="Maximum explored combinatorial depth of unitary actions")
    parser.add_argument("--n_significant_digits", required=False,
                        default=DEFAULT_SIGNIFICANT_DIGIT, type=int,
                        help="Number of significant digits to consider in reward")
    parser.add_argument("--n_cores", required=False,
                        default=DEFAULT_N_CORES, type=int,
                        help="Number of cpu cores to run multiprocessing on")
    parser.add_argument("--best_action_path_type", required=False,
                        default=DEFAULT_BEST_ACTION_PATH_TYPE, type=str,
                        help="shortest - if you want to minimize given the chosen reward inEnvConstants. longest - if you rather want to maximize the reward in")
    return parser.parse_args()


def train(env,chronic,env_seed,agent_seed,
          name = DEFAULT_NAME,
          iterations = DEFAULT_MAX_TIMESTEPS,
          max_combinatorial_depth=DEFAULT_MAX_DEPTH,
          save_path = DEFAULT_SAVE_DIR,
          logs_path=DEFAULT_LOG_DIR,
          action_file_path = DEFAULT_ACTION_FILE,
          nb_process=DEFAULT_N_CORES,
          best_action_path_type=DEFAULT_BEST_ACTION_PATH_TYPE,
          reward_significant_digit=DEFAULT_SIGNIFICANT_DIGIT):

    """
        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The environment on which the baseline will be evaluated.

        chronic: :class: ``str``
            The name of the scenario of interest

        name: :class: ``str``
            The name of the agent that will be used for saving results

        env_seed: :class: ``int``
            The env seed to set for the scenario of interest, to be able to reproduce or replay the scenario in the same conditions

        agent_seed: :class: ``int``
            The env seed to set for the scenario of interest, to be able to reproduce or replay the scenario in the same conditions

        save_path: ``str``
            The root path where the best action path, that is the output of training, is stored. This is used by the agent when reloading for evaluate or replay)

        logs_path: ``str``
            The path where the detailed agent training results will be stored.

        action_file_path: ``str``
            The path of the file where unitary actions of choice are described. Several format are possible (see https://oracle4grid.readthedocs.io/en/latest/DESCRIPTION.html#action-parsing )

        nb_process: ``int``
            Number of cores on which to run the training. A lot (above 100 CPUs) is preferred to speedup a lot the computation

        best_action_path_type: ``str``
            "shortest" if you want to minimize given the chosen reward in EnvConstants. "longest" if you rather want to maximize the reward in EnvConstants

        reward_significant_digit: ``int``
            number of relevant significant digit to consider for reward when finding best path

        Returns
        -------
        best_path: :class:`Oracle Action`
            list of oracle actions from best computation path. It is also saved from root folder save_path

        best_path_no_overload: :class:`Oracle Action`
            list of oracle actions from best computation path with the constraint of no overload at all. It is also saved from root folder save_path
    """

    CONFIG = {
        "max_depth": max_combinatorial_depth,
        "max_iter": iterations,
        "nb_process": nb_process,
        "best_path_type": best_action_path_type,
        "n_best_topos": 2,
        "reward_significant_digit": reward_significant_digit,
        "replay_reward_rel_tolerance": 1e7
    }

    param = Parameters()
    constants = EnvConstants()
    param.init_from_dict(constants.DICT_GAME_PARAMETERS_SIMULATION)

    env_dir=env.get_path_env()

    # Load unitary actions and env_oracle to run computation in
    atomic_actions, env_oracle, debug_directory, chronic_id = load(env_dir, chronic, action_file_path, debug=False,
                                                            constants=EnvConstants())
    parser = OracleParser(atomic_actions, env.action_space)
    atomic_actions = parser.parse()

    logs_dir=None
    do_degug=False
    if(logs_path is not None):
        logs_dir=os.path.join(logs_path, name, chronic)
        os.makedirs(logs_dir, exist_ok=True)
        do_degug=True

    res = oracle(atomic_actions, env_oracle, do_degug , config=CONFIG, debug_directory=logs_dir,
                 agent_seed=agent_seed, env_seed=env_seed,
                 grid_path=env_dir, chronic_scenario=chronic_id, constants=constants)  #

    best_path, grid2op_action_path, best_path_no_overload, grid2op_action_path_no_overload, kpis = res

    oracle_action_list = best_path

    save_dir = os.path.join(save_path,env.name, name, chronic)
    os.makedirs(save_dir,exist_ok=True)
    save_oracle_data_for_replay(oracle_action_list, save_dir) #save oracle actions path by action names
    return best_path,best_path_no_overload


if __name__ == "__main__":
    from grid2op.MakeEnv import make
    from grid2op.Reward import *
    from grid2op.Action import *
    from grid2op.Parameters import Parameters
    import sys

    args = cli()

    # Create grid2op game environement
    env = make(args.data_dir,backend=BACKEND())


    train(env,
          name = args.name,chronic=args.scenario,
          env_seed=args.env_seed,agent_seed=args.env_seed,
          iterations = args.max_timesteps,
          max_combinatorial_depth=args.max_depth,
          save_path = args.save_dir,
          logs_path = args.logs_dir,
          action_file_path = args.action_file,
          nb_process=args.n_cores,
          best_action_path_type=args.best_action_path_type,
          reward_significant_digit=args.n_significant_digits)
