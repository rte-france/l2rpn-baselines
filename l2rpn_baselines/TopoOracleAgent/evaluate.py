#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import logging
from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from train import cli,DEFAULT_SAVE_DIR,DEFAULT_NAME,DEFAULT_SCENARIO,DEFAULT_DATA_DIR,DEFAULT_MAX_DEPTH,DEFAULT_MAX_TIMESTEPS,DEFAULT_ENV_SEED,DEFAULT_AGENT_SEED,DEFAULT_ACTION_FILE,DEFAULT_BEST_ACTION_PATH_TYPE,DEFAULT_SIGNIFICANT_DIGIT
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from ressources.constants import EnvConstants,BACKEND


try:
    from l2rpn_baselines.TopoOracleAgent.TopoOracleAgent import TopoOracleAgent
    from l2rpn_baselines.utils.save_log_gif import save_log_gif
    from oracle4grid.core.oracle import load_oracle_data_for_replay
    from oracle4grid.core.utils.prepare_environment import prepare_env
    #from l2rpn_baselines.ExpertAgent.expertAgent import other_rewards
    _CAN_USE_ORACLE_BASELINE = True
except ImportError as exc_:
    _CAN_USE_ORACLE_BASELINE = False

#Default values are already set in train.py . Just changing log_dir value here
DEFAULT_LOG_DIR = "./agent_logs"


def evaluate(env,chronic,env_seed,agent_seed,
            name = DEFAULT_NAME,
            iterations = DEFAULT_MAX_TIMESTEPS,
            max_combinatorial_depth=DEFAULT_MAX_DEPTH,
            save_path = DEFAULT_SAVE_DIR,
            logs_path=DEFAULT_LOG_DIR,
            action_file_path = DEFAULT_ACTION_FILE,
            best_action_path_type=DEFAULT_BEST_ACTION_PATH_TYPE,
            reward_significant_digit=DEFAULT_SIGNIFICANT_DIGIT,
            verbose=False,
            save_gif=False):
    """
        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The environment on which the baseline will be evaluated.

        chronic: :class: ``str``
            The name of the scenario of interest

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

        verbose: ``bool``
            verbosity of the output

        save_gif: ``bool``
            Whether or not to save a gif into each episode folder corresponding to the representation of the said
            episode.

        Returns
        -------
        res: ``list``
            List of tuple. Each tuple having 3 elements:

              - "i" unique identifier of the episode (compared to :func:`Runner.run_sequential`, the elements of the
                returned list are not necessarily sorted by this value)
              - "cum_reward" the cumulative reward obtained by the :attr:`Runner.BaseAgent` on this episode i
              - "nb_time_step": the number of time steps played in this episode.
              - "max_ts" : the maximum number of time steps of the chronics
              - "episode_data" : The :class:`EpisodeData` corresponding to this episode run

    """
    if not _CAN_USE_ORACLE_BASELINE:
        raise ImportError("OracleAgent baseline impossible to load the required dependencies for using the model. "
                         )

    CONFIG = {
        "max_depth": max_combinatorial_depth,
        "max_iter": iterations,
        "nb_process": 1,
        "best_path_type": best_action_path_type,
        "n_best_topos": 2,
        "reward_significant_digit": reward_significant_digit,
        "replay_reward_rel_tolerance": 1e7
    }

    # Build runner
    param = Parameters()
    constants = EnvConstants()
    #to reload possible combined action list, need to allow for infinte sub changed and line changed
    param.init_from_dict(constants.DICT_GAME_PARAMETERS_REPLAY)#now we play with normal parameters
    env.change_parameters(param)
    env_temporary, chronic_id = prepare_env(env.get_path_env(), chronic, param, constants=constants)

    save_dir = os.path.join(save_path,env.name, name, chronic)
    action_list_reloaded, init_topo_vect, init_line_status, oracle_actions_in_path = load_oracle_data_for_replay(env_temporary,
                                                                                                                 action_file_path,
                                                                                                                 save_dir,
                                                                                                                 action_depth= CONFIG["max_depth"])
    agent = TopoOracleAgent(
                 env.action_space,
                 action_list_reloaded,
                 oracle_action_path=oracle_actions_in_path,
                 init_topo_vect = init_topo_vect,
                 init_line_status = init_line_status)


    env_seed = env_seed
    agent_seed = agent_seed

    #set env and chronic
    os.makedirs(logs_path, exist_ok=True)
    env.set_id(chronic_id)
    obs = env.reset()

    # Run
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent
                    )
    res =runner.run_one_episode(indx=chronic_id,
                             path_save=logs_path,
                             pbar=True,
                             env_seed=env_seed,  # ENV_SEEDS,
                             max_iter=iterations,
                             agent_seed=agent_seed,  # AGENT_SEEDS,
                             detailed_output=True)

    # Print summary
    logging.info("Evaluation summary:")
    msg_tmp = "chronics at: {}".format(res[0])
    msg_tmp += "\ttotal reward: {:.6f}".format(res[1])
    msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(res[2], iterations)
    logging.info(msg_tmp)

    if save_gif:
        save_log_gif(logs_path, res)
    return res

if __name__ == "__main__":
    # Parse command line
    args = cli()
    if(args.logs_dir is None):
        args.logs_dir=DEFAULT_LOG_DIR
    # Create dataset env
    param = Parameters()
    constants=EnvConstants()
    param.init_from_dict(constants.DICT_GAME_PARAMETERS_REPLAY)
    env = make(args.data_dir,backend=BACKEND(),param=param,action_class=PlayableAction)

    # Call evaluation interface

    evaluate(env,
          name = args.name,chronic=args.scenario,
          env_seed=args.env_seed,agent_seed=args.env_seed,
          iterations = args.max_timesteps,
          max_combinatorial_depth=args.max_depth,
          save_path = args.save_dir,
          logs_path = args.logs_dir,
          action_file_path = args.action_file,
          reward_significant_digit=args.n_significant_digits,
          verbose=args.verbose,
          save_gif=args.gif
          )

