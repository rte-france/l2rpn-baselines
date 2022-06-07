#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import numpy as np
from tqdm import tqdm

from grid2op.MakeEnv import make

from l2rpn_baselines.LeapNetEncoded.leapNetEncoded import LeapNetEncoded, DEFAULT_NAME
from l2rpn_baselines.LeapNetEncoded.leapNetEncoded_NNParam import LeapNetEncoded_NNParam
from l2rpn_baselines.LeapNetEncoded.leapNetEncoded_NN import LeapNetEncoded_NN

import pdb

DEFAULT_LOGS_DIR = "./logs-eval/do-nothing-baseline"
DEFAULT_NB_EPISODE = 1
DEFAULT_NB_PROCESS = 1
DEFAULT_MAX_STEPS = -1


def study(env,
          name=DEFAULT_NAME,
          load_path=None,
          logs_path=DEFAULT_LOGS_DIR,
          nb_episode=DEFAULT_NB_EPISODE,
          nb_process=DEFAULT_NB_PROCESS,
          max_steps=DEFAULT_MAX_STEPS,
          verbose=False,
          save_gif=False):
    """
    study the prediction of the grid_model
    
    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
    """

    import tensorflow as tf
    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    if load_path is None:
        raise RuntimeError("Cannot evaluate a model if there is nothing to be loaded.")
    path_model, path_target_model = LeapNetEncoded_NN.get_path_model(load_path, name)
    nn_archi = LeapNetEncoded_NNParam.from_json(os.path.join(path_model, "nn_architecture.json"))

    # Run
    # Create agent
    agent = LeapNetEncoded(action_space=env.action_space,
                           name=name,
                           store_action=nb_process == 1,
                           nn_archi=nn_archi,
                           observation_space=env.observation_space)

    # Load weights from file
    agent.load(load_path)

    # Print model summary
    stringlist = []
    agent.deep_q._model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    if verbose:
        print(short_model_summary)

    from grid2op.Agent import RandomAgent
    from grid2op.Agent import DoNothingAgent
    policy_agent = DoNothingAgent(env.action_space)
    policy_agent.seed(0)

    env.set_id(0)
    res = {k: ([], []) for k in nn_archi.list_attr_obs_gm_out}
    with tqdm(desc="step") as pbar:
        for i in range(nb_episode):
            obs = env.reset()
            reward = env.reward_range[0]
            done = False
            while not done:
                obs_converted = agent.convert_obs(obs)
                data_nn, true_output_grid = agent.deep_q._make_x_tau(obs_converted)

                for i, (var_n, add, mult) in enumerate(zip(nn_archi.list_attr_obs_gm_out,
                                                           nn_archi.gm_out_adds,
                                                           nn_archi.gm_out_mults)):
                    tmp = true_output_grid[i]
                    tmp = tmp / mult - add
                    true_output_grid[i] = tmp

                pred = agent.deep_q.grid_model.predict(data_nn, batch_size=1)
                real_pred = []
                for i, (var_n, add, mult) in enumerate(zip(nn_archi.list_attr_obs_gm_out,
                                                           nn_archi.gm_out_adds,
                                                           nn_archi.gm_out_mults)):
                    tmp = pred[i]
                    tmp = tmp / mult - add
                    real_pred.append(tmp)

                for i, var_n in enumerate(nn_archi.list_attr_obs_gm_out):
                    res[var_n][0].append(real_pred[i].reshape(-1))
                    res[var_n][1].append(true_output_grid[i].reshape(-1))

                obs, reward, done, info = env.step(policy_agent.act(obs, reward, done))
                pbar.update(1)

    print("Results")
    from sklearn.metrics import mean_squared_error
    for var_n, (pred, true) in res.items():
        true = np.array(true)
        pred = np.array(pred)
        RMSE = mean_squared_error(y_true=true, y_pred=pred, multioutput="raw_values", squared=False)
        print("RMSE for {}: {:.2f} % variance".format(var_n, 100. * np.mean(RMSE / np.std(true))))
    return agent


if __name__ == "__main__":
    from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward
    from l2rpn_baselines.utils import cli_eval
    from grid2op.Parameters import Parameters

    # Parse command line
    args = cli_eval().parse_args()

    # Create dataset env
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = True
    env = make(args.env_name,
               reward_class=L2RPNSandBoxScore,
               other_rewards={
                   "reward": L2RPNReward
               },
               param=param)

    # python3 study.py --env_name="l2rpn_wcci_2020" --load_path="model_saved" --logs_dir="tf_logs" --name="TestObserver"  --nb_episode 10
    # Call evaluation interface
    study(env,
             name=args.name,
             load_path=os.path.abspath(args.load_path),
             logs_path=args.logs_dir,
             nb_episode=args.nb_episode,
             nb_process=args.nb_process,
             max_steps=args.max_steps,
             verbose=args.verbose,
             save_gif=args.save_gif)
