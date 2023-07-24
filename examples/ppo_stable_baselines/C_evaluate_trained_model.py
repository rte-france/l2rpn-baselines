# Copyright (c) 2020-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import json
import numpy as np

import grid2op
from grid2op.utils import ScoreL2RPN2023
from grid2op.Agent import RecoPowerlineAgent

from lightsim2grid import LightSimBackend

from l2rpn_baselines.PPO_SB3 import evaluate

from A_prep_env import _aux_get_env, get_env_seed, name_stats
from B_train_agent import gymenv_class, name, safe_max_rho
# NB you can also chose to change the "safe_max_rho" parameter 
# and use a different parameter for evaluation than the one used for 
# training.

# env_name = "l2rpn_wcci_2022_val"
# SCOREUSED = ScoreL2RPN2022
env_name = "l2rpn_idf_2023_val"
SCOREUSED = ScoreL2RPN2023

agent_name = name
nb_scenario = 10
nb_process_stats = 1
load_path = "./saved_model_2023"
iter_num = None  # put None for the latest version
verbose = True


def load_agent(env, load_path, name,
               gymenv_class=gymenv_class,
               gymenv_kwargs={"safe_max_rho": safe_max_rho},
               obs_space_kwargs=None,
               act_space_kwargs=None):
    trained_agent, _ = evaluate(env,
                                nb_episode=0,
                                load_path=load_path,
                                name=name,
                                gymenv_class=gymenv_class,
                                iter_num=iter_num,
                                gymenv_kwargs=gymenv_kwargs,
                                obs_space_kwargs=obs_space_kwargs,
                                act_space_kwargs=act_space_kwargs)
    return trained_agent


def get_ts_survived_dn(env_name):
    dict_ = _aux_get_env(env_name, dn=True)
    res = []
    for kk in range(nb_scenario):
        tmp_ = dict_[f"{kk}"]["nb_step"]
        res.append(tmp_)
    res = np.array(res)
    res -= 1  # the first observation (after reset) is counted as a step in the runner
    return res


def get_ts_survived_reco(env_name):
    dict_ = _aux_get_env(env_name, name_stat=name_stats)
    res = []
    for kk in range(nb_scenario):
        tmp_ = dict_[f"{kk}"]["nb_step"]
        res.append(tmp_)
    res = np.array(res)
    res -= 1  # the first observation (after reset) is counted as a step in the runner
    return res


if __name__ == "__main__":
    
    # create the environment
    env_val = grid2op.make(env_name, backend=LightSimBackend())
    
    # retrieve the reference data
    dn_ts_survived = get_ts_survived_dn(env_name)
    reco_ts_survived = get_ts_survived_reco(env_name)

    my_score = SCOREUSED(env_val,
                         nb_scenario=nb_scenario,
                         env_seeds=get_env_seed(env_name)[:nb_scenario],
                         agent_seeds=[0 for _ in range(nb_scenario)],
                         verbose=verbose,
                         nb_process_stats=nb_process_stats,
                         )

    with open("preprocess_obs.json", "r", encoding="utf-8") as f:
        obs_space_kwargs = json.load(f)
    with open("preprocess_act.json", "r", encoding="utf-8") as f:
        act_space_kwargs = json.load(f)
        
    my_agent = load_agent(env_val,
                          load_path=load_path,
                          name=agent_name,
                          gymenv_class=gymenv_class,
                          obs_space_kwargs=obs_space_kwargs,
                          act_space_kwargs=act_space_kwargs)
    scores_r, n_played_r, total_ts_r = my_score.get(RecoPowerlineAgent(env_val.action_space))
    scores, n_played, total_ts = my_score.get(my_agent)
    
    res_scores = {"scores": [float(score[0]) for score in scores],
                  "n_played": [int(el) for el in n_played],
                  "total_ts": [int(el) for el in total_ts]}
    
    # compare with do nothing
    best_than_dn = 0
    for score, my_ts, dn_ts in zip(scores, n_played, dn_ts_survived):
        print(f"\t{':-)' if my_ts >= dn_ts else ':-('}:"
              f"\n\t\t- I survived {my_ts} steps vs {dn_ts} for do nothing ({my_ts - dn_ts})"
              f"\n\t\t- my score is {score[0]:.2f} (do nothing is 15.)")
        best_than_dn += my_ts >= dn_ts
    print(f"The agent \"{agent_name}\" beats \"do nothing\" baseline in {best_than_dn} out of {len(dn_ts_survived)} episodes")
    
    # compare with reco powerline
    best_than_reco = 0
    for score, my_ts, reco_ts, score_ in zip(scores, n_played, reco_ts_survived, scores_r):
        print(f"\t{':-)' if my_ts >= reco_ts else ':-('}:"
              f"\n\t\t- I survived {my_ts} steps vs {reco_ts} for reco powerline ({my_ts - reco_ts})"
              f"\n\t\t- my score is {score[0]:.2f} (reco powerline: {score_[0]:.2f})")
        best_than_reco += my_ts >= reco_ts
    print(f"The agent \"{agent_name}\" beats \"reco powerline\" baseline in {best_than_reco} out of {len(reco_ts_survived)} episodes")
