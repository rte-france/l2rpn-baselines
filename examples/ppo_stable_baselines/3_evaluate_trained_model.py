# Copyright (c) 2020-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import json
import grid2op
import numpy as np
from grid2op.utils import ScoreICAPS2021
from lightsim2grid import LightSimBackend
from l2rpn_baselines.PPO_SB3 import evaluate


env_name = "l2rpn_icaps_2021_small_val"
nb_scenario = 25
nb_process_stats = 1
load_path = "./saved_model"


def _aux_get_env(env_name, dn=True):
    path_ = grid2op.get_current_local_dir()
    path_env = os.path.join(path_, env_name)
    if not os.path.exists(path_env):
        raise RuntimeError(f"The environment \"{env_name}\" does not exist.")
    
    path_dn = os.path.join(path_env, "_statistics_icaps2021_dn")
    if not os.path.exists(path_dn):
        raise RuntimeError("The folder _statistics_icaps2021_dn used for computing the score do not exist")
    path_reco = os.path.join(path_env, "_statistics_l2rpn_no_overflow_reco")
    if not os.path.exists(path_reco):
        raise RuntimeError("The folder _statistics_l2rpn_no_overflow_reco used for computing the score do not exist")
    if dn:
        path_metadata = os.path.join(path_dn, "metadata.json")
    else:
        path_metadata = os.path.join(path_reco, "metadata.json")
        
    if not os.path.exists(path_metadata):
        raise RuntimeError("The folder _statistics_icaps2021_dn does not appear to be a score folder")
    
    with open(path_metadata, "r", encoding="utf-8") as f:
        dict_ = json.load(f)
    
    return dict_

def get_env_seed(env_name):
    dict_ = _aux_get_env(env_name)
    
    key = "env_seeds"
    if key not in dict_:
        raise RuntimeError(f"Impossible to find the key {key} in the dictionnary. You should re run the score function.")
    
    return dict_[key]

def load_agent(env, load_path, name):
    trained_agent, _ = evaluate(env,
                                nb_episode=0,
                                load_path=load_path,
                                name=name)
    return trained_agent


def get_ts_survived_dn(env_name):
    dict_ = _aux_get_env(env_name, dn=True)
    res = []
    for kk in range(nb_scenario):
        tmp_ = dict_[f"{kk}"]["nb_step"]
        res.append(tmp_)
    return res


if __name__ == "__main__":
    name = "expe_0"
    
    #
    env_val = grid2op.make(env_name, backend=LightSimBackend())
    my_score = ScoreICAPS2021(env_val,
                              nb_scenario=nb_scenario,
                              env_seeds=get_env_seed(env_name),
                              agent_seeds=[0 for _ in range(nb_scenario)],
                              verbose=False,
                              nb_process_stats=nb_process_stats,
                              )

    my_agent = load_agent(env_val, load_path=load_path, name=name)
    _, ts_survived, _ = my_score.get(my_agent)
    dn_ts_survived = get_ts_survived_dn(env_name)
    best_than_dn = 0
    for my_ts, dn_ts in zip(ts_survived, dn_ts_survived):
        print(f"\t{':-)' if my_ts >= dn_ts else ':-('} I survived {my_ts} steps vs {dn_ts} for do nothing ({my_ts - dn_ts})")
        best_than_dn += my_ts >= dn_ts
    print(f"The agent \"{name}\" beats do nothing in {best_than_dn} out of {len(dn_ts_survived)} episodes")
