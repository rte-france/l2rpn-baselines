# Copyright (c) 2020-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

# this file needs to be run only once, it might take a while !
import os
import json
import sys
import numpy as np
import grid2op
from grid2op.dtypes import dt_int
from grid2op.Agent import RecoPowerlineAgent
from grid2op.utils import EpisodeStatistics, ScoreL2RPN2022, ScoreICAPS2021
from lightsim2grid import LightSimBackend
import numpy as np

is_windows = sys.platform.startswith("win32")

env_name = "l2rpn_wcci_2022"
SCOREUSED = ScoreL2RPN2022  # ScoreICAPS2021

name_stats = "_reco_powerline"
nb_process_stats = 4 if not is_windows else 1
# if you still want to use multi processing on windows
# have a look at the `env.generate_classe()` function
verbose = 1
deep_copy = is_windows  # force the deep copy on windows (due to permission issue in symlink in windows)


def _aux_get_env(env_name, dn=True, name_stat=None):
    path_ = grid2op.get_current_local_dir()
    path_env = os.path.join(path_, env_name)
    if not os.path.exists(path_env):
        raise RuntimeError(f"The environment \"{env_name}\" does not exist.")
    
    if SCOREUSED == ScoreICAPS2021:
        path_dn = os.path.join(path_env, "_statistics_icaps2021_dn")
    else:
        path_dn = os.path.join(path_env, "_statistics_l2rpn_dn")
        
    if not os.path.exists(path_dn):
        raise RuntimeError("The folder _statistics_icaps2021_dn (or _statistics_l2rpn_dn) used for computing the score do not exist")
    path_reco = os.path.join(path_env, "_statistics_l2rpn_no_overflow_reco")
    if not os.path.exists(path_reco):
        raise RuntimeError("The folder _statistics_l2rpn_no_overflow_reco used for computing the score do not exist")
    
    if name_stat is None:
        if dn:
            path_metadata = os.path.join(path_dn, "metadata.json")
        else:
            path_metadata = os.path.join(path_reco, "metadata.json")
    else:
        path_stat = os.path.join(path_env, EpisodeStatistics.get_name_dir(name_stat))
        if not os.path.exists(path_stat):
            raise RuntimeError(f"No folder associated with statistics {name_stat}")
        path_metadata = os.path.join(path_stat, "metadata.json")
    
    if not os.path.exists(path_metadata):
        raise RuntimeError("No metadata can be found for the statistics you wanted to compute.")
    
    with open(path_metadata, "r", encoding="utf-8") as f:
        dict_ = json.load(f)
    
    return dict_


def get_env_seed(env_name: str):
    """This function ensures that you can reproduce the results of the computed scenarios.
    
    It forces the seeds of the environment, during evaluation to be the same as the one used during the evaluation of the score.
    
    As environments are stochastic in grid2op, it is very important that you use this function (or a similar one) before
    computing the scores of your agent.

    Args:
        env_name (str): The environment name on which you want to retrieve the seeds used

    Raises:
        RuntimeError: When it is not possible to retrieve the seeds (for example when the "statistics" has not been computed)

    Returns:
        [type]: [description]
    """

    dict_ = _aux_get_env(env_name)
    
    key = "env_seeds"
    if key not in dict_:
        raise RuntimeError(f"Impossible to find the key {key} in the dictionnary. You should re run the score function.")
    
    return dict_[key]


if __name__ == "__main__":
    # create the environment 
    env = grid2op.make(env_name)

    # split into train / val / test
    # it is such that there are 25 chronics for val and 24 for test
    env.seed(1)
    env.reset()
    nm_train, nm_val, nm_test = env.train_val_split_random(add_for_test="test",
                                                           pct_val=4.2,
                                                           pct_test=4.2,
                                                           deep_copy=deep_copy)

    # computes some statistics for val / test to compare performance of 
    # some agents with the do nothing for example
    max_int = max_int = np.iinfo(dt_int).max
    for nm_ in [nm_val, nm_test]:
        env_tmp = grid2op.make(nm_, backend=LightSimBackend())
        nb_scenario = len(env_tmp.chronics_handler.subpaths)
        print(f"{nm_}: {nb_scenario}")
        my_score = SCOREUSED(env_tmp,
                             nb_scenario=nb_scenario,
                             env_seeds=np.random.randint(low=0,
                                                         high=max_int,
                                                         size=nb_scenario,
                                                         dtype=dt_int),
                             agent_seeds=[0 for _ in range(nb_scenario)],
                             verbose=verbose,
                             nb_process_stats=nb_process_stats,
                             )

        # compute statistics for reco powerline
        seeds = get_env_seed(nm_)
        reco_powerline_agent = RecoPowerlineAgent(env_tmp.action_space)
        stats_reco = EpisodeStatistics(env_tmp, name_stats=name_stats)
        stats_reco.compute(nb_scenario=nb_scenario,
                           agent=reco_powerline_agent,
                           env_seeds=seeds)
        
        if nm_ == nm_val:
            # save the normalization parameters from the validation set
            dict_ = {"subtract": {}, 'divide': {}}
            for attr_nm in ["gen_p", "load_p", "p_or", "rho"]:
                avg_ = stats_reco.get(attr_nm)[0].mean(axis=0)
                std_ = stats_reco.get(attr_nm)[0].std(axis=0)
                dict_["subtract"][attr_nm] = [float(el) for el in avg_]
                dict_["divide"][attr_nm] = [max(float(el), 1.0) for el in std_]
            
            with open("preprocess_obs.json", "w", encoding="utf-8") as f:
                json.dump(obj=dict_, fp=f)
                
            act_space_kwargs = {"add": {"redispatch": [0. for gen_id in range(env.n_gen) if env.gen_redispatchable[gen_id]],
                                        "set_storage": [0. for _ in range(env.n_storage)]},
                                'multiply': {"redispatch": [1. / (max(float(el), 1.0)) for gen_id, el in enumerate(env.gen_max_ramp_up) if env.gen_redispatchable[gen_id]],
                                             "set_storage": [1. / (max(float(el), 1.0)) for el in env.storage_max_p_prod]}
                               }
            with open("preprocess_act.json", "w", encoding="utf-8") as f:
                json.dump(obj=act_space_kwargs, fp=f)
