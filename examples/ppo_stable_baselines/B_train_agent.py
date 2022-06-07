# Copyright (c) 2020-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

# this file is likely to be run multiple times. You might also want
# to customize the reward, the attributes of the observation
# you want to keep etc.
# Remember this is an example, that should perform relatively well (better than
# do nothing)

import os
import re
import json
import numpy as np
from grid2op.Reward import BaseReward
from grid2op.Action import PlayableAction
from l2rpn_baselines.utils import GymEnvWithReco, GymEnvWithRecoWithDN

env_name = "l2rpn_wcci_2022_train"
save_path = "./saved_model"
name = "FirstAgent"
gymenv_class = GymEnvWithRecoWithDN  # uses the heuristic to do nothing is the grid is not at risk and to reconnect powerline automatically
max_iter = 7 * 24 * 12  # None to deactivate it
safe_max_rho = 0.9  # the grid is said "safe" if the rho is lower than this value, it is a really important parameter to tune !


# customize the reward function (optional)
class CustomReward(BaseReward):
    def __init__(self, logger=None):
        """
        Initializes :attr:`BaseReward.reward_min` and :attr:`BaseReward.reward_max`

        """
        BaseReward.__init__(self, logger=logger)
        self.reward_min = 0.
        self.reward_max = 1.
        self._min_rho = 0.90
        self._max_rho = 2.0
        
        # parameters init with the environment
        self._max_redisp = None
        self._1_max_redisp = None
        self._is_renew_ = None
        self._1_max_redisp_act = None
        self._nb_renew = None
    
    def initialize(self, env):
        self._max_redisp = np.maximum(env.gen_pmax - env.gen_pmin, 0.)
        self._max_redisp += 1
        self._1_max_redisp = 1.0 / self._max_redisp / env.n_gen
        self._is_renew_ = env.gen_renewable
        self._1_max_redisp_act = np.maximum(np.maximum(env.gen_max_ramp_up, env.gen_max_ramp_down), 1.0)
        self._1_max_redisp_act = 1.0 / self._1_max_redisp_act / np.sum(env.gen_redispatchable)
        self._nb_renew = np.sum(self._is_renew_)
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            res = np.sqrt(env.nb_time_step / env.max_episode_duration())
            print(f"{os.path.split(env.chronics_handler.get_id())[-1]}: {env.nb_time_step = }, reward : {res:.3f}")
            if env.nb_time_step <= 5:
                print(f"reason game over: {env.infos['exception']}")
            # episode is over => 2 cases
            # if env.nb_time_step == env.max_episode_duration():
            #     return self.reward_max
            # else:
            #     return self.reward_min
            return res
        
        if is_illegal or is_ambiguous or has_error:
            return self.reward_min
        # penalize the dispatch
        obs = env.get_obs()
        score_redisp_state = 0.
        # score_redisp_state = np.sum(np.abs(obs.target_dispatch) * self._1_max_redisp)
        score_redisp_action = np.sum(np.abs(action.redispatch) * self._1_max_redisp_act) 
        score_redisp = 0.5 *(score_redisp_state + score_redisp_action)
        
        # penalize the curtailment
        score_curtail_state = 0.
        # score_curtail_state = np.sum(obs.curtailment_mw * self._1_max_redisp)
        curt_act = action.curtail
        score_curtail_action = np.sum(curt_act[curt_act != -1.0]) / self._nb_renew 
        score_curtail = 0.5 * (score_curtail_state + score_curtail_action)
        
        # rate the actions
        score_action = 0.5 * (np.sqrt(score_redisp) + np.sqrt(score_curtail))
        
        # score the "state" of the grid
        # tmp_state = np.minimum(np.maximum(obs.rho, self._min_rho), self._max_rho)
        # tmp_state -= self._min_rho
        # tmp_state /= (self._max_rho - self._min_rho) * env.n_line
        # score_state = np.sqrt(np.sqrt(np.sum(tmp_state)))
        score_state = 0.

        # score close to goal
        score_goal = 0.
        # score_goal = env.nb_time_step / env.max_episode_duration()
        # score_goal = 1.0
        
        # score too much redisp
        res = score_goal * (1.0 - 0.5 * (score_action + score_state))
        return score_goal * res
    
    
if __name__ == "__main__":
    
    import grid2op
    from l2rpn_baselines.PPO_SB3 import train
    from lightsim2grid import LightSimBackend  # highly recommended !
    from grid2op.Chronics import MultifolderWithCache  # highly recommended for training
    
    # you can change below (full list at https://grid2op.readthedocs.io/en/latest/observation.html#main-observation-attributes)
    obs_attr_to_keep = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                        "gen_p", "load_p", 
                        "p_or", "rho", "timestep_overflow", "line_status",
                        # dispatch part of the observation
                        "actual_dispatch", "target_dispatch",
                        # storage part of the observation
                        "storage_charge", "storage_power",
                        # curtailment part of the observation
                        "curtailment", "curtailment_limit",  "gen_p_before_curtail",
                        ]
    TODO = ...
    # same here you can change it as you please
    act_attr_to_keep = ["redispatch", "curtail", "set_storage"]
    # parameters for the learning
    nb_iter = 30_000
    learning_rate = 3e-4
    net_arch = [200, 200, 200, 200]
    gamma = 0.999
    
    env = grid2op.make(env_name,
                       action_class=PlayableAction,
                       reward_class=CustomReward,
                       backend=LightSimBackend(),
                       chronics_class=MultifolderWithCache)
    
    with open("preprocess_obs.json", "r", encoding="utf-8") as f:
        obs_space_kwargs = json.load(f)
    with open("preprocess_act.json", "r", encoding="utf-8") as f:
        act_space_kwargs = json.load(f)
    
    # for this, you might want to have a look at: 
    #  - https://grid2op.readthedocs.io/en/latest/parameters.html#grid2op.Parameters.Parameters.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION
    #  - https://grid2op.readthedocs.io/en/latest/action.html#grid2op.Action.BaseAction.limit_curtail_storage
    # This really helps the training, but you cannot change
    # this parameter when you evaluate your agent, so you need to rely
    # on act.limit_curtail_storage(...) before you give your action to the
    # environment
    
    param = env.parameters
    param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
    env.change_parameters(param)
    
    if max_iter is not None:
        env.set_max_iter(max_iter)  # one week
    obs = env.reset()
    # train on all february month, why not ?
    env.chronics_handler.real_data.set_filter(lambda x: re.match(r".*2050-02-.*$", x) is not None)
    env.chronics_handler.real_data.reset()
    # see https://grid2op.readthedocs.io/en/latest/environment.html#optimize-the-data-pipeline
    # for more information !
    
    print("environment loaded !")
    trained_agent = train(
            env,
            iterations=nb_iter,
            logs_dir="./logs",
            save_path=save_path, 
            obs_attr_to_keep=obs_attr_to_keep,
            obs_space_kwargs=obs_space_kwargs,
            act_attr_to_keep=act_attr_to_keep,
            act_space_kwargs=act_space_kwargs,
            normalize_act=True,
            normalize_obs=True,
            name=name,
            learning_rate=learning_rate,
            net_arch=net_arch,
            save_every_xxx_steps=min(nb_iter // 10, 100_000),
            verbose=1,
            gamma=0.999,
            gymenv_class=gymenv_class,
            gymenv_kwargs={"safe_max_rho": safe_max_rho}
            )
