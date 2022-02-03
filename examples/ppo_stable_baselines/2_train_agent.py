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
import numpy as np
from grid2op.Reward import BaseReward

env_name = "l2rpn_icaps_2021_small_train"
save_path = "./saved_model"

# customize the reward function (optional)
class CustomReward(BaseReward):
    def __init__(self):
        """
        Initializes :attr:`BaseReward.reward_min` and :attr:`BaseReward.reward_max`

        """
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
            print(f"{os.path.split(env.chronics_handler.get_id())[-1]}: {env.nb_time_step = }")
            # episode is over => 2 cases
            # if env.nb_time_step == env.max_episode_duration():
            #     return self.reward_max
            # else:
            #     return self.reward_min
            return env.nb_time_step / env.max_episode_duration()
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
    from l2rpn_baselines.utils import GymEnvWithReco
    
    obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour",
                        "gen_p", "load_p", "p_or",
                        "actual_dispatch", "target_dispatch",
                        "rho", "timestep_overflow", "line_status",
                        "curtailment", "gen_p_before_curtail"]

    act_attr_to_keep = ["redispatch", "curtail"]
    nb_iter = 6_000_000
    learning_rate = 3e-3
    net_arch = [300, 300, 300]
    name = "expe_with_auto_reco_onlyend_ep"
    gamma = 0.999
    
    env = grid2op.make(env_name,
                       reward_class=CustomReward,
                       backend=LightSimBackend(),
                       chronics_class=MultifolderWithCache)

    obs = env.reset()
    # env.chronics_handler.real_data.set_filter(lambda x: re.match(r".*00$", x) is not None)
    env.chronics_handler.real_data.set_filter(lambda x: True)
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
            act_attr_to_keep=act_attr_to_keep,
            normalize_act=True,
            normalize_obs=True,
            name=name,
            learning_rate=learning_rate,
            net_arch=net_arch,
            save_every_xxx_steps=min(nb_iter // 10, 100_000),
            verbose=1,
            gamma=0.999,
            gymenv_class=GymEnvWithReco,
            )
    
    print("After training, ")
    # TODO evaluate it !