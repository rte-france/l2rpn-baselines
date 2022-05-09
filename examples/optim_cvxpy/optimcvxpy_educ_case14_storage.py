# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import grid2op
from grid2op.Action import PlayableAction
from l2rpn_baselines.OptimCVXPY import OptimCVXPY
from lightsim2grid import LightSimBackend
from tqdm import tqdm
import pdb

if __name__ == "__main__":
    env = grid2op.make("educ_case14_storage",
                       test=True,
                       backend=LightSimBackend(),
                       action_class=PlayableAction)

    agent = OptimCVXPY(env.action_space,
                       env,
                       penalty_redispatching_unsafe=0.,
                       penalty_storage_unsafe=0.1,
                       penalty_curtailment_unsafe=0.01,
                       rho_safe=0.95,
                       rho_danger=0.97,
                       margin_th_limit=0.95,
                       alpha_por_error=0.5,
                       weight_redisp_target=0.3,
                       )

    # in safe / recovery mode agent tries to fill the storage units as much as possible
    agent.storage_setpoint = env.storage_Emax  

    # print("For do nothing: ")
    # dn_act = env.action_space()
    # for scen_id in range(7):
    #     env.set_id(scen_id)
    #     obs = env.reset()
    #     done = False
    #     for nb_step in tqdm(range(288)):
    #         obs, reward, done, info = env.step(dn_act)
    #         if done:
    #             break
    #     print(f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / 288")

    print("For the optimizer: ")
    for scen_id in range(7):
        # if scen_id != 2:
            # continue
        
        env.set_id(scen_id)
        obs = env.reset()
        agent.reset(obs)
        done = False
        for nb_step in tqdm(range(288)):
            prev_obs = obs
            # agent._DEBUG = nb_step >= 22
            # agent._DEBUG = nb_step >= 10
            # agent._DEBUG = nb_step >= 190
            act = agent.act(obs)
            obs, reward, done, info = env.step(act)
            if done:
                # print(info)
                # print(prev_obs.storage_charge)
                # print(prev_obs.target_dispatch)
                break
        print(f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / 288")
