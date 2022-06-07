# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import sys
import logging
import grid2op
from l2rpn_baselines.OptimCVXPY import OptimCVXPY
from lightsim2grid import LightSimBackend
from tqdm import tqdm
import pdb

env_name = "l2rpn_wcci_2022"
is_test = False

env = grid2op.make(env_name,
                   test=is_test,
                   backend=LightSimBackend()
                   )
logger = None

agent = OptimCVXPY(env.action_space,
                   env,
                   penalty_redispatching_unsafe=0.,
                   penalty_storage_unsafe=0.01,
                   penalty_curtailment_unsafe=0.01,
                   penalty_curtailment_safe=0.1,
                   penalty_redispatching_safe=0.1,
                   logger=logger
                   )

scen_test = ["2050-01-03_31",
             "2050-02-21_31",
             "2050-03-07_31",
             "2050-04-18_31",
             "2050-05-09_31",
             "2050-06-27_31",
             "2050-07-25_31",
             "2050-08-01_31",
             "2050-09-26_31",
             "2050-10-03_31",
             "2050-11-14_31",
             "2050-12-19_31",
             ]

print("For do nothing: ")
dn_act = env.action_space()
for scen_id in scen_test:
    env.set_id(scen_id)
    obs = env.reset()
    done = False
    for nb_step in tqdm(range(obs.max_step)):
        prev_obs = obs
        obs, reward, done, info = env.step(dn_act)
        if done and (nb_step != prev_obs.max_step - 1):
            break
    print(f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / {obs.max_step}")

print("For the optimizer: ")
for scen_id in scen_test:
    act = None
    env.set_id(scen_id)
    env.seed(0)
    obs = env.reset()
    agent.reset(obs)
    done = False
    for nb_step in tqdm(range(obs.max_step)):
        prev_obs = obs
        prev_act = act
        act = agent.act(obs)
        obs, reward, done, info = env.step(act)
        if done and (nb_step != prev_obs.max_step - 1):
            break
    print(f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / {obs.max_step}")
