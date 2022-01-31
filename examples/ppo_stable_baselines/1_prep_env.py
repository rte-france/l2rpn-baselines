# Copyright (c) 2020-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

# this file needs to be run only once, it might take a while !

import grid2op
from grid2op.dtypes import dt_int
from grid2op.utils import ScoreICAPS2021
from lightsim2grid import LightSimBackend
import numpy as np


env_name = "l2rpn_icaps_2021_small"
nb_process_stats = 8
verbose = 1

# create the environment
env = grid2op.make(env_name)

# split into train / val / test
# it is such that there are 25 chronics for val and 24 for test
env.seed(1)
env.reset()
nm_train, nm_test, nm_val = env.train_val_split_random(add_for_test="test",
                                                       pct_val=4.2,
                                                       pct_test=4.2)

# computes some statistics for val / test to compare performance of 
# some agents with the do nothing for example
max_int = max_int = np.iinfo(dt_int).max
for nm_ in [nm_val, nm_test]:
    env_tmp = grid2op.make(nm_, backend=LightSimBackend())
    nb_scenario = len(env_tmp.chronics_handler.subpaths)
    print(f"{nm_}: {nb_scenario}")
    my_score = ScoreICAPS2021(env_tmp,
                              nb_scenario=nb_scenario,
                              env_seeds=np.random.randint(low=0,
                                                          high=max_int,
                                                          size=nb_scenario,
                                                          dtype=dt_int),
                              agent_seeds=[0 for _ in range(nb_scenario)],
                              verbose=verbose,
                              nb_process_stats=nb_process_stats,
                              )

# my_agent = DoNothingAgent(env.action_space)
# print(my_score.get(my_agent))