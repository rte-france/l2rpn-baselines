# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import json

class DoubleDuelingRDQNConfig():
    """
    DoubleDuelingRDQN configurable hyperparameters as class attributes
    
    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
    """

    INITIAL_EPSILON = 0.99
    FINAL_EPSILON = 0.01
    DECAY_EPSILON = 1024*32
    STEP_EPSILON = (INITIAL_EPSILON-FINAL_EPSILON)/DECAY_EPSILON
    DISCOUNT_FACTOR = 0.99
    REPLAY_BUFFER_SIZE = 1024*4
    UPDATE_FREQ = 64
    UPDATE_TARGET_HARD_FREQ = -1
    UPDATE_TARGET_SOFT_TAU = 0.001
    TRACE_LENGTH = 8
    BATCH_SIZE = 32
    LR = 1e-5
    VERBOSE = True

    @staticmethod
    def from_json(json_in_path):
        with open(json_in_path, 'r') as fp:
            conf_json = json.load(fp)
        
        for k,v in conf_json.items():
            if hasattr(DoubleDuelingDQNConfig, k):
                setattr(DoubleDuelingDQNConfig, k, v)

    @staticmethod
    def to_json(json_out_path):
        conf_json = {}
        for attr in dir(DoubleDuelingDQNConfig):
            if attr.startswith('__') or callable(attr):
                continue
            conf_json[attr] = getattr(DoubleDuelingDQNConfig, attr)

        with open(json_out_path, 'w+') as fp:
            json.dump(fp, conf_json, indent=2)
