# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from l2rpn_baselines.utils import BaseConfig

class SAC_Config_Train(BaseConfig):

    _int_attr = [
        "replay_buffer_size",
        "min_replay_buffer_size",
        "batch_size",
        "update_freq",
        "save_freq",
        "log_freq"
    ]

    _bool_attr = [
        "cv"
    ]
    
    def __init__(self):
        super().__init__()

        # Set some defaults
        self.replay_buffer_size = int(1e6)
        self.min_replay_buffer_size = 256
        self.update_freq = 1
        self.save_freq = 10000
        self.log_freq = 100
        self.batch_size = 128
        self.cv = True

if __name__ == "__main__":
    conf = SAC_Config_Train()
    conf.to_json_file("SAC_Config_Train.default.json")
