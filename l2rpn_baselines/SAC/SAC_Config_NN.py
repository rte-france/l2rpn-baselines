# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
from l2rpn_baselines.utils import BaseConfig

class SAC_Config_NN(BaseConfig):

    _float_attr = [
        "log_std_min", "log_std_max",
        "tau", "alpha", "gamma",
        "lr_critic", "lr_policy", "lr_alpha"
    ]
    _list_str = ["activations_critic", "activations_policy"]
    _list_int = ["sizes_critic", "sizes_policy"]

    def __init__(self):
        super().__init__()

        # Set some defaults
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        self.gamma = 0.99
        self.tau = 1e-3
        self.alpha = 0.2
        self.lr_critic = 1e-4
        self.lr_policy = 1e-4
        self.lr_alpha = 1e-4
        self.sizes_critic = [512, 512, 256, 256, 256, 1]
        self.activations_critic = ["relu"] * 5
        self.sizes_policy = [512, 512, 256, 256, 256, 1]
        self.activations_policy = ["relu"] * 5

if __name__ == "__main__":
    conf = SAC_Config_NN()
    conf.to_json_file("SAC_Config_NN.default.json")
