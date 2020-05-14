# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import os
import json
import numpy as np


class TrainingParam(object):
    """
    A class to store the training parameters of the models. It was hard coded in the getting_started/notebook 3
    of grid2op and put in this repository instead.
    """
    __int_attr = ["buffer_size", "minibatch_size", "step_for_final_epsilon", "min_observation", "last_step"]
    __float_attr = ["final_epsilon", "initial_epsilon", "lr", "lr_decay_steps", "lr_decay_rate"]

    def __init__(self,
                 buffer_size=40000,
                 minibatch_size=64,
                 step_for_final_epsilon=100000,  # step at which min_espilon is obtain
                 min_observation=5000,  # 5000
                 final_epsilon=1./(7*288.),  # have on average 1 random action per week of approx 7*288 time steps
                 initial_epsilon=0.4,
                 lr=1e-4,
                 lr_decay_steps=10000,
                 lr_decay_rate=0.999,
                 TAU=0.01,
                 ALPHA=1,
                 NUM_FRAMES=1,
                 DECAY_RATE=0.9,
                 ):

        # self.DECAY_RATE = DECAY_RATE
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.min_observation = min_observation  # 5000
        self.final_epsilon = float(final_epsilon)  # have on average 1 random action per day of approx 288 timesteps at the end (never kill completely the exploration)
        self.initial_epsilon = float(initial_epsilon)
        self.step_for_final_epsilon = float(step_for_final_epsilon)
        # self.TAU = TAU
        # self.NUM_FRAMES = NUM_FRAMES
        # self.ALPHA = ALPHA
        self.lr = lr
        self.lr_decay_steps = float(lr_decay_steps)
        self.lr_decay_rate = float(lr_decay_rate)
        self.last_step = 0

        self._exp_facto = np.log(self.initial_epsilon/self.final_epsilon)

    def tell_step(self, current_step):
        self.last_step = current_step

    def get_next_epsilon(self, current_step):
        self.last_step = current_step
        if current_step > self.step_for_final_epsilon:
            res = self.final_epsilon
        else:
            # exponential decrease
            res = self.initial_epsilon * np.exp(- (current_step / self.step_for_final_epsilon) * self._exp_facto )
        return res

    def to_dict(self):
        res = {}
        for attr_nm in self.__int_attr:
            res[attr_nm] = int(getattr(self, attr_nm))
        for attr_nm in self.__float_attr:
            res[attr_nm] = float(getattr(self, attr_nm))
        return res

    @staticmethod
    def from_dict(tmp):
        res = TrainingParam()
        for attr_nm in TrainingParam.__int_attr:
            if attr_nm in tmp:
                setattr(res, attr_nm, int(tmp[attr_nm]))

        for attr_nm in TrainingParam.__float_attr:
            if attr_nm in tmp:
                setattr(res, attr_nm, float(tmp[attr_nm]))

        res._exp_facto = np.log(res.initial_epsilon / res.final_epsilon)
        return res

    @staticmethod
    def from_json(json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError("No path are located at \"{}\"".format(json_path))
        with open(json_path, "r") as f:
            dict_ = json.load(f)
        return TrainingParam.from_dict(dict_)

    def save_as_json(self, path, name=None):
        res = self.to_dict()
        if name is None:
            name = "training_parameters.json"
        if not os.path.exists(path):
            raise RuntimeError("Directory \"{}\" not found to save the training parameters".format(path))
        if not os.path.isdir(path):
            raise NotADirectoryError("\"{}\" should be a directory".format(path))
        path_out = os.path.join(path, name)
        with open(path_out, "w", encoding="utf-8") as f:
            json.dump(res, fp=f, indent=4, sort_keys=True)