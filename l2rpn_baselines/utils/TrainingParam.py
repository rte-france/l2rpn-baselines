# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import numpy as np


class TrainingParam(object):
    """
    A class to store the training parameters of the models. It was hard coded in the getting_started/notebook 3
    of grid2op and put in this repository instead.
    """

    def __init__(self,
                 DECAY_RATE=0.9,
                 BUFFER_SIZE=40000,
                 MINIBATCH_SIZE=64,
                 STEP_FOR_FINAL_EPSILON=100000,  # step at which min_espilon is obtain
                 MIN_OBSERVATION=5000,  # 5000
                 FINAL_EPSILON=1./(7*288.),  # have on average 1 random action per week of approx 7*288 time steps
                 INITIAL_EPSILON=0.4,
                 TAU=0.01,
                 ALPHA=1,
                 NUM_FRAMES=1,
                 ):

        self.DECAY_RATE = DECAY_RATE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.MIN_OBSERVATION = MIN_OBSERVATION  # 5000
        self.FINAL_EPSILON = float(FINAL_EPSILON)  # have on average 1 random action per day of approx 288 timesteps at the end (never kill completely the exploration)
        self.INITIAL_EPSILON = float(INITIAL_EPSILON)
        self.STEP_FOR_FINAL_EPSILON = float(STEP_FOR_FINAL_EPSILON)
        self.TAU = TAU
        self.NUM_FRAMES = NUM_FRAMES
        self.ALPHA = ALPHA

        self._exp_facto = np.log(self.INITIAL_EPSILON/self.FINAL_EPSILON)

    def get_next_epsilon(self, current_step):
        if current_step > self.STEP_FOR_FINAL_EPSILON:
            res = self.FINAL_EPSILON
        else:
            # exponential decrease
            res = self.INITIAL_EPSILON * np.exp(- (current_step / self.STEP_FOR_FINAL_EPSILON) * self._exp_facto )
        return res

