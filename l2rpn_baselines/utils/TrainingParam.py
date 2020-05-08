# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.


class TrainingParam(object):
    """
    A class to store the training parameters of the models. It was hard coded in the getting_started/notebook 3
    of grid2op and put in this repository instead.
    """

    def __init__(self,
                 DECAY_RATE=0.9,
                 BUFFER_SIZE=40000,
                 MINIBATCH_SIZE=64,
                 TOT_FRAME=3000000,
                 EPSILON_DECAY=10000,
                 MIN_OBSERVATION=50,  # 5000
                 FINAL_EPSILON=1 / 300,  # have on average 1 random action per scenario of approx 287 time steps
                 INITIAL_EPSILON=0.1,
                 TAU=0.01,
                 ALPHA=1,
                 NUM_FRAMES=1,
                 ):

        self.DECAY_RATE = DECAY_RATE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.TOT_FRAME = TOT_FRAME
        self.EPSILON_DECAY = EPSILON_DECAY
        self.MIN_OBSERVATION = MIN_OBSERVATION  # 5000
        self.FINAL_EPSILON = FINAL_EPSILON  # have on average 1 random action per scenario of approx 287 time steps
        self.INITIAL_EPSILON = INITIAL_EPSILON
        self.TAU = TAU
        self.NUM_FRAMES = NUM_FRAMES
        self.ALPHA = ALPHA