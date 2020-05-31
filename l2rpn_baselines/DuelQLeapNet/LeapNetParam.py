# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from l2rpn_baselines.utils.TrainingParam import TrainingParam


class LeapNetParam(TrainingParam):
    _int_attr = TrainingParam._int_attr
    _float_attr = TrainingParam._float_attr

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
                 num_frames=1,
                 discount_factor=0.9,
                 tau=0.01,
                 update_freq=256
                 ):
        TrainingParam.__init__(self,
                               buffer_size=buffer_size,
                               minibatch_size=minibatch_size,
                               step_for_final_epsilon=step_for_final_epsilon,
                               min_observation=min_observation,
                               final_epsilon=final_epsilon,
                               initial_epsilon=initial_epsilon,
                               lr=lr,
                               lr_decay_steps=lr_decay_steps,
                               lr_decay_rate=lr_decay_rate,
                               num_frames=num_frames,
                               discount_factor=discount_factor,
                               tau=tau,
                               update_freq=update_freq)
