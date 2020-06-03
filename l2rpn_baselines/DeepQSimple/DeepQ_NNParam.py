# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import os

from l2rpn_baselines.utils import NNParam
from l2rpn_baselines.DeepQSimple.DeepQ_NN import DeepQ_NN


class DeepQ_NNParam(NNParam):
    _int_attr = NNParam._int_attr
    _float_attr = NNParam._float_attr
    _str_attr = NNParam._str_attr
    _list_float = NNParam._list_float
    _list_str = NNParam._list_str
    _list_int = NNParam._list_int

    nn_class = DeepQ_NN

    def __init__(self,
                 action_size,
                 observation_size,  # TODO this might not be usefull
                 sizes,
                 activs,
                 list_attr_obs
                 ):
        NNParam.__init__(self,
                         action_size,
                         observation_size,  # TODO this might not be usefull
                         sizes,
                         activs,
                         list_attr_obs
                         )