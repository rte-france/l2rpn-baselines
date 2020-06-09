# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

# tf2.0 friendly
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Activation, Dense
    from tensorflow.keras.layers import Input, Lambda, subtract, add
    import tensorflow.keras.backend as K

from l2rpn_baselines.utils import BaseDeepQ, TrainingParam


class DuelQ_NN(BaseDeepQ):
    """Constructs the desired duelling deep q learning network"""
    def __init__(self,
                 nn_params,
                 training_param=None):
        if training_param is None:
            training_param = TrainingParam()
        BaseDeepQ.__init__(self,
                           nn_params,
                           training_param)
        self.construct_q_network()

    def construct_q_network(self):
        """
        It uses the architecture defined in the `nn_archi` attributes.

        """
        self._model = Sequential()
        input_layer = Input(shape=(self._nn_archi.observation_size,),
                            name="observation")

        lay = input_layer
        for lay_num, (size, act) in enumerate(zip(self._nn_archi.sizes, self._nn_archi.activs)):
            lay = Dense(size, name="layer_{}".format(lay_num))(lay)  # put at self.action_size
            lay = Activation(act)(lay)

        fc1 = Dense(self._action_size)(lay)
        advantage = Dense(self._action_size, name="advantage")(fc1)

        fc2 = Dense(self._action_size)(lay)
        value = Dense(1, name="value")(fc2)

        meaner = Lambda(lambda x: K.mean(x, axis=1))
        mn_ = meaner(advantage)
        tmp = subtract([advantage, mn_])
        policy = add([tmp, value], name="policy")

        self._model = Model(inputs=[input_layer], outputs=[policy])
        self._schedule_model, self._optimizer_model = self.make_optimiser()
        self._model.compile(loss='mse', optimizer=self._optimizer_model)

        self._target_model = Model(inputs=[input_layer], outputs=[policy])
