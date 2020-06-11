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
    from tensorflow.keras.layers import Input

from l2rpn_baselines.utils import BaseDeepQ, TrainingParam


class DeepQ_NN(BaseDeepQ):
    """
    Constructs the desired deep q learning network

    Attributes
    ----------
    schedule_lr_model:
        The schedule for the learning rate.
    """

    def __init__(self,
                 nn_params,
                 training_param=None):
        if training_param is None:
            training_param = TrainingParam()
        BaseDeepQ.__init__(self,
                           nn_params,
                           training_param)
        self.schedule_lr_model = None
        self.construct_q_network()

    def construct_q_network(self):
        """
        The network architecture can be changed with the :attr:`l2rpn_baselines.BaseDeepQ.nn_archi`

        This function will make 2 identical models, one will serve as a target model, the other one will be trained
        regurlarly.
        """
        self._model = Sequential()
        input_layer = Input(shape=(self._nn_archi.observation_size,),
                            name="state")
        lay = input_layer
        for lay_num, (size, act) in enumerate(zip(self._nn_archi.sizes, self._nn_archi.activs)):
            lay = Dense(size, name="layer_{}".format(lay_num))(lay)  # put at self.action_size
            lay = Activation(act)(lay)

        output = Dense(self._action_size, name="output")(lay)

        self._model = Model(inputs=[input_layer], outputs=[output])
        self._schedule_lr_model, self._optimizer_model = self.make_optimiser()
        self._model.compile(loss='mse', optimizer=self._optimizer_model)

        self._target_model = Model(inputs=[input_layer], outputs=[output])
        self._target_model.set_weights(self._model.get_weights())