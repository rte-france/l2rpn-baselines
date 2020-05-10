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
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Activation, Dense
    from tensorflow.keras.layers import Input

from l2rpn_baselines.utils import BaseDeepQ, TrainingParam


class DeepQ_NN(BaseDeepQ):
    """Constructs the desired deep q learning network"""

    def __init__(self,
                 action_size,
                 observation_size,
                 lr=1e-5,
                 learning_rate_decay_steps=1000,
                 learning_rate_decay_rate=0.95,
                 training_param=TrainingParam()):
        BaseDeepQ.__init__(self, action_size, observation_size,
                           lr, learning_rate_decay_steps, learning_rate_decay_rate,
                           training_param)
        self.construct_q_network()

    def construct_q_network(self):
        # replacement of the Convolution layers by Dense layers, and change the size of the input space and output space

        # Uses the network architecture found in DeepMind paper
        self.model = Sequential()
        input_layer = Input(shape=(self.observation_size * self.training_param.NUM_FRAMES,),
                            name="state")
        layer1 = Dense(self.observation_size * self.training_param.NUM_FRAMES)(input_layer)
        layer1 = Activation('relu')(layer1)
        layer2 = Dense(self.observation_size)(layer1)
        layer2 = Activation('relu')(layer2)
        layer3 = Dense(self.observation_size)(layer2)
        layer3 = Activation('relu')(layer3)
        layer4 = Dense(2 * self.action_size)(layer3)
        layer4 = Activation('relu')(layer4)
        output = Dense(self.action_size, name="output")(layer4)

        self.model = Model(inputs=[input_layer], outputs=[output])
        self.schedule_lr_model, self.optimizer_model = self.make_optimiser()
        self.model.compile(loss='mse', optimizer=self.optimizer_model)

        self.target_model = Model(inputs=[input_layer], outputs=[output])
        # self.target_model.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        self.target_model.set_weights(self.model.get_weights())