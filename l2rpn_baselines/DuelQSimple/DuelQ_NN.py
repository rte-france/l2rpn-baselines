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
                 action_size,
                 observation_size,
                 lr=0.00001,
                 learning_rate_decay_steps=1000,
                 learning_rate_decay_rate=0.95,
                 training_param=TrainingParam()):
        BaseDeepQ.__init__(self, action_size, observation_size, lr,
                           learning_rate_decay_steps=learning_rate_decay_steps,
                           learning_rate_decay_rate=learning_rate_decay_rate,
                           training_param=training_param)
        self.construct_q_network()

    def construct_q_network(self):
        # Uses the network architecture found in DeepMind paper
        # The inputs and outputs size have changed, as well as replacing the convolution by dense layers.
        self.model = Sequential()
        input_layer = Input(shape=(self.observation_size,),
                            name="observation")
        lay1 = Dense(self.observation_size)(input_layer)
        lay1 = Activation('relu')(lay1)

        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)

        lay3 = Dense(2 * self.action_size)(lay2)  # put at self.action_size
        lay3 = Activation('relu')(lay3)

        fc1 = Dense(self.action_size)(lay3)
        advantage = Dense(self.action_size)(fc1)
        fc2 = Dense(self.action_size)(lay3)
        value = Dense(1)(fc2)

        meaner = Lambda(lambda x: K.mean(x, axis=1))
        mn_ = meaner(advantage)
        tmp = subtract([advantage, mn_])
        policy = add([tmp, value], name="policy")

        self.model = Model(inputs=[input_layer], outputs=[policy])
        self.schedule_model, self.optimizer_model = self.make_optimiser()
        self.model.compile(loss='mse', optimizer=self.optimizer_model)

        self.target_model = Model(inputs=[input_layer], outputs=[policy])
        print("Successfully constructed networks.")
