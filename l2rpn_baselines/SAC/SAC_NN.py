# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.


import random
import numpy as np
import pdb
import os

# tf2.0 friendly
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow.keras
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import load_model, Sequential, Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, subtract, add
    from tensorflow.keras.layers import Input, Lambda, Concatenate

import grid2op
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct



# This class implements the "Sof Actor Critic" model.
# It is a custom implementation, courtesy to Clement Goubet
# The original paper is: https://arxiv.org/abs/1801.01290
class SAC(RLQvalue):
    """Constructs the desired soft actor critic network"""

    def __init__(self, action_size, observation_size, lr=1e-5,
                 training_param=TrainingParam()):
        RLQvalue.__init__(self, action_size, observation_size, lr, training_param)
        # TODO add as meta param the number of "Q" you want to use (here 2)
        # TODO add as meta param size and types of the networks
        self.average_reward = 0
        self.life_spent = 1
        self.qvalue_evolution = np.zeros((0,))
        self.Is_nan = False

        self.model_value_target = None
        self.model_value = None
        self.model_Q = None
        self.model_Q2 = None
        self.model_policy = None

        self.construct_q_network()

    def _build_q_NN(self):
        input_states = Input(shape=(self.observation_size,))
        input_action = Input(shape=(self.action_size,))
        input_layer = Concatenate()([input_states, input_action])

        lay1 = Dense(self.observation_size)(input_layer)
        lay1 = Activation('relu')(lay1)

        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)

        lay3 = Dense(2 * self.action_size)(lay2)
        lay3 = Activation('relu')(lay3)

        advantage = Dense(1, activation='linear')(lay3)

        model = Model(inputs=[input_states, input_action], outputs=[advantage])
        model.compile(loss='mse', optimizer=Adam(lr=self.lr_))

        return model

    def _build_model_value(self):
        input_states = Input(shape=(self.observation_size,))
        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)

        lay3 = Dense(2 * self.action_size)(lay1)
        lay3 = Activation('relu')(lay3)
        advantage = Dense(self.action_size, activation='relu')(lay3)
        state_value = Dense(1, activation='linear')(advantage)
        model = Model(inputs=[input_states], outputs=[state_value])
        model.compile(loss='mse', optimizer=Adam(lr=self.lr_))
        return model

    def construct_q_network(self):
        # construct double Q networks
        self.model_Q = self._build_q_NN()
        self.model_Q2 = self._build_q_NN()

        # state value function approximation
        self.model_value = self._build_model_value()
        self.model_value_target = self._build_model_value()
        self.model_value_target.set_weights(self.model_value.get_weights())

        # policy function approximation
        self.model_policy = Sequential()
        # proba of choosing action a depending on policy pi
        input_states = Input(shape=(self.observation_size,))
        lay1 = Dense(self.observation_size)(input_states)
        lay1 = Activation('relu')(lay1)
        lay2 = Dense(self.observation_size)(lay1)
        lay2 = Activation('relu')(lay2)
        lay3 = Dense(2 * self.action_size)(lay2)
        lay3 = Activation('relu')(lay3)
        soft_proba = Dense(self.action_size, activation="softmax", kernel_initializer='uniform')(lay3)
        self.model_policy = Model(inputs=[input_states], outputs=[soft_proba])
        self.model_policy.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr_))

        print("Successfully constructed networks.")

    def predict_movement(self, data, epsilon):
        rand_val = np.random.random(data.shape[0])
        # q_actions = self.model.predict(data)
        p_actions = self.model_policy.predict(data)
        opt_policy_orig = np.argmax(np.abs(p_actions), axis=-1)
        opt_policy = 1.0 * opt_policy_orig
        opt_policy[rand_val < epsilon] = np.random.randint(0, self.action_size, size=(np.sum(rand_val < epsilon)))

        # store the qvalue_evolution (lots of computation time maybe here)
        tmp = np.zeros((data.shape[0], self.action_size))
        tmp[np.arange(data.shape[0]), opt_policy_orig] = 1.0
        q_actions0 = self.model_Q.predict([data, tmp])
        q_actions2 = self.model_Q2.predict([data, tmp])
        q_actions = np.fmin(q_actions0, q_actions2).reshape(-1)
        self.qvalue_evolution = np.concatenate((self.qvalue_evolution, q_actions))
        # above is not mandatory for predicting a movement so, might need to be moved somewhere else...

        opt_policy = opt_policy.astype(np.int)
        return opt_policy, p_actions[:, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains networks to fit given parameters"""
        batch_size = s_batch.shape[0]
        target = np.zeros((batch_size, 1))

        # training of the action state value networks
        last_action = np.zeros((batch_size, self.action_size))
        fut_action = self.model_value_target.predict(s2_batch).reshape(-1)
        target[:, 0] = r_batch + (1 - d_batch) * self.training_param.DECAY_RATE * fut_action
        loss = self.model_Q.train_on_batch([s_batch, last_action], target)
        loss_2 = self.model_Q2.train_on_batch([s_batch, last_action], target)

        self.life_spent += 1
        temp = 1 / np.log(self.life_spent) / 2
        tiled_batch = np.tile(s_batch, (self.action_size, 1))
        # tiled_batch: output something like: batch, batch, batch
        # TODO save that somewhere not to compute it each time, you can even save this in the
        # TODO tensorflow graph!
        tmp = np.repeat(np.eye(self.action_size), batch_size * np.ones(self.action_size, dtype=np.int), axis=0)
        # tmp is something like [1,0,0] (batch size times), [0,1,0,...] batch size time etc.

        action_v1_orig = self.model_Q.predict([tiled_batch, tmp]).reshape(batch_size, -1)
        action_v2_orig = self.model_Q2.predict([tiled_batch, tmp]).reshape(batch_size, -1)
        action_v1 = action_v1_orig - np.amax(action_v1_orig, axis=-1).reshape(batch_size, 1)
        new_proba = np.exp(action_v1 / temp) / np.sum(np.exp(action_v1 / temp), axis=-1).reshape(batch_size, 1)
        loss_policy = self.model_policy.train_on_batch(s_batch, new_proba)

        # training of the value_function
        target_pi = self.model_policy.predict(s_batch)
        value_target = np.fmin(action_v1_orig[0, a_batch], action_v2_orig[0, a_batch]) - np.sum(
            target_pi * np.log(target_pi + 1e-6))

        loss_value = self.model_value.train_on_batch(s_batch, value_target.reshape(-1, 1))

        self.Is_nan = np.isnan(loss) + np.isnan(loss_2) + np.isnan(loss_policy) + np.isnan(loss_value)
        # Print the loss every 100 iterations.
        if observation_num % 100 == 0:
            print("We had a loss equal to ", loss, loss_2, loss_policy, loss_value)
        return np.all(np.isfinite(loss)) & np.all(np.isfinite(loss_2)) & np.all(np.isfinite(loss_policy)) & \
               np.all(np.isfinite(loss_value))

    @staticmethod
    def _get_path_model(path, name=None):
        if name is None:
            path_model = path
        else:
            path_model = os.path.join(path, name)
        path_target_model = "{}_target".format(path_model)
        path_modelQ = "{}_Q".format(path_model)
        path_modelQ2 = "{}_Q2".format(path_model)
        path_policy = "{}_policy".format(path_model)
        return path_model, path_target_model, path_modelQ, path_modelQ2, path_policy

    def save_network(self, path, name=None, ext="h5"):
        # Saves model at specified path as h5 file
        path_model, path_target_model, path_modelQ, path_modelQ2, path_policy = self._get_path_model(path, name)
        self.model_value.save('{}.{}'.format(path_model, ext))
        self.model_value_target.save('{}.{}'.format(path_target_model, ext))
        self.model_Q.save('{}.{}'.format(path_modelQ, ext))
        self.model_Q2.save('{}.{}'.format(path_modelQ2, ext))
        self.model_policy.save('{}.{}'.format(path_policy, ext))
        print("Successfully saved network.")

    def load_network(self, path, name=None, ext="h5"):
        # nothing has changed
        path_model, path_target_model, path_modelQ, path_modelQ2, path_policy = self._get_path_model(path, name)
        self.model_value = load_model('{}.{}'.format(path_model, ext))
        self.model_value_target = load_model('{}.{}'.format(path_target_model, ext))
        self.model_Q = load_model('{}.{}'.format(path_modelQ, ext))
        self.model_Q2 = load_model('{}.{}'.format(path_modelQ2, ext))
        self.model_policy = load_model('{}.{}'.format(path_policy, ext))
        print("Succesfully loaded network.")

    def target_train(self):
        # nothing has changed from the original implementation
        model_weights = self.model_value.get_weights()
        target_model_weights = self.model_value_target.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = self.training_param.TAU * model_weights[i] + (1 - self.training_param.TAU) * \
                                      target_model_weights[i]
        self.model_value_target.set_weights(model_weights)
