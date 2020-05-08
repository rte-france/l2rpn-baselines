# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import numpy as np
from l2rpn_baselines.utils.TrainingParam import TrainingParam
from tensorflow.keras.models import load_model


# refactorization of the code in a base class to avoid copy paste.
class BaseDeepQ(object):
    """
    This class aims at representing the Q value (or more in case of SAC) parametrization by
    a neural network.

    It is composed of 2 different networks:
    - model: which is the main model
    - target_model: which has the same architecture and same initial weights as "model" but is updated less frequently
      to stabilize training

    It has basic methods to make predictions, to train the model, and train the target model.
    """

    def __init__(self,
                 action_size,
                 observation_size,
                 lr=1e-5,
                 training_param=TrainingParam()):
        # TODO add more flexibilities when building the deep Q networks, with a "NNParam" for example.
        self.action_size = action_size
        self.observation_size = observation_size
        self.lr_ = lr
        self.qvalue_evolution = np.zeros((0,))
        self.training_param = training_param

        self.model = None
        self.target_model = None

    def construct_q_network(self):
        raise NotImplementedError("Not implemented")

    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        rand_val = np.random.random(data.shape[0])
        q_actions = self.model.predict(data)
        opt_policy = np.argmax(np.abs(q_actions), axis=-1)
        opt_policy[rand_val < epsilon] = np.random.randint(0, self.action_size, size=(np.sum(rand_val < epsilon)))

        self.qvalue_evolution = np.concatenate((self.qvalue_evolution, q_actions[0, opt_policy]))
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        targets = self.model.predict(s_batch)
        fut_action = self.target_model.predict(s2_batch)
        targets[:, a_batch] = r_batch
        targets[d_batch, a_batch[d_batch]] += self.training_param.DECAY_RATE * np.max(fut_action[d_batch], axis=-1)

        loss = self.model.train_on_batch(s_batch, targets)
        # Print the loss every 100 iterations.
        if observation_num % 100 == 0:
            print("We had a loss equal to ", loss)
        return np.all(np.isfinite(loss))

    @staticmethod
    def _get_path_model(path, name=None):
        if name is None:
            path_model = path
        else:
            path_model = os.path.join(path, name)
        path_target_model = "{}_target".format(path_model)
        return path_model, path_target_model

    def save_network(self, path, name=None, ext="h5"):
        # Saves model at specified path as h5 file
        # nothing has changed
        path_model, path_target_model = self._get_path_model(path, name)
        self.model.save('{}.{}'.format(path_model, ext))
        self.target_model.save('{}.{}'.format(path_target_model, ext))
        print("Successfully saved network.")

    def load_network(self, path, name=None, ext="h5"):
        # nothing has changed
        path_model, path_target_model = self._get_path_model(path, name)
        self.model = load_model('{}.{}'.format(path_model, ext))
        self.target_model = load_model('{}.{}'.format(path_target_model, ext))
        print("Succesfully loaded network.")

    def target_train(self):
        # nothing has changed from the original implementation
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = self.training_param.TAU * model_weights[i] + (1 - self.training_param.TAU) * \
                                      target_model_weights[i]
        self.target_model.set_weights(target_model_weights)