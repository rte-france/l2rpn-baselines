# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
from abc import ABC, abstractmethod
import numpy as np
import warnings
import tensorflow as tf
import tensorflow.keras.optimizers as tfko

from tensorflow.keras.models import load_model

from l2rpn_baselines.utils.TrainingParam import TrainingParam


# refactorization of the code in a base class to avoid copy paste.
class BaseDeepQ(ABC):
    """
    This class aims at representing the Q value (or more in case of SAC) parametrization by
    a neural network.

    It is composed of 2 different networks:

    - model: which is the main model
    - target_model: which has the same architecture and same initial weights as "model" but is updated less frequently
      to stabilize training

    It has basic methods to make predictions, to train the model, and train the target model.

    This class is abstraction and need to be overide in order to create object from this class. The only pure virtual
    function is :func:`BaseDeepQ.construct_q_network` that creates the neural network from the nn_params
    (:class:`NNParam`) provided as input

    Attributes
    ----------
    _action_size: ``int``
        Total number of actions

    _observation_size: ``int``
        Size of the observation space considered

    _nn_archi: :class:`NNParam`
        The parameters of the neural networks that will be created

    _training_param: :class:`TrainingParam`
        The meta parameters for the training scheme (used especially for learning rate or gradient clipping for example)

    _lr: ``float``
        The  initial learning rate

    _lr_decay_steps: ``float``
        The decay step of the learning rate

    _lr_decay_rate: ``float``
        The rate at which the learning rate will decay

    _model:
        Main neural network model, here a keras Model object.

    _target_model:
        a copy of the main neural network that will be updated less frequently (also known as "target model" in RL
        community)


    """

    def __init__(self,
                 nn_params,
                 training_param=None,
                 verbose=False):
        self._action_size = nn_params.action_size
        self._observation_size = nn_params.observation_size
        self._nn_archi = nn_params
        self.verbose = verbose

        if training_param is None:
            self._training_param = TrainingParam()
        else:
            self._training_param = training_param

        self._lr = training_param.lr
        self._lr_decay_steps = training_param.lr_decay_steps
        self._lr_decay_rate = training_param.lr_decay_rate

        self._model = None
        self._target_model = None
        self._schedule_model = None
        self._optimizer_model = None
        self._custom_objects = None  # to be able to load other keras layers type

    def make_optimiser(self):
        """
        helper function to create the proper optimizer (Adam) with the learning rates and its decay
        parameters.
        """
        schedule = tfko.schedules.InverseTimeDecay(self._lr, self._lr_decay_steps, self._lr_decay_rate)
        return schedule, tfko.Adam(learning_rate=schedule)

    @abstractmethod
    def construct_q_network(self):
        """
         Abstract method that need to be overide.

         It should create :attr:`BaseDeepQ._model` and :attr:`BaseDeepQ._target_model`
        """
        raise NotImplementedError("Not implemented")

    def predict_movement(self, data, epsilon, batch_size=None):
        """
        Predict movement of game controler where is epsilon probability randomly move."""
        if batch_size is None:
            batch_size = data.shape[0]

        rand_val = np.random.random(batch_size)
        q_actions = self._model.predict(data, batch_size=batch_size)

        opt_policy = np.argmax(np.abs(q_actions), axis=-1)
        opt_policy[rand_val < epsilon] = np.random.randint(0, self._action_size, size=(np.sum(rand_val < epsilon)))
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, tf_writer=None, batch_size=None):
        """Trains network to fit given parameters"""
        if batch_size is None:
            batch_size = s_batch.shape[0]

        # Save the graph just the first time
        if tf_writer is not None:
            tf.summary.trace_on()
        targets = self._model.predict(s_batch, batch_size=batch_size)
        if tf_writer is not None:
            with tf_writer.as_default():
                tf.summary.trace_export("model-graph", 0)
            tf.summary.trace_off()
        fut_action = self._target_model.predict(s2_batch, batch_size=batch_size)

        targets[:, a_batch.flatten()] = r_batch
        targets[d_batch, a_batch[d_batch]] += self._training_param.discount_factor * np.max(fut_action[d_batch], axis=-1)

        loss = self.train_on_batch(self._model, self._optimizer_model, s_batch, targets)
        return loss

    def train_on_batch(self, model, optimizer_model, x, y_true):
        """train the model on a batch of example. This can be overide"""
        loss = model.train_on_batch(x, y_true)
        return loss

    @staticmethod
    def get_path_model(path, name=None):
        """
        Get the location at which the neural networks will be saved.

        Returns
        -------
        path_model: ``str``
            The path at which the model will be saved (path include both path and name, it is the full path at which
            the neural networks are saved)

        path_target_model: ``str``
            The path at which the target model will be saved
        """
        if name is None:
            path_model = path
        else:
            path_model = os.path.join(path, name)
        path_target_model = "{}_target".format(path_model)
        return path_model, path_target_model

    def save_network(self, path, name=None, ext="h5"):
        """
        save the neural networks.

        Parameters
        ----------
        path: ``str``
            The path at which the models need to be saved
        name: ``str``
            The name given to this model

        ext: ``str``
            The file extension (by default h5)
        """
        # Saves model at specified path as h5 file
        # nothing has changed
        path_model, path_target_model = self.get_path_model(path, name)
        self._model.save('{}.{}'.format(path_model, ext))
        self._target_model.save('{}.{}'.format(path_target_model, ext))

    def load_network(self, path, name=None, ext="h5"):
        """
        Load the neural networks.
        Parameters
        ----------
        path: ``str``
            The path at which the models need to be saved
        name: ``str``
            The name given to this model

        ext: ``str``
            The file extension (by default h5)
        """
        path_model, path_target_model = self.get_path_model(path, name)
        self._model = load_model('{}.{}'.format(path_model, ext), custom_objects=self._custom_objects)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._target_model = load_model('{}.{}'.format(path_target_model, ext), custom_objects=self._custom_objects)
        if self.verbose:
            print("Succesfully loaded network.")

    def target_train(self):
        """
        update the target model with the parameters given in the :attr:`BaseDeepQ._training_param`.
        """
        # nothing has changed from the original implementation
        model_weights = self._model.get_weights()
        target_model_weights = self._target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = self._training_param.tau * model_weights[i] + (1 - self._training_param.tau) * \
                                      target_model_weights[i]
        self._target_model.set_weights(target_model_weights)