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

try:
    import tensorflow as tf
    import tensorflow.keras.optimizers as tfko
    _CAN_USE_TENSORFLOW = True
except ImportError:
    _CAN_USE_TENSORFLOW = False
    


from l2rpn_baselines.utils.trainingParam import TrainingParam


# refactorization of the code in a base class to avoid copy paste.
class BaseDeepQ(ABC):
    """
    This class aims at representing the Q value (or more in case of SAC) parametrization by
    a neural network.

    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
        Prefer to use the :class:`GymAgent` class and the :class:`GymEnvWithHeuristics`
        classes to train agent interacting with grid2op and fully compatible
        with gym framework.	
        
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
        if not _CAN_USE_TENSORFLOW:
            raise RuntimeError("Cannot import tensorflow, this function cannot be used.")
        
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

    def predict_movement(self, data, epsilon, batch_size=None, training=False):
        """
        Predict movement of game controler where is epsilon probability randomly move.
        """
        if batch_size is None:
            batch_size = data.shape[0]

        # q_actions = self._model.predict(data, batch_size=batch_size)  # q value of each action
        q_actions = self._model(data, training=training).numpy()
        opt_policy = np.argmax(q_actions, axis=-1)
        if epsilon > 0.:
            rand_val = np.random.random(batch_size)
            opt_policy[rand_val < epsilon] = np.random.randint(0, self._action_size, size=(np.sum(rand_val < epsilon)))
        return opt_policy, q_actions[np.arange(batch_size), opt_policy], q_actions

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, tf_writer=None, batch_size=None):
        """
        Trains network to fit given parameters:
        
        .. seealso::
            https://towardsdatascience.com/dueling-double-deep-q-learning-using-tensorflow-2-x-7bbbcec06a2a
            for the update rules
        
        Parameters
        ----------
        s_batch:
            the state vector (before the action is taken)
        a_batch:
            the action taken
        s2_batch:
            the state vector (after the action is taken)
        d_batch:
            says whether or not the episode was over
        r_batch:
            the reward obtained this step
        """
        if batch_size is None:
            batch_size = s_batch.shape[0]

        # Save the graph just the first time
        if tf_writer is not None:
            tf.summary.trace_on()
        target = self._model(s_batch, training=True).numpy()
        fut_action = self._model(s2_batch, training=True).numpy()
        if tf_writer is not None:
            with tf_writer.as_default():
                tf.summary.trace_export("model-graph", 0)
            tf.summary.trace_off()
        target_next = self._target_model(s2_batch, training=True).numpy()

        idx = np.arange(batch_size)
        target[idx, a_batch] = r_batch
        # update the value for not done episode
        nd_batch = ~d_batch  # update with this rule only batch that did not game over
        next_a = np.argmax(fut_action, axis=-1)  # compute the future action i will take in the next state
        fut_Q = target_next[idx, next_a]  # get its Q value
        target[nd_batch, a_batch[nd_batch]] += self._training_param.discount_factor * fut_Q[nd_batch]
        loss = self.train_on_batch(self._model, self._optimizer_model, s_batch, target)
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
        # fix for issue https://github.com/keras-team/keras/issues/7440
        self.construct_q_network()

        self._model.load_weights('{}.{}'.format(path_model, ext))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._target_model.load_weights('{}.{}'.format(path_target_model, ext))
        if self.verbose:
            print("Succesfully loaded network.")

    def target_train(self, tau=None):
        """
        update the target model with the parameters given in the :attr:`BaseDeepQ._training_param`.
        """
        if tau is None:
            tau = self._training_param.tau
        tau_inv = 1.0 - tau

        target_params = self._target_model.trainable_variables
        source_params = self._model.trainable_variables
        for src, dest in zip(source_params, target_params):
            # Polyak averaging
            var_update = src.value() * tau
            var_persist = dest.value() * tau_inv
            dest.assign(var_update + var_persist)

    def save_tensorboard(self, current_step):
        """function used to save other information to tensorboard"""
        pass
