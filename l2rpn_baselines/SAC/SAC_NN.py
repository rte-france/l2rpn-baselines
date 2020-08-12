# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow.keras.optimizers as tfko
import tensorflow.keras.models as tfkm
import tensorflow_probability as tfp
import tensorflow_probability.distributions as tfpd

from l2rpn_baselines.utils import BaseDeepQ, TrainingParam

class SAC_NN(BaseDeepQ):
    """
    Constructs the desired soft actor critic network.

    References
    -----------
    Original paper:
    https://arxiv.org/abs/1801.01290

    Follow up paper (learned entropy):
    https://arxiv.org/abs/1812.05905

    Discrete action space mofifications:
    https://arxiv.org/abs/1910.07207

    Reparametrization:
    https://arxiv.org/abs/1611.01144
    https://arxiv.org/abs/1611.00712

    """
    def __init__(self,
                 nn_params,
                 training_param=None,
                 verbose=False):
        if training_param is None:
            training_param = TrainingParam()
        BaseDeepQ.__init__(self,
                           nn_params,
                           training_param,
                           verbose=verbose)

        self.log_std_min = -20
        self.log_std_max = 2
        self.tau = 1e-3
        self.gamma = 0.99
        self.alpha = 0.2

        self.construct_q_network()

    def _build_q_NN(self, model_name):
        input_state = Input(shape=(self._observation_size,))
        input_action = Input(shape=(self._action_size,))

        input_layer = tf.concat([input_state, input_action], axis=-1)

        lay = input_layer
        lay_size_activ = zip(self._nn_archi.q_sizes,
                             self._nn_archi.q_activs)
        for lay_id, (size, act) in enumerate(lay_size_activs):
            lay_name = "{}_fc_{}".format(model_name, lay_id)
            act_name = "{}_act_{}".format(model_name, lay_id)
            lay = tfkl.Dense(size, name=lay_name)(lay)
            if act is not None:
                lay = tfkl.Activation(act, name=act_name)(lay)

        final_name = "{}_final".format(model_name)
        Q = Dense(1, name=final_name)(lay)

        model_inputs = [input_state, input_action]
        model_outputs = [Q]
        model = tfk.Model(inputs=model_input,
                          outputs=model_outputs,
                          name=model_name)
        return model

    def _build_policy_NN(self, model_name, epsilon=1e-8):
        input_state = Input(shape=(self._observation_size,))

        lay = input_state
        lay_size_activ = zip(self._nn_archi.policy_sizes,
                             self._nn_archi.policy_activs)
        for lay_id, (size, act) in enumerate(lay_size_activs):
            lay_name = "{}_fc_{}".format(model_name, lay_id)
            act_name = "{}_act_{}".format(model_name, lay_id)
            lay = tfkl.Dense(size, name=lay_name)(lay)
            if act is not None:
                lay = tfkl.Activation(act, name=act_name)(lay)

        mean_name = "{}_fc_mean".format(model_name)
        mean = tfkl.Dense(self._action_size, name=mean_name)(lay)
        log_std_name = "{}_log_std".format(model_name)
        log_std = tfkl.Dense(self._action_size, name=log_std_name)(lay)
        log_std = tf.clip_by_value(log_std,
                                   self.log_std_min,
                                   self.log_std_max)

        model_inputs = [input_state]
        model_outputs = [mean, log_std]
        model = tfk.Model(inputs=model_inputs,
                          outputs=model_outputs,
                          name=model_name)

    def construct_q_network(self):
        """
        This constructs all the networks needed for the SAC agent.
        """
        self.q1 = self._build_q_NN(name="q1")
        self.q1_schedule, self.q1_opt = self.make_optimiser()
        self.q1.compile(optimizer=self.q1_opt)

        self.target_q1 = self._build_q_NN(name="target-q1")
        self.target_q1_schedule, self.target_q1_opt = self.make_optimiser()
        self.target_q1.compile(optimizer=self.target_q1_opt)

        self.q2 = self._build_q_NN(name="q2")
        self.q2_schedule, self.q2_opt = self.make_optimiser()
        self.q2.compile(optimizer=self.q2_opt)

        self.target_q2 = self._build_q_NN(name="target-q2")
        self.target_q2_schedule, self.target_q2_opt = self.make_optimiser()
        self.target_q2.compile(optimizer=self.target_q2_opt)

        self.policy = self._build_policy_NN(name="policy")
        self.policy_schedule, self.policy_opt = self.make_optimizer()
        self.policy.compile(optimizer=self.policy_opt)

        self.target_entropy = -(self._action_size * 1.0)
        self.log_alpha = tf.zeros(1)
        self.alpha_schedule, self.alpha_opt = self.make_optimizer()


    def sample(self, state):
        p_actions = self.policy(state)

        pd_actions = tfpd.RelaxedOneHotCategorical(self.alpha, probs=p_actions)
        actions = pd_actions.sample()
        log_actions = pd_actions.log_prob(actions)

        return actions, p_actions, log_actions

    def predict_movement(self, data, epsilon,
                         batch_size=None,
                         training=False):
        """
        Predict the next action
        """
        actions, _, _ = self.sample(data)
        return policy

    def train(self,
              s_batch, a_batch, r_batch, d_batch, s2_batch,
              tf_writer=None,
              batch_size=None):
        """Trains networks to fit given parameters"""

        # Compute Q target
        a_next, log_next = self.sample(s2_batch)
        q1_next = self.target_q1([s2_batch, a_next], training=True)
        q2_next = self.target_q2([s2_batch, a_next], training=True)
        q_next = tf.math.minimum(q1_next, q2_next) - self.alpha * log_next
        q_target = r_batch + (1.0 - d_batch) * self.gamma * q_next

        # Train Q1
        with tf.GradientTape() as q1_tape:
            # Compute loss under gradient tape
            q1 = self.q1([s_batch, a_batch])
            q1_sq_td_error = tf.square(q_target - q1)
            q1_loss = tf.reduce_mean(q1_sq_td_error, axis=0)

        # Q1 Compute & Apply gradients
        q1_vars = self.q1.trainable_variables
        q1_grads = q1_tape.gradient(q1_loss, q1_vars)
        self.q1_opt.apply_gradients(zip(q1_grads, q1_vars))

        # Train Q2
        with tf.GradientTape() as q2_tape:
            q2 = self.q2([s_batch, a_batch])
            q2_sq_td_error = tf.square(q_target - q2)
            q2_loss = tf.reduce_mean(q2_sq_td_error, axis=0)

        # Q2 Compute & Apply gradients
        q2_vars = self.q2.trainable_variables
        q2_grads = q2_tape.gradient(q2_loss, q2_vars)
        self.q2_opt.apply_gradients(zip(q2_grads, q2_vars))

        # Policy train
        with tf.GradientTape() as policy_tape:
            a_new, _, log_new = self.sample(s_batch)
            q1_new = self.q1([s_batch, a_new])
            q2_new = self.q2([s_batch, a_new])
            q_new = tf.math.minimum(q1_new, q2_new)
            policy_loss = self.alpha * log_new - min_q
            policy_loss = tf.reduce_mean(policy_loss, axis=0)

        # Policy compute & apply gradients
        policy_vars = self.policy.trainable_variables
        policy_grads = policy_tape.gradient(policy_loss, policy_vars)
        self.policy_opt.apply_gradients(zip(policy_grads, policy_vars))

        # Q Target networks soft update
        self.soft_target_update(self.q1, self.q1_target, tau=self.tau)
        self.soft_target_update(self.q2, self.q2_target, tau=self.tau)

        # Entropy train
        with tf.GradientTape() as entropy_tape:
            alpha_loss = self.log_alpha * (-log_new - self.target_entropy)
            alpha_loss = tf.reduce_mean(alpha_loss)

        # Entropy compute & apply gradients
        alpha_vars = self.log_alpha.trainable_variables
        alpha_grads = entropy_tape.gradient(alpha_loss, alpha_vars)
        self.alpha_opt.apply_gradients(zip(alpha_grads, alpha_vars))

        # Update entropy
        self.alpha = tf.math.exp(self.log_alpha)

        # Get losses values from GPU
        q1l = q1_loss.numpy()
        q2l = q2_loss.numpy()
        policyl = policy_loss.numpy()
        alphal = alpha_loss.numpy()
        # Return them
        return q1l, q2l, policyl, alphal
    

    @staticmethod
    def model_paths(path, name, ext):
        if name is None:
            model_dir = path
        else:
            model_dir = os.path.join(path, name)

        q1_path = os.path.join(model_dir, "q1.{}".format(ext))
        q1_path = os.path.abspath(q1_path)
        
        q2_path = os.path.join(model_dir, "q2.{}".format(ext))
        q2_path = os.path.abspath(q2_path)
        
        policy_path = os.path.join(model_dir, "policy.{}".format(ext))
        policy_path = os.path.abspath(policy_path)

        return  model_dir, q1_path, q2_path, policy_path
        
    def save_network(self, path, name=None, ext="h5"):
        """
        Saves all the models network weigths
        """
        model_paths = self.model_paths(path, name, ext)
        model_dir = model_paths[0]
        q1_path = model_paths[1]
        q2_path = model_paths[2]
        policy_path = model_paths[3]
        
        os.makedirs(path_dir, exist_ok=True)

        self.q1.save(q1_p)
        self.q2.save(q2_p)
        self.policy.save(policy_p)

    def load_network(self, path, name=None, ext="h5"):
        """
        Loqd the models networks weights from disk
        """

        model_paths = self.model_paths(path, name, ext)
        model_dir = model_paths[0]
        q1_path = model_paths[1]
        q2_path = model_paths[2]
        policy_path = model_paths[3]
        
        self.q1.load_weights(q1_path)
        self.q2.load_weights(q2_path)
        self.policy.load_weights(policy_path)

        self.hard_target_update(self.q1, self.target_q1)
        self.hard_target_update(self.q2, self.target_q2)

        if self.verbose:
            print("Succesfully loaded network.")

    @staticmethod
    def soft_target_update(source, target, tau=1e-3):
        tau_inv = 1.0 - tau

        source_params = source.trainable_variables
        target_params = target.trainable_variables
        
        for src, dest in zip(source_params, target_params):
            # Polyak averaging
            var_update = src.value() * tau
            var_persist = dest.value() * tau_inv
            dest.assign(var_update + var_persist)

    @staticmethod
    def hard_target_update(source, target):
        source_params = source.trainable_variables
        target_params = target.trainable_variables
        
        for src, dest in zip(source_params, target_params):
            dest.assign(src.value())
