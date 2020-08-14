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
from tensorflow_probability import distributions as tfpd

from l2rpn_baselines.utils import BaseDeepQ, TrainingParam

class SAC_NN(object):
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
                 observation_shape,
                 action_shape,
                 nn_config,
                 training=False,
                 verbose=True):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self._cfg = nn_config
        self.verbose = verbose
        
        self.training = training
        self.alpha = 0.01
        if self.training:
            self.alpha = self._cfg.alpha
        
        self.construct_q_network()

    def _build_q_NN(self, model_name):
        input_state = tfkl.Input(shape=self.observation_shape)
        input_action = tfkl.Input(shape=self.action_shape)

        input_action_flat = tfkl.Flatten()(input_action)
        input_layer = tf.concat([input_state, input_action_flat], axis=-1)

        lay = input_layer
        lay_size_activ = zip(self._cfg.sizes_critic,
                             self._cfg.activations_critic)
        for lay_id, (size, act) in enumerate(lay_size_activ):
            lay_name = "{}_fc_{}".format(model_name, lay_id)
            act_name = "{}_act_{}".format(model_name, lay_id)
            lay = tfkl.Dense(size, name=lay_name)(lay)
            if act is not None:
                lay = tfkl.Activation(act, name=act_name)(lay)

        final_name = "{}_final".format(model_name)
        Q = tfkl.Dense(1, name=final_name)(lay)

        model_inputs = [input_state, input_action]
        model_outputs = [Q]
        model = tfk.Model(inputs=model_inputs,
                          outputs=model_outputs,
                          name=model_name)
        return model

    def _build_policy_NN(self, model_name, epsilon=1e-8):
        input_state = tfkl.Input(shape=self.observation_shape)

        lay = input_state
        lay_size_activ = zip(self._cfg.sizes_policy,
                             self._cfg.activations_policy)
        for lay_id, (size, act) in enumerate(lay_size_activ):
            lay_name = "{}_fc_{}".format(model_name, lay_id)
            act_name = "{}_act_{}".format(model_name, lay_id)
            lay = tfkl.Dense(size, name=lay_name)(lay)
            if act is not None:
                lay = tfkl.Activation(act, name=act_name)(lay)

        action_size = np.prod(self.action_shape)
        logits_name = "{}_fc_logits".format(model_name)
        logits = tfkl.Dense(action_size, name=logits_name)(lay)

        logits_2d = tf.reshape(logits, (-1,) + self.action_shape)
        probs = tf.nn.softmax(logits_2d, axis=-1)

        model_inputs = [input_state]
        model_outputs = [probs]
        model = tfk.Model(inputs=model_inputs,
                          outputs=model_outputs,
                          name=model_name)
        return model

    def construct_q_network(self):
        """
        This constructs all the networks needed for the SAC agent.
        """
        self.q1 = self._build_q_NN(model_name="q1")
        self.q1_opt = tfko.Adam(lr=self._cfg.lr_critic)
        self.q1.compile(optimizer=self.q1_opt)

        self.q2 = self._build_q_NN(model_name="q2")
        self.q2_opt = tfko.Adam(lr=self._cfg.lr_critic)
        self.q2.compile(optimizer=self.q2_opt)

        if self.training:
            self.target_q1 = self._build_q_NN(model_name="target-q1")
            self.target_q1_opt = tfko.Adam(lr=self._cfg.lr_critic)
            self.target_q1.compile(optimizer=self.target_q1_opt)
            
            self.target_q2 = self._build_q_NN(model_name="target-q2")
            self.target_q2_opt = tfko.Adam(lr=self._cfg.lr_critic)
            self.target_q2.compile(optimizer=self.target_q2_opt)

            self.hard_target_update(self.q1, self.target_q1)
            self.hard_target_update(self.q2, self.target_q2)

        self.policy = self._build_policy_NN(model_name="policy")
        self.policy_opt = tfko.Adam(lr=self._cfg.lr_policy)
        self.policy.compile(optimizer=self.policy_opt)

        self.target_entropy = -(self.action_shape[0] * 1.0)
        self.log_alpha = tf.Variable(0.0)
        self.alpha_opt = tfko.Adam(lr=self._cfg.lr_alpha)

    def predict(self, net_observation):
        """
        Predict the next action
        """
        actions, _, _, _ = self.sample(net_observation)
        return actions

    def sample(self, net_observation):
        p_actions = self.policy(net_observation)
        alpha = tf.math.exp(self.log_alpha)
        pd_actions = tfpd.RelaxedOneHotCategorical(alpha,
                                                   probs=p_actions)
        actions_samples = pd_actions.sample()
        actions = tf.argmax(actions_samples, axis=-1)
        actions_oh = tf.cast(
            tf.one_hot(actions, depth=p_actions.shape[-1]),
            tf.float32)
        log_actions = pd_actions.log_prob(actions_samples)
        return actions, actions_oh, p_actions, log_actions

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch):
        """Train model on experience batch"""

        a_batch_oh = tf.one_hot(a_batch, depth=self.action_shape[-1])
        # Compute Q target
        _, a_next, _, log_next = self.sample(s2_batch)
        q1_next = self.target_q1([s2_batch, a_next], training=True)
        q2_next = self.target_q2([s2_batch, a_next], training=True)
        q_next_min = tf.math.minimum(q1_next, q2_next)
        q_next = tf.reduce_mean(q_next_min - self.alpha * log_next, axis=-1)
        q_target = r_batch + (1.0 - d_batch) * self._cfg.gamma * q_next

        # Train Q1
        with tf.GradientTape() as q1_tape:
            # Compute loss under gradient tape
            q1 = self.q1([s_batch, a_batch_oh])
            q1_sq_td_error = tf.square(q_target - q1)
            q1_loss = tf.reduce_mean(q1_sq_td_error, axis=0)

        # Q1 Compute & Apply gradients
        q1_vars = self.q1.trainable_variables
        q1_grads = q1_tape.gradient(q1_loss, q1_vars)
        self.q1_opt.apply_gradients(zip(q1_grads, q1_vars))

        # Train Q2
        with tf.GradientTape() as q2_tape:
            q2 = self.q2([s_batch, a_batch_oh])
            q2_sq_td_error = tf.square(q_target - q2)
            q2_loss = tf.reduce_mean(q2_sq_td_error, axis=0)

        # Q2 Compute & Apply gradients
        q2_vars = self.q2.trainable_variables
        q2_grads = q2_tape.gradient(q2_loss, q2_vars)
        self.q2_opt.apply_gradients(zip(q2_grads, q2_vars))

        # Policy train
        with tf.GradientTape() as policy_tape:
            _, a_new, _ , log_new = self.sample(s_batch)
            q1_new = self.q1([s_batch, a_new])
            q2_new = self.q2([s_batch, a_new])
            q_new_min = tf.math.minimum(q1_new, q2_new)
            policy_loss = self.alpha * log_new - q_new_min
            policy_loss = tf.reduce_mean(policy_loss, axis=0)

        # Policy compute & apply gradients
        policy_vars = self.policy.trainable_variables
        policy_grads = policy_tape.gradient(policy_loss, policy_vars)
        self.policy_opt.apply_gradients(zip(policy_grads, policy_vars))

        # Q Target networks soft update
        self.soft_target_update(self.q1, self.target_q1, tau=self._cfg.tau)
        self.soft_target_update(self.q2, self.target_q2, tau=self._cfg.tau)

        # Entropy train
        with tf.GradientTape() as entropy_tape:
            _, _, _ , log_new = self.sample(s_batch)
            alpha_loss = - (self.log_alpha * (log_new + self.target_entropy))
            alpha_loss = tf.reduce_mean(alpha_loss)

        # Entropy compute & apply gradients
        alpha_vars = self.log_alpha
        alpha_grads = entropy_tape.gradient(alpha_loss, alpha_vars)
        self.alpha_opt.apply_gradients(zip([alpha_grads], [alpha_vars]))

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
        
        os.makedirs(model_dir, exist_ok=True)

        self.q1.save(q1_path)
        self.q2.save(q2_path)
        self.policy.save(policy_path)

        if self.verbose:
            print("Succesfully saved networks.")

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
            print("Succesfully loaded networks.")

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
