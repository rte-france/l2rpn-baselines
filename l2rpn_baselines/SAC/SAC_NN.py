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

    LeapNet layer:
    https://arxiv.org/abs/1908.08314

    Reference implementation:
    https://github.com/cyoon1729/Policy-Gradient-Methods
    """
    def __init__(self,
                 observation_shape,
                 bridge_shape,
                 split_shape,
                 action_shape,
                 impact_shape,
                 nn_config,
                 training=False,
                 verbose=False):
        self.emb_shape = (nn_config.sizes_emb[-1],)
        self.observation_shape = observation_shape
        self.bridge_shape = bridge_shape
        self.split_shape = split_shape
        self.action_shape = action_shape
        self.impact_shape = impact_shape
        self._cfg = nn_config
        self.verbose = verbose

        self.training = training
        self.alpha = self._cfg.alpha

        self.construct_networks()

    def _build_mlp(self, input_layer,
                   sizes, activations,
                   name, norm=False):
        if norm:
            # Use laynorm + tanh reg
            lay = tfkl.LayerNormalization()(input_layer)
            lay = tf.nn.tanh(lay)
        else:
            lay = input_layer

        lay_size_activ = zip(sizes, activations)
        for lay_id, (size, act) in enumerate(lay_size_activ):
            lay_name = "{}_fc_{}".format(name, lay_id)
            act_name = "{}_act_{}".format(name, lay_id)
            lay = tfkl.Dense(size, name=lay_name)(lay)
            if act is not None and act != 'None':
                lay = tfkl.Activation(act, name=act_name)(lay)

        return lay

    def _build_ltau(self,
                    input_x,
                    input_tau,
                    model_name):
        e_name = "{}-e-ltau-fc".format(model_name)
        e = tfkl.Dense(input_tau.shape[-1], name=e_name)(input_x)

        lt_name = "{}-mul-ltau".format(model_name)
        lt = tfkl.multiply([input_tau, e], name=lt_name)

        d_name = "{}-d-ltau-fc".format(model_name)
        d = tfkl.Dense(input_x.shape[-1], use_bias=False,
                       name=d_name)(lt)
        return d

    def _build_emb_NN(self, model_name):
        #input_state = tfkl.Input(shape=self.observation_shape)
        input_bridge = tfkl.Input(shape=self.bridge_shape)
        input_split = tfkl.Input(shape=self.split_shape)

        input_cat = tf.concat([input_bridge, input_split], axis=1)
        emb_obs = self._build_mlp(input_cat,
                                  self._cfg.sizes_emb,
                                  self._cfg.activations_emb,
                                  model_name,
                                  norm=self._cfg.norm_emb)
        #emb_bridge = self._build_ltau(emb_obs, input_bridge,
        #                              model_name + "-b")
        #emb_split = self._build_ltau(emb_obs, input_split,
        #                             model_name + "-s")
        #emb = tfkl.add([emb_obs, emb_bridge, emb_split],
        #               name=model_name + "-acc")

        #model_inputs = [input_state, input_bridge, input_split]
        model_inputs = [input_bridge, input_split]
        model_outputs = [emb_obs]
        model = tfk.Model(inputs=model_inputs,
                          outputs=model_outputs,
                          name=model_name)
        return model

    def _build_critic_NN(self, model_name):
        input_state = tfkl.Input(shape=self.observation_shape)
        input_action = tfkl.Input(shape=self.action_shape)
        input_impact = tfkl.Input(shape=self.impact_shape)

        input_layer = tf.concat([input_state, input_action, input_impact],
                                axis=-1)

        hidden = self._build_mlp(input_layer,
                                 self._cfg.sizes_critic,
                                 self._cfg.activations_critic,
                                 model_name,
                                 norm=self._cfg.norm_critic)

        Q = tfkl.Dense(1, name=model_name + "-Q")(hidden)

        model_inputs = [input_state, input_action, input_impact]
        model_outputs = [Q]
        model = tfk.Model(inputs=model_inputs,
                          outputs=model_outputs,
                          name=model_name)
        return model

    def _build_actor_NN(self, model_name):
        input_state = tfkl.Input(shape=self.emb_shape)

        # Get target state distribution
        lay = self._build_mlp(input_state,
                              self._cfg.sizes_policy,
                              self._cfg.activations_policy,
                              model_name,
                              norm=self._cfg.norm_policy)

        action_size = self.action_shape[0]

        mean_name = "{}_fc_mean".format(model_name)
        mean = tfkl.Dense(action_size, name=mean_name)(lay)
        
        log_std_name = "{}_fc_log_std".format(model_name)
        log_std = tfkl.Dense(action_size, name=log_std_name)(lay)
        log_std = tf.clip_by_value(log_std,
                                   self._cfg.log_std_min,
                                   self._cfg.log_std_max)


        # Output action distribution
        model_inputs = [input_state]
        model_outputs = [mean, log_std]
        model = tfk.Model(inputs=model_inputs,
                          outputs=model_outputs,
                          name=model_name)
        return model

    def _build_impact_NN(self, model_name):
        input_state = tfkl.Input(shape=self.emb_shape)
        input_action = tfkl.Input(shape=self.action_shape)

        # Get impact execution
        input_impact = tf.concat([input_state, input_action], axis=-1)
        impact_lay = self._build_mlp(input_impact,
                                     self._cfg.sizes_policy,
                                     self._cfg.activations_policy,
                                     model_name,
                                     norm=self._cfg.norm_policy)

        impact_size = self.impact_shape[0]
        mean2_name = "{}_fc_mean2".format(model_name)
        mean2 = tfkl.Dense(impact_size, name=mean2_name)(impact_lay)

        log_std2_name = "{}_fc_log_std2".format(model_name)
        log_std2 = tfkl.Dense(impact_size, name=log_std2_name)(impact_lay)
        log_std2 = tf.clip_by_value(log_std2,
                                    self._cfg.log_std_min,
                                    self._cfg.log_std_max)

        # Output impact distribution
        model_inputs = [input_state, input_action]
        model_outputs = [mean2, log_std2]
        model = tfk.Model(inputs=model_inputs,
                          outputs=model_outputs,
                          name=model_name)
        return model

    def construct_networks(self):
        """
        This constructs all the networks needed for the SAC agent.
        """
        self.q1 = self._build_critic_NN(model_name="q1")
        self.q1_opt = tfko.Adam(lr=self._cfg.lr_critic)
        self.q1.compile(optimizer=self.q1_opt)

        self.q2 = self._build_critic_NN(model_name="q2")
        self.q2_opt = tfko.Adam(lr=self._cfg.lr_critic)
        self.q2.compile(optimizer=self.q2_opt)

        if self.training:
            self.target_q1 = self._build_critic_NN(model_name="target-q1")
            self.target_q1_opt = tfko.Adam(lr=self._cfg.lr_critic)
            self.target_q1.compile(optimizer=self.target_q1_opt)

            self.target_q2 = self._build_critic_NN(model_name="target-q2")
            self.target_q2_opt = tfko.Adam(lr=self._cfg.lr_critic)
            self.target_q2.compile(optimizer=self.target_q2_opt)

            self.hard_target_update(self.q1, self.target_q1)
            self.hard_target_update(self.q2, self.target_q2)

        self.emb = self._build_emb_NN(model_name="emb")
        self.actor = self._build_actor_NN(model_name="actor")
        self.impact = self._build_impact_NN(model_name="impact")
        self.policy_opt = tfko.Adam(lr=self._cfg.lr_policy)
        self.emb.compile(optimizer=self.policy_opt)
        self.actor.compile(optimizer=self.policy_opt)
        self.impact.compile(optimizer=self.policy_opt)

        if self.training:
            actor_outsize = self.action_shape[0] + self.impact_shape[0]
            self.target_entropy = -(actor_outsize)
            self.log_alpha = tf.Variable(float(np.log(self.alpha)),
                                         trainable=True)
            self.alpha_opt = tfko.Adam(lr=self._cfg.lr_alpha)

    def predict(self, net_observation, net_bridge, net_split, sample=True):
        """
        Predict the next action
        """
        if sample:
            actions, _, impacts, _, _ = self.sample(net_observation,
                                                    net_bridge,
                                                    net_split)
        else:
            actions, impacts = self.mean(net_observation,
                                         net_bridge,
                                         net_split)
        return actions, impacts

    def sample(self, net_observation, net_bridge, net_split, eps=1e-6):
        # Compute policy common embedding
        #emb = self.emb([net_observation, net_bridge, net_split])
        emb = self.emb([net_bridge, net_split])

        # Get actor rsample
        mean, log_std = self.actor(emb)
        std = tf.math.exp(log_std)
        pd_actions = tfpd.MultivariateNormalDiag(loc=mean, scale_diag=std)
        samples_actions = pd_actions.sample()
        actions = tf.nn.tanh(samples_actions)
        logp_actions = pd_actions.log_prob(samples_actions)
        corr_actions = tf.math.log(1.0 - tf.math.square(actions) + eps)
        corr_actions = tf.reduce_sum(corr_actions, axis=1)
        log_pi_actions = logp_actions - corr_actions

        # Get impact rsample
        mean2, log_std2 = self.impact([emb, actions])
        std2 = tf.math.exp(log_std2)
        pd_impacts = tfpd.MultivariateNormalDiag(loc=mean2, scale_diag=std2)
        samples_impacts = pd_impacts.sample()
        impacts = tf.nn.tanh(samples_impacts)
        logp_impacts = pd_impacts.log_prob(samples_impacts)
        corr_impacts = tf.math.log(1.0 - tf.math.square(impacts) + eps)
        corr_impacts = tf.reduce_sum(corr_impacts, axis=1)
        log_pi_impacts = logp_impacts - corr_impacts

        # Return rsampled policy 
        return actions, log_pi_actions, impacts, log_pi_impacts, emb

    def mean(self, net_observation, net_bridge, net_split):
        # Compute common embedding
        #emb = self.emb([net_observation, net_bridge, net_split])
        emb = self.emb([net_bridge, net_split])

        # Get actor eval
        mean, _ = self.actor(emb)
        actions = tf.nn.tanh(mean)

        # Get impact eval
        mean2, _ = self.impact([emb, actions])
        impacts = tf.nn.tanh(mean2)

        # Return policy eval
        return actions, impacts

    def train(self, s_batch, a_batch, i_batch, r_batch, d_batch, s2_batch,
              delayed=False):
        """Train model on experience batch"""

        # Unpack states batches
        s_obs = np.vstack(s_batch[:, 0])
        s_bridge = np.vstack(s_batch[:, 1])
        s_split = np.vstack(s_batch[:, 2])
        s2_obs = np.vstack(s2_batch[:, 0])
        s2_bridge = np.vstack(s2_batch[:, 1])
        s2_split = np.vstack(s2_batch[:, 2])

        # Compute Q target
        #emb = self.emb([s_obs, s_bridge, s_split])
        emb = self.emb([s_bridge, s_split])
        sample_next = self.sample(s2_obs, s2_bridge, s2_split)
        (a_next, a_log_next, i_next, i_log_next, emb_next) = sample_next
        log_next = a_log_next + i_log_next
        q1_next = self.target_q1([s2_obs, a_next, i_next], training=True)
        q2_next = self.target_q2([s2_obs, a_next, i_next], training=True)
        q_next_min = tf.math.minimum(q1_next, q2_next)
        q_next = q_next_min - self.alpha * log_next
        q_target = r_batch + (1.0 - d_batch) * self._cfg.gamma * q_next

        # Update Q1
        with tf.GradientTape() as q1_tape:
            # Compute loss under gradient tape
            q1 = self.q1([s_obs, a_batch, i_batch])
            q1_sq_td_error = tf.square(q_target - q1)
            q1_loss = 0.5 * tf.reduce_mean(q1_sq_td_error)

        # Q1 Compute & Apply gradients
        q1_vars = self.q1.trainable_variables
        q1_grads = q1_tape.gradient(q1_loss, q1_vars)
        self.q1_opt.apply_gradients(zip(q1_grads, q1_vars))

        # Update Q2
        with tf.GradientTape() as q2_tape:
            q2 = self.q2([s_obs, a_batch, i_batch])
            q2_sq_td_error = tf.square(q_target - q2)
            q2_loss = 0.5 * tf.reduce_mean(q2_sq_td_error)

        # Q2 Compute & Apply gradients
        q2_vars = self.q2.trainable_variables
        q2_grads = q2_tape.gradient(q2_loss, q2_vars)
        self.q2_opt.apply_gradients(zip(q2_grads, q2_vars))

        # Get losses values from GPU
        q1l = q1_loss.numpy()
        q2l = q2_loss.numpy()

        if delayed:
            return q1l, q2l, None, None

        # Policy train
        with tf.GradientTape() as policy_tape:
            sample_new = self.sample(s_obs, s_bridge, s_split)
            (a_new, a_log_new, i_new, i_log_new, emb_new) = sample_new
            log_new = a_log_new + i_log_new
            q1_new = self.q1([s_obs, a_new, i_new])
            q2_new = self.q2([s_obs, a_new, i_new])
            q_new_min = tf.math.minimum(q1_new, q2_new)
            policy_loss = self.alpha * log_new - q_new_min
            policy_loss = tf.reduce_mean(policy_loss)

        # Policy compute & apply gradients
        policy_vars = (self.emb.trainable_variables + \
                       self.actor.trainable_variables + \
                       self.impact.trainable_variables)
        policy_grads = policy_tape.gradient(policy_loss, policy_vars)
        self.policy_opt.apply_gradients(zip(policy_grads, policy_vars))

        # Q Target networks soft update
        self.soft_target_update(self.q1, self.target_q1, tau=self._cfg.tau)
        self.soft_target_update(self.q2, self.target_q2, tau=self._cfg.tau)

        # Update Entropy
        with tf.GradientTape() as entropy_tape:
            alpha_loss = self.log_alpha * (-log_new - self.target_entropy)
            alpha_loss = tf.reduce_mean(alpha_loss)

        # Entropy compute & apply gradients
        alpha_vars = self.log_alpha
        alpha_grads = entropy_tape.gradient(alpha_loss, alpha_vars)
        self.alpha_opt.apply_gradients(zip([alpha_grads], [alpha_vars]))

        # Update alpha
        self.alpha = tf.math.exp(self.log_alpha)

        # Get losses values from GPU
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

        emb_path = os.path.join(model_dir, "emb.{}".format(ext))
        emb_path = os.path.abspath(emb_path)

        actor_path = os.path.join(model_dir, "actor.{}".format(ext))
        actor_path = os.path.abspath(actor_path)

        impact_path = os.path.join(model_dir, "impact.{}".format(ext))
        impact_path = os.path.abspath(impact_path)

        return  model_dir, q1_path, q2_path, emb_path, actor_path, impact_path

    def save_network(self, path, name=None, ext="h5"):
        """
        Saves all the models network weigths
        """
        model_paths = self.model_paths(path, name, ext)
        model_dir = model_paths[0]
        q1_path = model_paths[1]
        q2_path = model_paths[2]
        emb_path = model_paths[3]
        actor_path = model_paths[4]
        impact_path = model_paths[5]

        os.makedirs(model_dir, exist_ok=True)

        self.q1.save(q1_path)
        self.q2.save(q2_path)
        self.emb.save(emb_path)
        self.actor.save(actor_path)
        self.impact.save(impact_path)

        if self.verbose:
            print("Succesfully saved networks.")

    def load_network(self, path, name=None, ext="h5"):
        """
        Load the models networks weights from disk
        """

        model_paths = self.model_paths(path, name, ext)
        model_dir = model_paths[0]
        q1_path = model_paths[1]
        q2_path = model_paths[2]
        emb_path = model_paths[3]
        actor_path = model_paths[4]
        impact_path = model_paths[5]

        self.q1.load_weights(q1_path)
        self.q2.load_weights(q2_path)
        self.emb.load_weights(emb_path)
        self.actor.load_weights(actor_path)
        self.impact.load_weights(impact_path)

        if self.training:
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
