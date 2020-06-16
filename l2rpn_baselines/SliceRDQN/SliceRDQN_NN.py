# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
import tensorflow.keras.models as tfkm
import tensorflow.keras.optimizers as tfko
import tensorflow.keras.layers as tfkl
import tensorflow.keras.activations as tfka


class SliceRDQN_NN(object):
    def __init__(self,
                 action_size,
                 observation_shape,
                 slices,
                 learning_rate = 1e-5):
        self.action_size = action_size
        self.observation_shape = observation_shape
        self.slices = slices
        self.n_slices = len(slices.keys())
        self.encoded_size = 384
        self.h_size = 512
        self.lr = learning_rate
        self.model = None
        self.construct_q_network()

    def forward_slice_encode(self, input_layer, slice_indexes, name):
        batch_size = tf.shape(input_layer)[0]
        trace_size = tf.shape(input_layer)[1]
        sliced = tf.gather(input_layer, slice_indexes, axis=2, name=name+"_slice")
        # Reshape to (batch_size * trace_len, data_size)
        # So that Dense process each data vect separately
        sliced = tf.reshape(sliced, (-1, sliced.shape[2] * sliced.shape[3]), name=name+"_enc_reshape")

        # Bayesian NN simulate using dropout
        lay1 = tfkl.Dropout(self.dropout_rate, name=name+"_bnn")(sliced)

        # Three layers encoder
        lay1 = tfkl.Dense(496, name=name+"_fc1")(lay1)
        lay1 = tf.nn.leaky_relu(lay1, alpha=0.01, name=name+"_leak_fc1")
        lay2 = tfkl.Dense(412, name=name+"_fc2")(lay1)
        lay2 = tf.nn.leaky_relu(lay2, alpha=0.01, name=name+"_leak_fc2")
        lay3 = tfkl.Dense(self.encoded_size, name=name+"_fc3")(lay2)
        lay3 = tf.nn.leaky_relu(lay3, alpha=0.01, name=name+"_leak_fc3")

        # Reshape encoded results to (batch_size, trace_len, encoded_size)
        encoded_shape = (batch_size, trace_size, self.encoded_size)
        encoded = tf.reshape(lay3, encoded_shape, name=name+"_encoded")
        return encoded

    def forward_slice_recur(self, input_rnn, mem_state, carry_state, name):
        # Single LSTM cell over trace_length
        lstm_layer = tfkl.LSTM(self.h_size, return_state=True,
                               name=name+"_lstm")
        states = [mem_state, carry_state]
        lstm_out, mem, carry = lstm_layer(input_rnn, initial_state=states)
        return lstm_out, mem, carry

    def forward_streams(self, hidden, q_len, name):
        # Advantage stream
        advantage = tfkl.Dense(64, name=name+"_fcadv")(hidden)
        advantage = tf.nn.leaky_relu(advantage, alpha=0.01, name=name+"_leakyadv")
        advantage = tfkl.Dense(q_len, name=name+"_adv")(advantage)
        advantage_mean = tf.math.reduce_mean(advantage, axis=1,
                                             keepdims=True,
                                             name=name+"_adv_mean")
        advantage = tfkl.subtract([advantage, advantage_mean],
                                  name= name+"_adv_sub")

        # Value stream
        value = tfkl.Dense(64, name=name+"_fcval")(hidden)
        value = tf.nn.leaky_relu(value, alpha=0.01, name=name+"_leakyval")
        value = tfkl.Dense(1, name=name+"_val")(value)

        # Q values = val + adv
        slice_q = tf.math.add(value, advantage, name=name+"_sliceq")
        return slice_q

    def construct_q_network(self):
        # Defines input tensors and scalars
        self.trace_length = tf.Variable(1, trainable=False,
                                        dtype=tf.int32, name="trace_len")
        self.dropout_rate = tf.Variable(0.0, trainable=False,
                                        dtype=tf.float32, name="drop_rate")
        states_shape = (self.n_slices, self.h_size)
        input_mem_states = tfk.Input(dtype=tf.float32, shape=states_shape,
                                     name='input_mem_states')
        input_carry_states = tfk.Input(dtype=tf.float32, shape=states_shape,
                                       name='input_carry_states')
        input_shape = (None,) + self.observation_shape
        input_layer = tfk.Input(dtype=tf.float32, shape=input_shape,
                                name='input_obs')

        # Forward encode slices
        slice_to_encoded = {}
        for slice_name, slice_v in self.slices.items():
            slice_idxs = slice_v["indexes"]
            encoded_slice = self.forward_slice_encode(input_layer,
                                                      slice_idxs,
                                                      slice_name)
            slice_to_encoded[slice_name] = encoded_slice

        # Forward recurring part
        slice_to_hidden = {}
        mem_states = []
        carry_states = []
        for idx, (slice_name, rnn_input) in enumerate(slice_to_encoded.items()):
            mem = input_mem_states[:, idx]
            carry = input_carry_states[:, idx]
            h, m, c = self.forward_slice_recur(rnn_input, mem, carry,
                                               slice_name)
            slice_to_hidden[slice_name] = h
            mem_states.append(m)
            carry_states.append(c)

        # Pack states outputs
        output_mem_states = tf.stack(mem_states, axis=1, name="mem_stack")
        output_carry_states = tf.stack(carry_states, axis=1, name="carry_stack")

        # Advantage and Value streams
        q_slices_li = []
        q_dn_li = []
        for slice_name, hidden in slice_to_hidden.items():
            q_len = self.slices[slice_name]["q_len"]
            slice_q = self.forward_streams(hidden, q_len, slice_name)
            q_slices_li.append(slice_q[:, 1:])
            q_dn_li.append(slice_q[:, :1])

        
        # Concatenate final Q
        q_dn = tf.stack(q_dn_li, 1, name="q_dn_stack")
        q_dn_mean = tf.reduce_mean(q_dn, axis=1, name="q_dn_mean")
        q_slices = tf.concat(q_slices_li, 1, name="q_slices_concat")
        output_q = tf.concat([q_dn_mean, q_slices], 1, name="q_batch")

        # Backwards pass
        model_inputs = [
            input_mem_states,
            input_carry_states,
            input_layer
        ]
        model_outputs = [
            output_q,
            output_mem_states,
            output_carry_states
        ]
        self.model = tfk.Model(inputs=model_inputs,
                               outputs=model_outputs,
                               name=self.__class__.__name__)
        losses = [
            self._clipped_mse_loss,
            self._no_loss,
            self._no_loss
        ]
        self.optimizer = tfko.Adam(lr=self.lr, clipnorm=1.0)
        self.model.compile(loss=losses, optimizer=self.optimizer)

    def _no_loss(self, y_true, y_pred):
        return 0.0

    def _clipped_mse_loss(self, Qnext, Q):
        loss = tf.math.reduce_mean(tf.math.square(Qnext - Q), name="loss_mse")
        clipped_loss = tf.clip_by_value(loss, 0.0, 1e3, name="loss_clip")
        return clipped_loss

    def bayesian_move(self, data, mem, carry, rate = 0.0):
        self.dropout_rate.assign(float(rate))
        self.trace_length.assign(1)

        input_shape = (1, 1) + self.observation_shape
        data_input = data.reshape(input_shape)
        mem_input = mem.reshape(1, self.n_slices, self.h_size)
        carry_input = carry.reshape(1, self.n_slices, self.h_size)
        model_input = [mem_input, carry_input, data_input]
        
        Q, mem, carry = self.model.predict(model_input, batch_size = 1)
        move = np.argmax(Q)

        return move, Q, mem, carry

    def random_move(self, data, mem, carry):
        self.trace_length.assign(1)
        self.dropout_rate.assign(0.0)

        input_shape = (1, 1) + self.observation_shape
        data_input = data.reshape(input_shape)
        mem_input = mem.reshape(1, self.n_slices, self.h_size)
        carry_input = carry.reshape(1, self.n_slices, self.h_size)
        model_input = [mem_input, carry_input, data_input]

        Q, mem, carry = self.model.predict(model_input, batch_size = 1) 
        move = np.random.randint(0, self.action_size)

        return move, mem, carry

    def predict_move(self, data, mem, carry):
        self.trace_length.assign(1)
        self.dropout_rate.assign(0.0)

        input_shape = (1, 1) + self.observation_shape
        data_input = data.reshape(input_shape)
        mem_input = mem.reshape(1, self.n_slices, self.h_size)
        carry_input = carry.reshape(1, self.n_slices, self.h_size)
        model_input = [mem_input, carry_input, data_input]
        
        Q, mem, carry = self.model.predict(model_input, batch_size = 1)
        move = np.argmax(Q)

        return move, Q, mem, carry

    def update_target_hard(self, target_model):
        this_weights = self.model.get_weights()
        target_model.set_weights(this_weights)

    def update_target_soft(self, target_model, tau=1e-2):
        tau_inv = 1.0 - tau
        # Get parameters to update
        target_params = target_model.trainable_variables
        main_params = self.model.trainable_variables

        # Update each param
        for i, var in enumerate(target_params):
            var_persist = var.value() * tau_inv
            var_update = main_params[i].value() * tau
            # Poliak averaging
            var.assign(var_update + var_persist)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        # nothing has changed
        self.model.save_weights(path)
        print("Successfully saved model at: {}".format(path))

    def load_network(self, path):
        # nothing has changed
        self.model.load_weights(path)
        print("Successfully loaded network from: {}".format(path))

