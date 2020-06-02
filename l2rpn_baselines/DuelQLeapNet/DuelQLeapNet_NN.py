# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import numpy as np

# tf2.0 friendly
import warnings

import tensorflow as tf
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Input, Lambda, subtract, add
    import tensorflow.keras.backend as K

from l2rpn_baselines.utils import BaseDeepQ, TrainingParam

try:
    from leap_net import Ltau  # this import might change if you use the "quick and dirty way".
except ImportError:
    # Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
    # See AUTHORS.txt
    # This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    # If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    # you can obtain one at http://mozilla.org/MPL/2.0/.
    # SPDX-License-Identifier: MPL-2.0
    # This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
    MSG_WARNING = "Leap net model is not installed on your system. Please visit \n" \
                  "https://github.com/BDonnot/leap_net \n" \
                  "to have the latest Leap net implementation."
    warnings.warn(MSG_WARNING)

    from tensorflow.keras.layers import Layer
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import add as tfk_add
    from tensorflow.keras.layers import multiply as tfk_multiply

    class Ltau(Layer):
        """
        This layer implements the Ltau layer.

        This kind of leap net layer computes, from their input `x`: `d.(e.x * tau)` where `.` denotes the
        matrix multiplication and `*` the elementwise multiplication.

        """

        def __init__(self, initializer='glorot_uniform', use_bias=True, trainable=True, name=None, **kwargs):
            super(Ltau, self).__init__(trainable=trainable, name=name, **kwargs)
            self.initializer = initializer
            self.use_bias = use_bias
            self.e = None
            self.d = None

        def build(self, input_shape):
            is_x, is_tau = input_shape
            nm_e = None
            nm_d = None
            if self.name is not None:
                nm_e = '{}_e'.format(self.name)
                nm_d = '{}_d'.format(self.name)
            self.e = Dense(is_tau[-1],
                           kernel_initializer=self.initializer,
                           use_bias=self.use_bias,
                           trainable=self.trainable,
                           name=nm_e)
            self.d = Dense(is_x[-1],
                           kernel_initializer=self.initializer,
                           use_bias=False,
                           trainable=self.trainable,
                           name=nm_d)

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'initializer': self.initializer,
                'use_bias': self.use_bias
            })
            return config

        def call(self, inputs, **kwargs):
            x, tau = inputs
            tmp = self.e(x)
            tmp = tfk_multiply([tau, tmp])  # element wise multiplication
            tmp = self.d(tmp)
            res = tfk_add([x, tmp])
            return res

import pdb


class DuelQLeapNet_NN(BaseDeepQ):
    """Constructs the desired duelling deep q learning network"""
    def __init__(self,
                 nn_params,
                 training_param=None):
        if training_param is None:
            training_param = TrainingParam()
        BaseDeepQ.__init__(self,
                           nn_params,
                           training_param)
        # self.tau_dim_start = nn_params.tau_dim_start
        # self.tau_dim_end = nn_params.tau_dim_end
        # self.add_tau = nn_params.add_tau
        self.custom_objects = {"Ltau": Ltau}
        self.construct_q_network()
        self.max_global_norm_grad = training_param.max_global_norm_grad
        self.max_value_grad = training_param.max_value_grad
        self.max_loss = training_param.max_loss

    def construct_q_network(self):
        # Uses the network architecture found in DeepMind paper
        # The inputs and outputs size have changed, as well as replacing the convolution by dense layers.
        self.model = Sequential()
        input_x = Input(shape=(self.observation_size - (self.nn_archi.tau_dim_end-self.nn_archi.tau_dim_start),),
                        name="x")
        input_tau = Input(shape=(self.nn_archi.tau_dim_end-self.nn_archi.tau_dim_start,),
                          name="tau")
        # input_tau_status = Input(shape=(self.nn_archi.tau_dim_end-self.nn_archi.tau_dim_start,),
        #                          name="tau_status")

        # # lay1 = Dense(self.observation_size)(input_x)
        ## lay1 = Activation('relu')(lay1)
        ##
        ## lay2 = Dense(self.observation_size)(lay1)
        ## lay2 = Activation('relu')(lay2)
        ##
        ## lay3 = Dense(2 * self.action_size)(lay2)  # put at self.action_size
        ## lay3 = Activation('relu')(lay3)
        ##
        ## l_tau1 = Ltau(name="encode_topo1")((lay3, input_tau_topo))
        ## l_tau2 = Ltau(name="encode_topo2")((l_tau1, input_tau_topo))
        ## l_tau3 = Ltau(name="encode_status1")((l_tau2, input_tau_status))
        ## l_tau = Ltau(name="encode_status2")((l_tau3, input_tau_status))

        # fc1 = Dense(self.action_size)(l_tau)
        # advantage = Dense(self.action_size)(fc1)
        #
        # fc2 = Dense(self.action_size)(l_tau)
        # value = Dense(1)(fc2)

        # lay1 = Dense(4 * self.observation_size)(input_x)
        # lay1 = Activation('relu')(lay1)
        #
        # lay2 = Dense(4 * self.observation_size)(lay1)
        # lay2 = Activation('relu')(lay2)
        #
        # lay3 = Dense(4 * self.observation_size)(lay2)
        # lay3 = Activation('relu')(lay3)
        #
        # lay4 = Dense(2 * self.action_size)(lay3)  # put at self.action_size
        # lay4 = Activation('relu')(lay4)
        #
        # lay5 = Dense(2 * self.action_size)(lay4)  # put at self.action_size
        # lay5 = Activation('relu')(lay5)

        lay = input_x
        for (size, act) in zip(self.nn_archi.sizes, self.nn_archi.activs):
            lay = Dense(size)(lay)  # put at self.action_size
            lay = Activation(act)(lay)

        # TODO
        l_tau = Ltau(name="tau1")((lay, input_tau))
        l_tau = Ltau(name="tau2")((l_tau, input_tau))
        # l_tau3 = Ltau(name="encode_status1")((l_tau2, input_tau_status))
        # l_tau = Ltau(name="encode_status2")((l_tau3, input_tau_status))

        advantage = Dense(self.action_size)(l_tau)
        value = Dense(1)(l_tau)

        meaner = Lambda(lambda x: K.mean(x, axis=1))
        mn_ = meaner(advantage)
        tmp = subtract([advantage, mn_])
        policy = add([tmp, value], name="policy")

        self.model = Model(inputs=[input_x, input_tau], outputs=[policy])
        self.schedule_model, self.optimizer_model = self.make_optimiser()
        self.model.compile(loss='mse', optimizer=self.optimizer_model)

        self.target_model = Model(inputs=[input_x, input_tau], outputs=[policy])
        print("Successfully constructed networks.")

    def _make_x_tau(self, data):
        data_x_1 = data[:, :self.nn_archi.tau_dim_start]
        data_x_2 = data[:, self.nn_archi.tau_dim_end:]
        data_x = np.concatenate((data_x_1, data_x_2), axis=1)

        # for the taus

        data_tau = data[:, self.nn_archi.tau_dim_start:self.nn_archi.tau_dim_end] + self.nn_archi.add_tau
        # data_tau_topo = 1.0 * data_tau
        # data_tau_topo[data_tau_topo < 0.] = 1.0
        # data_tau_status = 0.0 * data_tau
        # data_tau_status[data_tau < 0.] = 1.0
        return data_x, data_tau

    def predict_movement(self, data, epsilon, batch_size=None):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        if batch_size is None:
            batch_size = data.shape[0]
        data_split = self._make_x_tau(data)
        res = super().predict_movement(data_split, epsilon=epsilon, batch_size=batch_size)
        return res

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, tf_writer=None, batch_size=None):
        if batch_size is None:
            batch_size = s_batch.shape[0]
        s_batch_split = self._make_x_tau(s_batch)
        s2_batch_split = self._make_x_tau(s2_batch)
        res = super().train(s_batch_split,
                            a_batch,
                            r_batch,
                            d_batch,
                            s2_batch_split,
                            tf_writer=tf_writer,
                            batch_size=batch_size)
        return res

    def train_on_batch(self, model, optimizer_model, x, y_true):
        """
        clip the loss
        """
        with tf.GradientTape() as tape:
            # Get y_pred for batch
            y_pred = model(x)
            # Compute loss for each sample in the batch
            # and then clip it
            batch_loss = self._clipped_batch_loss(y_true, y_pred)
            # Compute mean scalar loss
            loss = tf.math.reduce_mean(batch_loss)
        loss_npy = loss.numpy()

        # Compute gradients
        grads = tape.gradient(loss, model.trainable_variables)

        # clip gradients
        if self.max_global_norm_grad is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_global_norm_grad)
        if self.max_value_grad is not None:
            grads = [tf.clip_by_value(grad, -self.max_value_grad, self.max_value_grad) for grad in grads]

        # Apply gradients
        optimizer_model.apply_gradients(zip(grads, model.trainable_variables))
        # Store LR
        self.train_lr = optimizer_model._decayed_lr('float32').numpy()
        # Return loss scalar
        return loss_npy

    def _clipped_batch_loss(self, y_true, y_pred):
        sq_error = tf.math.square(y_true - y_pred, name="sq_error")
        batch_sq_error = tf.math.reduce_sum(sq_error, axis=1, name="batch_sq_error")
        return tf.clip_by_value(batch_sq_error, 0.0, self.max_loss, name="batch_sq_error_clip")