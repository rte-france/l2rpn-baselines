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

# try:
#     from leap_net import Ltau  # this import might change if you use the "quick and dirty way".
# except ImportError:
#     # Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
#     # See AUTHORS.txt
#     # This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
#     # If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
#     # you can obtain one at http://mozilla.org/MPL/2.0/.
#     # SPDX-License-Identifier: MPL-2.0
#     # This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
# MSG_WARNING = "Leap net model is not installed on your system. Please visit \n" \
#               "https://github.com/BDonnot/leap_net \n" \
#               "to have the latest Leap net implementation."
# warnings.warn(MSG_WARNING)

# TODO implement that in the leap net package too
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import add as tfk_add
from tensorflow.keras.layers import multiply as tfk_multiply


class LtauBis(Layer):
    """
    This layer implements the Ltau layer.

    This kind of leap net layer computes, from their input `x`: `d.(e.x * tau)` where `.` denotes the
    matrix multiplication and `*` the elementwise multiplication.
    """

    def __init__(self, initializer='glorot_uniform', use_bias=True, trainable=True, name=None, **kwargs):
        super(LtauBis, self).__init__(trainable=trainable, name=name, **kwargs)
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
        res = self.d(tmp)  # no addition of x
        # res = tfk_add([x, tmp])
        return res


class DuelQLeapNet_NN(BaseDeepQ):
    """
    Constructs the desired duelling deep q learning network with a leap neural network as a modeling
    of the q function
    """
    def __init__(self,
                 nn_params,
                 training_param=None):
        if training_param is None:
            training_param = TrainingParam()
        BaseDeepQ.__init__(self,
                           nn_params,
                           training_param)
        self._custom_objects = {"LtauBis": LtauBis}
        self.construct_q_network()
        self._max_global_norm_grad = training_param.max_global_norm_grad
        self._max_value_grad = training_param.max_value_grad
        self._max_loss = training_param.max_loss

    def construct_q_network(self):
        """
        First the :attr:`l2rpn_baselines.BaseDeepQ.nn_archi` parameters are used to create a neural network
        to 'encode' the data. Then the leaps occur.

        Afterward the model is split into value an advantage, and treated as usually in any D3QN.

        """
        # Uses the network architecture found in DeepMind paper
        # The inputs and outputs size have changed, as well as replacing the convolution by dense layers.
        self._model = Sequential()
        input_x = Input(shape=(self._nn_archi.x_dim,),
                        name="x")
        inputs_tau = [Input(shape=(el,), name="tau_{}".format(nm_)) for el, nm_ in
                      zip(self._nn_archi.tau_dims, self._nn_archi.list_attr_obs_tau)]

        lay = input_x
        for (size, act) in zip(self._nn_archi.sizes, self._nn_archi.activs):
            lay = Dense(size)(lay)  # put at self.action_size
            lay = Activation(act)(lay)

        # TODO multiple taus
        l_tau = lay
        for el, nm_ in zip(inputs_tau, self._nn_archi.list_attr_obs_tau):
            l_tau = l_tau + LtauBis(name="leap_{}".format(nm_))([lay, el])

        advantage = Dense(self._action_size)(l_tau)
        value = Dense(1, name="value")(l_tau)

        meaner = Lambda(lambda x: K.mean(x, axis=1))
        mn_ = meaner(advantage)
        tmp = subtract([advantage, mn_])
        policy = add([tmp, value], name="policy")

        self._model = Model(inputs=[input_x, *inputs_tau], outputs=[policy])
        self._schedule_model, self._optimizer_model = self.make_optimiser()
        self._model.compile(loss='mse', optimizer=self._optimizer_model)

        self._target_model = Model(inputs=[input_x, *inputs_tau], outputs=[policy])

    def _make_x_tau(self, data):
        data_x = data[:, :self._nn_archi.x_dim]

        # for the taus
        data_tau = []
        prev = self._nn_archi.x_dim
        for sz, add_, mul_ in zip(self._nn_archi.tau_dims, self._nn_archi.tau_adds, self._nn_archi.tau_mults):
            data_tau.append((data[:, prev:prev+sz] + add_) * mul_)
            prev += sz

        res = [data_x, *data_tau]
        return res

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
        if self._max_global_norm_grad is not None:
            grads, _ = tf.clip_by_global_norm(grads, self._max_global_norm_grad)
        if self._max_value_grad is not None:
            grads = [tf.clip_by_value(grad, -self._max_value_grad, self._max_value_grad) for grad in grads]

        # Apply gradients
        optimizer_model.apply_gradients(zip(grads, model.trainable_variables))
        # Store LR
        self.train_lr = optimizer_model._decayed_lr('float32').numpy()
        # Return loss scalar
        return loss_npy

    def _clipped_batch_loss(self, y_true, y_pred):
        sq_error = tf.math.square(y_true - y_pred, name="sq_error")
        batch_sq_error = tf.math.reduce_sum(sq_error, axis=1, name="batch_sq_error")
        if self._max_loss is not None:
            res = tf.clip_by_value(batch_sq_error, 0.0, self._max_loss, name="batch_sq_error_clip")
        else:
            res = batch_sq_error
        return res