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


# TODO implement that in the leap net package too
from tensorflow.keras.layers import Dense


from l2rpn_baselines.DuelQLeapNet.DuelQLeapNet_NN import LtauBis


class TestLeapNet_NN(BaseDeepQ):
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
        self._max_global_norm_grad = training_param.max_global_norm_grad
        self._max_value_grad = training_param.max_value_grad
        self._max_loss = training_param.max_loss

        self.train_lr = 1.0

        # added
        self.encoded_state = None
        self.grid_model = None
        self._schedule_grid_model = None
        self._optimizer_grid_model = None

        self.construct_q_network()

    def construct_q_network(self):
        """
        First the :attr:`l2rpn_baselines.BaseDeepQ.nn_archi` parameters are used to create a neural network
        to 'encode' the data. Then the leaps occur.

        Afterward the model is split into value an advantage, and treated as usually in any D3QN.

        """
        # Uses the network architecture found in DeepMind paper
        # The inputs and outputs size have changed, as well as replacing the convolution by dense layers.
        self._model = Sequential()

        inputs_x = [Input(shape=(el,), name="x_{}".format(nm_)) for el, nm_ in
                      zip(self._nn_archi.x_dims, self._nn_archi.list_attr_obs_x)]
        inputs_tau = [Input(shape=(el,), name="tau_{}".format(nm_)) for el, nm_ in
                      zip(self._nn_archi.tau_dims, self._nn_archi.list_attr_obs_tau)]
        input_topo = Input(shape=(self._nn_archi.dim_topo,), name="topo")

        # encode each data type in initial layers
        encs_out = []
        for init_val, nm_ in zip(inputs_x, self._nn_archi.list_attr_obs_x):
            lay = init_val
            for i, size in enumerate(self._nn_archi.sizes_enc):
                lay = Dense(size, name="enc_{}_{}".format(nm_, i))(lay) # TODO resnet instead of Dense
                lay = Activation("relu")(lay)
            encs_out.append(lay)

        # concatenate all that
        lay = tf.keras.layers.concatenate(encs_out)
        # now "lay" is the encoded observation

        # i do a few layer
        for i, size in enumerate(self._nn_archi.sizes_main):
            lay = Dense(size, name="main_{}".format(i))(lay)  # TODO resnet instead of Dense
            lay = Activation("relu")(lay)

        # now i do the leap net to encode the state
        encoded_state = lay + LtauBis(name="leap_topo")([lay, input_topo])
        self.encoded_state = encoded_state

        # i predict the flows, that i should be able to do
        lay = encoded_state
        for i, size in enumerate(self._nn_archi.sizes_for_flow):
            lay = Dense(size, name="flow_{}".format(i))(lay)
            lay = Activation("relu")(lay)

        # predict now the flows
        flow_hat = Dense(self._nn_archi.dim_flow)(lay)

        # NB grid_model does not use inputs_tau
        self.grid_model = Model(inputs=[*inputs_x,  *inputs_tau, input_topo], outputs=[flow_hat])
        self._schedule_grid_model, self._optimizer_grid_model = self.make_optimiser()
        self.grid_model.compile(loss='mse', optimizer=self._optimizer_grid_model)

        # And now let's predict the Q value of the action.
        lay = self.encoded_state
        for i, size in enumerate(self._nn_archi.sizes_Qnet):
            lay = Dense(size, name="qvalue_{}".format(i))(lay)  # TODO resnet instead of Dense
            lay = Activation("relu")(lay)

        # And i predict the Q value of the action
        l_tau = lay
        for el, nm_ in zip(inputs_tau, self._nn_archi.list_attr_obs_tau):
            l_tau = l_tau + LtauBis(name="leap_{}".format(nm_))([lay, el])

        advantage = Dense(self._action_size)(l_tau)
        value = Dense(1, name="value")(l_tau)

        meaner = Lambda(lambda x: K.mean(x, axis=1))
        mn_ = meaner(advantage)
        tmp = subtract([advantage, mn_])
        policy = add([tmp, value], name="policy")

        self._model = Model(inputs=[*inputs_x, *inputs_tau, input_topo], outputs=[policy])
        self._schedule_model, self._optimizer_model = self.make_optimiser()
        self._model.compile(loss='mse', optimizer=self._optimizer_model)

        self._target_model = Model(inputs=[*inputs_x, *inputs_tau, input_topo], outputs=[policy])

    def _make_x_tau(self, data):
        data_x = []
        # for the x's
        prev = 0

        for sz, add_, mul_ in zip(self._nn_archi.x_dims, self._nn_archi.x_adds, self._nn_archi.x_mults):
            tmp = (data[:, prev:(prev+sz)] + add_) * mul_
            data_x.append(tmp)
            prev += sz

        # for the taus
        data_tau = []
        prev = self._nn_archi.x_dim
        for sz, add_, mul_ in zip(self._nn_archi.tau_dims, self._nn_archi.tau_adds, self._nn_archi.tau_mults):
            data_tau.append((data[:, prev:(prev+sz)] + add_) * mul_)
            prev += sz

        data_topo = data[:, prev:(prev+self._nn_archi.dim_topo)]
        prev += self._nn_archi.dim_topo
        data_flow = data[:, prev:]
        res = [*data_x, *data_tau, data_topo, data_flow]
        return res

    def predict_movement(self, data, epsilon, batch_size=None):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        if batch_size is None:
            batch_size = data.shape[0]
        data_split = self._make_x_tau(data)
        *data_split, flow = data_split
        res = super().predict_movement(data_split, epsilon=epsilon, batch_size=batch_size)
        return res

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, tf_writer=None, batch_size=None):
        if batch_size is None:
            batch_size = s_batch.shape[0]
        s_batch_split = self._make_x_tau(s_batch)
        s2_batch_split = self._make_x_tau(s2_batch)

        # train the grid model to accurately predict the state of the grid
        *s_batch_split, data_flow1 = s_batch_split
        loss1 = self.grid_model.train_on_batch(s_batch_split, data_flow1)
        *s2_batch_split, data_flow2 = s2_batch_split
        loss2 = self.grid_model.train_on_batch(s2_batch_split, data_flow2)

        # and now train the q network
        res = super().train(s_batch_split,
                            a_batch,
                            r_batch,
                            d_batch,
                            s2_batch_split,
                            tf_writer=tf_writer,
                            batch_size=batch_size)
        return loss1 + loss2

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