# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import numpy as np
import os

# tf2.0 friendly
import warnings

try:
    import tensorflow as tf
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import Input, Lambda, subtract, add
        import tensorflow.keras.backend as K
    # TODO implement that in the leap net package too
    from tensorflow.keras.layers import Dense
    _CAN_USE_TENSORFLOW = True
except ImportError:
    _CAN_USE_TENSORFLOW = False
    
from l2rpn_baselines.utils import BaseDeepQ, TrainingParam
from l2rpn_baselines.DuelQLeapNet.duelQLeapNet_NN import LtauBis


class LeapNetEncoded_NN(BaseDeepQ):
    """
    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
    Constructs the desired neural networks.

    More information on the leap net can be found at `Leap Net on Github <https://github.com/BDonnot/leap_net>`_

    These are:

    - a "state encoder" that uses a leap net to "encode" the observation, or at least the part
      related to powergrid
    - a q network, that uses the output of the state encoder to predict which action is best.

    The Q network can have other types of input, and can also be a leap net, see the class
    :class:`l2rpn_baselines.LeapNetEncoded.leapNetEncoded_NNParam.LeapNetEncoded_NNParam` for more information

    """
    def __init__(self,
                 nn_params,
                 training_param=None):
        if not _CAN_USE_TENSORFLOW:
            raise RuntimeError("Cannot import tensorflow, this function cannot be used.")
        
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
        self._qnet_variables = []
        self.grid_model_losses_npy = None

        self.construct_q_network()

    def construct_q_network(self):
        """
        Builds the Q network.
        """
        # Uses the network architecture found in DeepMind paper
        # The inputs and outputs size have changed, as well as replacing the convolution by dense layers.
        self._model = Sequential()
        inputs_x = [Input(shape=(el,), name="x_{}".format(nm_)) for el, nm_ in
                    zip(self._nn_archi.x_dims, self._nn_archi.list_attr_obs_x)]
        inputs_q = [Input(shape=(el,), name="input_q_{}".format(nm_)) for el, nm_ in
                    zip(self._nn_archi.input_q_dims, self._nn_archi.list_attr_obs_input_q)]
        inputs_tau = [Input(shape=(el,), name="tau_{}".format(nm_)) for el, nm_ in
                      zip(self._nn_archi.tau_dims, self._nn_archi.list_attr_obs_tau)]
        input_topo = Input(shape=(2*self._nn_archi.dim_topo,), name="topo")
        models_all_inputs = [*inputs_x, *inputs_q, *inputs_tau, input_topo]

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
        encoded_state = tf.keras.layers.add([lay, LtauBis(name="leap_topo")([lay, input_topo])],
                                            name="encoded_state")
        self.encoded_state = tf.keras.backend.stop_gradient(encoded_state)

        # i predict the full state of the grid given the "control" variables
        outputs_gm = []
        grid_model_losses = {}
        lossWeights = {}  # TODO
        for sz_out, nm_ in zip(self._nn_archi.gm_out_dims,
                               self._nn_archi.list_attr_obs_gm_out):
            lay = encoded_state  # carefull i need my gradients here ! (don't use self.encoded_state)
            for i, size in enumerate(self._nn_archi.sizes_out_gm):
                lay = Dense(size, name="{}_{}".format(nm_, i))(lay)
                lay = Activation("relu")(lay)

            # predict now the variable
            name_output = "{}_hat".format(nm_)
            pred_ = Dense(sz_out, name=name_output)(lay)
            outputs_gm.append(pred_)
            grid_model_losses[name_output] = "mse"

        # NB grid_model does not use inputs_tau
        self.grid_model = Model(inputs=models_all_inputs, outputs=outputs_gm, name="grid_model")
        self._schedule_grid_model, self._optimizer_grid_model = self.make_optimiser()
        self.grid_model.compile(loss=grid_model_losses, optimizer=self._optimizer_grid_model) # , loss_weights=lossWeights

        # And now let's predict the Q values of each actions given the encoded grid state
        input_Qnet = inputs_q + [self.encoded_state]
        # TODO do i pre process the data coming from inputs_q ???

        lay = tf.keras.layers.concatenate(input_Qnet, name="input_Q_network")
        for i, size in enumerate(self._nn_archi.sizes_Qnet):
            tmp = Dense(size, name="qvalue_{}".format(i))  # TODO resnet instead of Dense
            lay = tmp(lay)
            lay = Activation("relu")(lay)
            self._qnet_variables += tmp.trainable_weights

        # And i predict the Q value of the action
        l_tau = lay
        for el, nm_ in zip(inputs_tau, self._nn_archi.list_attr_obs_tau):
            tmp = LtauBis(name="leap_{}".format(nm_))
            l_tau = l_tau + tmp([lay, el])
            self._qnet_variables += tmp.trainable_weights

        tmp = Dense(self._action_size)
        advantage = tmp(l_tau)
        self._qnet_variables += tmp.trainable_weights
        tmp = Dense(1, name="value")
        value = tmp(l_tau)
        self._qnet_variables += tmp.trainable_weights

        meaner = Lambda(lambda x: K.mean(x, axis=1))
        mn_ = meaner(advantage)
        tmp = subtract([advantage, mn_])
        policy = add([tmp, value], name="policy")

        model_all_outputs = [policy]
        self._model = Model(inputs=models_all_inputs, outputs=model_all_outputs)
        self._schedule_model, self._optimizer_model = self.make_optimiser()
        self._model.compile(loss='mse', optimizer=self._optimizer_model)

        self._target_model = Model(inputs=models_all_inputs, outputs=model_all_outputs)

    def _make_x_tau(self, data):
        # for the x's
        data_x = []
        prev = 0
        for sz, add_, mul_ in zip(self._nn_archi.x_dims,
                                  self._nn_archi.x_adds,
                                  self._nn_archi.x_mults):
            tmp = (data[:, prev:(prev+sz)] + add_) * mul_
            data_x.append(tmp)
            prev += sz

        # for the input of the q network
        data_q = []
        for sz, add_, mul_ in zip(self._nn_archi.input_q_dims,
                                  self._nn_archi.input_q_adds,
                                  self._nn_archi.input_q_mults):
            data_q.append((data[:, prev:(prev+sz)] + add_) * mul_)
            prev += sz

        # for the taus
        data_tau = []
        for sz, add_, mul_ in zip(self._nn_archi.tau_dims,
                                  self._nn_archi.tau_adds,
                                  self._nn_archi.tau_mults):
            data_tau.append((data[:, prev:(prev+sz)] + add_) * mul_)
            prev += sz

        # TODO pre process that into different vector
        data_topo = self._process_topo(data[:, prev:(prev+self._nn_archi.dim_topo)])

        prev += self._nn_archi.dim_topo
        # TODO predict also gen_q and load_v here, and p_or and q_or and p_ex and q_ex
        data_flow = []
        for sz, add_, mul_ in zip(self._nn_archi.gm_out_dims,
                                  self._nn_archi.gm_out_adds,
                                  self._nn_archi.gm_out_mults):
            data_flow.append((data[:, prev:(prev+sz)] + add_) * mul_)
            prev += sz

        res = [*data_x, *data_q, *data_tau, data_topo], data_flow
        return res

    def _process_topo(self, topo_vect):
        """process the topology vector.

         As input grid2op encode it:
         
         - -1 disconnected
         - 1 connected to bus 1
         - 2 connected to bus 2

         I transform it in a vector having twice as many component with the encoding, if we move
         "by pairs":
         
         - [1,0] -> disconnected
         - [0,0] -> connected to bus 1  # normal situation
         - [0,1] -> connected to bus 2
         """
        res = np.zeros((topo_vect.shape[0], 2*topo_vect.shape[1]),
                       dtype=np.float32)
        tmp_ = np.where(topo_vect == -1.)
        res[tmp_[0], 2*tmp_[1]] = 1.
        tmp_ = np.where(topo_vect == 2.)
        res[tmp_[0], 2*tmp_[1]+1] = 1.
        return res

    def predict_movement(self, data, epsilon, batch_size=None, training=False):
        """Predict movement of game controller where is epsilon
        probability randomly move."""
        if batch_size is None:
            batch_size = data.shape[0]
        data_nn, true_output_grid = self._make_x_tau(data)
        res = super().predict_movement(data_nn, epsilon=epsilon, batch_size=batch_size, training=training)
        return res

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, tf_writer=None, batch_size=None):
        if batch_size is None:
            batch_size = s_batch.shape[0]
        data_nn, true_output_grid = self._make_x_tau(s_batch)
        data_nn2, true_output_grid2 = self._make_x_tau(s2_batch)

        # train the grid model to accurately predict the state of the grid
        # TODO predict also gen_q and load_v here, and p_or and q_or and p_ex and q_ex
        loss1 = self.grid_model.train_on_batch(data_nn, true_output_grid)
        loss2 = self.grid_model.train_on_batch(data_nn2, true_output_grid2)

        # and now train the q network
        res = super().train(data_nn,
                            a_batch,
                            r_batch,
                            d_batch,
                            data_nn2,
                            tf_writer=tf_writer,
                            batch_size=batch_size)

        self.grid_model_losses_npy = 0.5*(np.array(loss1) + np.array(loss2))
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
        grads = tape.gradient(loss, self._qnet_variables)

        # clip gradients
        if self._max_global_norm_grad is not None:
            grads, _ = tf.clip_by_global_norm(grads, self._max_global_norm_grad)
        if self._max_value_grad is not None:
            grads = [tf.clip_by_value(grad, -self._max_value_grad, self._max_value_grad)
                     for grad in grads]

        # Apply gradients
        optimizer_model.apply_gradients(zip(grads, self._qnet_variables))
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

    def save_tensorboard(self, current_step):
        if self.grid_model_losses_npy is not None:
            for i, el in enumerate(self._nn_archi.list_attr_obs_gm_out):
                tf.summary.scalar("loss_gridmodel_{}".format(el),
                                  self.grid_model_losses_npy[i],
                                  current_step,
                                  description="Loss of the neural network representing the powergrid "
                                              "for predicting {}"
                                              "".format(el))

    @staticmethod
    def _get_path_model(path, name=None):
        if name is None:
            path_model = path
        else:
            path_model = os.path.join(path, name)
        path_target_model = "{}_target".format(path_model)
        path_grid_model = "{}_grid_model".format(path_model)
        return path_model, path_target_model, path_grid_model

    def save_network(self, path, name=None, ext="h5"):
        """
        Saves all the models with unique names
        """
        path_model, path_target_model, path_grid_model = self._get_path_model(path, name)
        self._model.save('{}.{}'.format(path_model, ext))
        self._target_model.save('{}.{}'.format(path_target_model, ext))
        self.grid_model.save('{}.{}'.format(path_grid_model, ext))

    def load_network(self, path, name=None, ext="h5"):
        """
        We load all the models using the keras "load_model" function.
        """
        path_model, path_target_model, path_grid_model = self._get_path_model(path, name)
        self.construct_q_network()
        self._model.load_weights('{}.{}'.format(path_model, ext))
        self._target_model.load_weights('{}.{}'.format(path_target_model, ext))
        self.grid_model.load_weights('{}.{}'.format(path_grid_model, ext))
        if self.verbose:
            print("Succesfully loaded network.")