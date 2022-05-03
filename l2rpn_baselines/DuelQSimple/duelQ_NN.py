# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

# tf2.0 friendly
import warnings

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Activation, Dense
        from tensorflow.keras.layers import Input, Lambda, subtract, add
        import tensorflow.keras.backend as K
    _CAN_USE_TENSORFLOW = True
except ImportError:
    _CAN_USE_TENSORFLOW = False

from l2rpn_baselines.utils import BaseDeepQ, TrainingParam


class DuelQ_NN(BaseDeepQ):
    """Constructs the desired duelling deep q learning network
    
    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
    
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
        if self._action_size == 0:
            raise RuntimeError("Impossible to make a DeepQ network with an action space of size 0!")
        self.construct_q_network()

    def construct_q_network(self):
        """
        It uses the architecture defined in the `nn_archi` attributes.

        """
        self._model = Sequential()
        input_layer = Input(shape=(self._nn_archi.observation_size,),
                            name="observation")

        lay = input_layer
        for lay_num, (size, act) in enumerate(zip(self._nn_archi.sizes, self._nn_archi.activs)):
            lay = Dense(size, name="layer_{}".format(lay_num))(lay)  # put at self.action_size
            lay = Activation(act)(lay)

        fc1 = Dense(self._action_size)(lay)
        advantage = Dense(self._action_size, name="advantage")(fc1)

        fc2 = Dense(self._action_size)(lay)
        value = Dense(1, name="value")(fc2)

        meaner = Lambda(lambda x: K.mean(x, axis=1))
        mn_ = meaner(advantage)
        tmp = subtract([advantage, mn_])
        policy = add([tmp, value], name="policy")

        self._model = Model(inputs=[input_layer], outputs=[policy])
        self._schedule_model, self._optimizer_model = self.make_optimiser()
        self._model.compile(loss='mse', optimizer=self._optimizer_model)

        self._target_model = Model(inputs=[input_layer], outputs=[policy])
