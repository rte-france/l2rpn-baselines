# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import numpy as np
from grid2op.Agent import AgentWithConverter


class DeepQAgent(AgentWithConverter):

    def convert_obs(self, observation):
        return np.concatenate((observation.rho, observation.line_status, observation.topo_vect))

    def my_act(self, transformed_observation, reward, done=False):
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)
        predict_movement_int, *_ = self.deep_q.predict_movement(transformed_observation.reshape(1, -1), epsilon=0.0)
        return int(predict_movement_int)

    def init_deep_q(self, transformed_observation):
        if self.deep_q is None:
            # the first time an observation is observed, I set up the neural network with the proper dimensions.
            if self.mode == "DQN":
                cls = DeepQ
            elif self.mode == "DDQN":
                cls = DuelQ
            elif self.mode == "SAC":
                cls = SAC
            else:
                raise RuntimeError("Unknown neural network named \"{}\". Supported types are \"DQN\", \"DDQN\" and "
                                   "\"SAC\"".format(self.mode))
            self.deep_q = cls(self.action_space.size(), observation_size=transformed_observation.shape[-1], lr=self.lr)

    def __init__(self, action_space, mode="DDQN", lr=1e-5, training_param=TrainingParam()):
        # this function has been adapted.

        # to built a AgentWithConverter, we need an action_space.
        # No problem, we add it in the constructor.
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        # and now back to the origin implementation
        self.replay_buffer = ReplayBuffer(training_param.BUFFER_SIZE)

        # compare to original implementation, i don't know the observation space size.
        # Because it depends on the component of the observation we want to look at. So these neural network will
        # be initialized the first time an observation is observe.
        self.deep_q = None
        self.mode = mode
        self.lr = lr
        self.training_param = training_param

    def load_network(self, path):
        # not modified compare to original implementation
        self.deep_q.load_network(path)