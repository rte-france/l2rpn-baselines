# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import numpy as np

from grid2op.Agent import BaseAgent


class RLAgent(BaseAgent):
    def __init__(self, name):
        raise NotImplementedError()
        self.name = name

        self._training_param = None
        self._tf_writer = None

        self.brain = None  # the "stuff" that takes the decisions

        self._max_iter_env_ = 1000000
        self._curr_iter_env = 0
        self._max_reward = 0.

        # action type
        self.nb_injection = 0
        self.nb_voltage = 0
        self.nb_topology = 0
        self.nb_line = 0
        self.nb_redispatching = 0
        self.nb_do_nothing = 0

        # for over sampling the hard scenarios
        self._prev_obs_num = 0
        self._time_step_lived = None
        self._nb_chosen = None
        self._proba = None
        self._prev_id = 0
        # this is for the "limit the episode length" depending on your previous success
        self._total_sucesses = 0

    # BaseAgent interface
    def act(self, obs, reward, done=False):
        act = self.brain.predict(obs, reward, done, train=False)
        return act

    # Baseline interface
    def load(self, path):
        """
        Part of the l2rpn_baselines interface, this function allows to read back a trained model, to continue the
        training or to evaluate its performance for example.

        **NB** To reload an agent, it must have exactly the same name and have been saved at the right location.

        Parameters
        ----------
        path: ``str``
            The path where the agent has previously beens saved.

        """
        # not modified compare to original implementation
        tmp_me = os.path.join(path, self.name)
        if not os.path.exists(tmp_me):
            raise RuntimeError("The model should be stored in \"{}\". But this appears to be empty".format(tmp_me))
        self._load_action_space(tmp_me)

        # TODO handle case where training param class has been overidden
        self._training_param = TrainingParam.from_json(os.path.join(tmp_me, "training_params.json".format(self.name)))
        self.deep_q = self._nn_archi.make_nn(self._training_param)
        try:
            self.deep_q.load_network(tmp_me, name=self.name)
        except Exception as e:
            raise RuntimeError("Impossible to load the model located at \"{}\" with error \n{}".format(path, e))

        for nm_attr in ["_time_step_lived", "_nb_chosen", "_proba"]:
            conv_path = os.path.join(tmp_me, "{}.npy".format(nm_attr))
            if os.path.exists(conv_path):
                setattr(self, nm_attr, np.load(file=conv_path))