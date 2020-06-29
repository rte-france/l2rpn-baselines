# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf

import grid2op
from grid2op.Exceptions import Grid2OpException
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from l2rpn_baselines.utils.ReplayBuffer import ReplayBuffer
from l2rpn_baselines.utils.TrainingParam import TrainingParam

try:
    from grid2op.Chronics import MultifolderWithCache
    _CACHE_AVAILABLE_DEEPQAGENT = True
except ImportError:
    _CACHE_AVAILABLE_DEEPQAGENT = False


class DeepQAgent(AgentWithConverter):
    """
    This class allows to train and log the training of different Q learning algorithm.

    It is not meant to be the state of the art implement of some baseline. It is rather meant to be a set of
    useful functions that allows to easily develop an environment if we want to get started in RL using grid2op.

    It derives from :class:`grid2op.Agent.AgentWithConverter` and as such implements the :func:`DeepQAgent.convert_obs`
    and :func:`DeepQAgent.my_act`

    It is suppose to be a Baseline, so it implements also the

    - :func:`DeepQAgent.load`: to load the agent
    - :func:`DeepQAgent.save`: to save the agent
    - :func:`DeepQAgent.train`: to train the agent

    TODO description of the training scheme!

    Attributes
    ----------
    filter_action_fun: ``callable``
        The function used to filter the action of the action space. See the documentation of grid2op:
        :class:`grid2op.Converter.IdToAct`
        `here <https://grid2op.readthedocs.io/en/v0.9.3/converter.html#grid2op.Converter.IdToAct>`_ for more
        information.

    replay_buffer:
        The experience replay buffer

    deep_q: :class:`BaseDeepQ`
        The neural network, represented as a :class:`BaseDeepQ` object.

    name: ``str``
        The name of the Agent

    store_action: ``bool``
        Whether you want to register which action your agent took or not. Saving the action can slow down a bit
        the computation (less than 1%) but can help understand what your agent is doing during its learning process.

    dict_action: ``str``
        The action taken by the agent, represented as a dictionnary. This can be useful to know which type of actions
        is taken by your agent. Only filled if :attr:DeepQAgent.store_action` is ``True``

    istraining: ``bool``
        Whether you are training this agent or not. No more really used. Mainly used for backward compatibility.

    epsilon: ``float``
        The epsilon greedy exploration parameter.

    nb_injection: ``int``
        Number of action tagged as "injection". See the
        `official grid2op documentation <https://grid2op.readthedocs.io/en/v0.9.3/action.html?highlight=get_types#grid2op.Action.BaseAction.get_types>`_
        for more information.

    nb_voltage: ``int``
        Number of action tagged as "voltage". See the
        `official grid2op documentation <https://grid2op.readthedocs.io/en/v0.9.3/action.html?highlight=get_types#grid2op.Action.BaseAction.get_types>`_
        for more information.

    nb_topology: ``int``
        Number of action tagged as "topology". See the
        `official grid2op documentation <https://grid2op.readthedocs.io/en/v0.9.3/action.html?highlight=get_types#grid2op.Action.BaseAction.get_types>`_
        for more information.

    nb_redispatching: ``int``
        Number of action tagged as "redispatching". See the
        `official grid2op documentation <https://grid2op.readthedocs.io/en/v0.9.3/action.html?highlight=get_types#grid2op.Action.BaseAction.get_types>`_
        for more information.

    nb_do_nothing: ``int``
        Number of action tagged as "do_nothing", *ie* when an action is not modifiying the state of the grid. See the
        `official grid2op documentation <https://grid2op.readthedocs.io/en/v0.9.3/action.html?highlight=get_types#grid2op.Action.BaseAction.get_types>`_
        for more information.

    verbose: ``bool``
        An effort will be made on the logging (outside of trensorboard) of the training. For now: verbose=True will
        allow some printing on the command prompt, and verbose=False will drastically reduce the amount of information
        printed during training.

    """
    def __init__(self,
                 action_space,
                 nn_archi,
                 name="DeepQAgent",
                 store_action=True,
                 istraining=False,
                 filter_action_fun=None,
                 verbose=False,
                 observation_space=None,
                 **kwargs_converters):
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct, **kwargs_converters)
        self.filter_action_fun = filter_action_fun
        if self.filter_action_fun is not None:
            self.action_space.filter_action(self.filter_action_fun)

        # and now back to the origin implementation
        self.replay_buffer = None
        self.__nb_env = None

        self.deep_q = None
        self._training_param = None
        self._tf_writer = None
        self.name = name
        self._losses = None
        self.__graph_saved = False
        self.store_action = store_action
        self.dict_action = {}
        self.istraining = istraining
        self._actions_per_1000steps = np.zeros((1000, self.action_space.size()), dtype=np.int)
        self._illegal_actions_per_1000steps = np.zeros(1000, dtype=np.int)
        self._ambiguous_actions_per_1000steps = np.zeros(1000, dtype=np.int)
        self.epsilon = 1.0

        # for tensorbaord
        self._train_lr = None

        self._reset_num = None

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

        # update frequency of action types
        self._nb_updated_act_tensorboard = None

        # neural network architecture
        self._nn_archi = nn_archi

        # observation tranformers
        self._obs_as_vect = None
        self._tmp_obs = None
        self._indx_obs = None
        self.verbose = verbose
        if observation_space is None:
            pass
        else:
            self.init_obs_extraction(observation_space)

    # grid2op.Agent interface
    def convert_obs(self, observation):
        """
        Generic way to convert an observation. This transform it to a vector and the select the attributes that were
        selected in :attr:`l2rpn_baselines.utils.NNParams.list_attr_obs` (that have been extracted once and for all
        in the :attr:`DeepQAgent._indx_obs` vector).

        Parameters
        ----------
        observation: :class:`grid2op.Observation.BaseObservation`
            The current observation sent by the environment

        Returns
        -------
        _tmp_obs: ``numpy.ndarray``
            The observation as vector with only the proper attribute selected (TODO scaling will be available
            in future version)

        """
        obs_as_vect = observation.to_vect()
        self._tmp_obs[:] = obs_as_vect[self._indx_obs]
        return self._tmp_obs

    def my_act(self, transformed_observation, reward, done=False):
        """
        This function will return the action (its id) selected by the underlying :attr:`DeepQAgent.deep_q` network.

        Before being used, this method require that the :attr:`DeepQAgent.deep_q` is created. To that end a call
        to :func:`DeepQAgent.init_deep_q` needs to have been performed (this is automatically done if you use
        baseline we provide and their `evaluate` and `train` scripts).

        Parameters
        ----------
        transformed_observation: ``numpy.ndarray``
            The observation, as transformed after :func:`DeepQAgent.convert_obs`

        reward: ``float``
            The reward of the last time step. Ignored by this method. Here for retro compatibility with openAI
            gym interface.

        done: ``bool``
            Whether the episode is over or not. This is not used, and is only present to be compliant with
            open AI gym interface

        Returns
        -------
        res: ``int``
            The id the action taken.

        """
        predict_movement_int, *_ = self.deep_q.predict_movement(transformed_observation, epsilon=0.0)
        res = int(predict_movement_int)
        self._store_action_played(res)
        return res

    @staticmethod
    def get_action_size(action_space, filter_fun, kwargs_converters):
        """
        This function allows to get the size of the action space if we were to built a :class:`DeepQAgent`
        with this parameters.

        Parameters
        ----------
        action_space: :class:`grid2op.ActionSpace`
            The grid2op action space used.

        filter_fun: ``callable``
            see :attr:`DeepQAgent.filter_fun` for more information

        kwargs_converters: ``dict``
            see the documentation of grid2op for more information:
            `here <https://grid2op.readthedocs.io/en/v0.9.3/converter.html?highlight=idToAct#grid2op.Converter.IdToAct.init_converter>`_


        See Also
        --------
            The official documentation of grid2Op, and especially its class "IdToAct" at this address
            `IdToAct <https://grid2op.readthedocs.io/en/v0.9.3/converter.html?highlight=idToAct#grid2op.Converter.IdToAct>`_

        """
        converter = IdToAct(action_space)
        converter.init_converter(**kwargs_converters)
        if filter_fun is not None:
            converter.filter_action(filter_fun)
        return converter.n

    def init_obs_extraction(self, observation_space):
        """
        This method should be called to initialize the observation (feed as a vector in the neural network)
        from its description as a list of its attribute names.
        """
        tmp = np.zeros(0, dtype=np.uint)  # TODO platform independant
        for obs_attr_name in self._nn_archi.get_obs_attr():
            beg_, end_, dtype_ = observation_space.get_indx_extract(obs_attr_name)
            tmp = np.concatenate((tmp, np.arange(beg_, end_, dtype=np.uint)))
        self._indx_obs = tmp
        self._tmp_obs = np.zeros((1, tmp.shape[0]), dtype=np.float32)

    # baseline interface
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

    def save(self, path):
        """
        Part of the l2rpn_baselines interface, this allows to save a model. Its name is used at saving time. The
        same name must be reused when loading it back.

        Parameters
        ----------
        path: ``str``
            The path where to save the agent.

        """
        if path is not None:
            tmp_me = os.path.join(path, self.name)
            if not os.path.exists(tmp_me):
                os.mkdir(tmp_me)
            nm_conv = "action_space.npy"
            conv_path = os.path.join(tmp_me, nm_conv)
            if not os.path.exists(conv_path):
                self.action_space.save(path=tmp_me, name=nm_conv)

            self._training_param.save_as_json(tmp_me, name="training_params.json")
            self._nn_archi.save_as_json(tmp_me, "nn_architecture.json")
            self.deep_q.save_network(tmp_me, name=self.name)

            # TODO save the "oversampling" part, and all the other info
            for nm_attr in ["_time_step_lived", "_nb_chosen", "_proba"]:
                conv_path = os.path.join(tmp_me, "{}.npy".format(nm_attr))
                attr_ = getattr(self, nm_attr)
                if attr_ is not None:
                    np.save(arr=attr_, file=conv_path)

    def train(self,
              env,
              iterations,
              save_path,
              logdir,
              training_param=None):
        """
        Part of the public l2rpn-baselines interface, this function allows to train the baseline.

        If `save_path` is not None, the the model is saved regularly, and also at the end of training.

        TODO explain a bit more how you can train it.

        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment` or :class:`grid2op.Environment.MultiEnvironment`
            The environment used to train your model.

        iterations: ``int``
            The number of training iteration. NB when reloading a model, this is **NOT** the training steps that will
            be used when re training. Indeed, if `iterations` is 1000 and the model was already trained for 750 time
            steps, then when reloaded, the training will occur on 250 (=1000 - 750) time steps only.

        save_path: ``str``
            Location at which to save the model

        logdir: ``str``
            Location at which tensorboard related information will be kept.

        training_param: :class:`l2rpn_baselines.utils.TrainingParam`
            The meta parameters for the training procedure. This is currently ignored if the model is reloaded (in that
            case the parameters used when first created will be used)

        """

        if training_param is None:
            training_param = TrainingParam()

        self._train_lr = training_param.lr

        if self._training_param is None:
            self._training_param = training_param
        else:
            training_param = self._training_param
        self._init_deep_q(self._training_param, env)

        self._init_replay_buffer()

        # efficient reading of the data (read them by chunk of roughly 1 day
        nb_ts_one_day = 24 * 60 / 5  # number of time steps per day
        self._set_chunk(env, nb_ts_one_day)

        # Create file system related vars
        if save_path is not None:
            save_path = os.path.abspath(save_path)
            os.makedirs(save_path, exist_ok=True)

        if logdir is not None:
            logpath = os.path.join(logdir, self.name)
            self._tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        else:
            logpath = None
            self._tf_writer = None
        UPDATE_FREQ = training_param.update_tensorboard_freq  # update tensorboard every "UPDATE_FREQ" steps
        SAVING_NUM = training_param.save_model_each

        if isinstance(env, grid2op.Environment.Environment):
            self.__nb_env = 1
        else:
            import warnings
            nb_env = env.nb_env
            warnings.warn("Training using {} environments".format(nb_env))
            self.__nb_env = nb_env

        self.init_obs_extraction(env.observation_space)

        training_step = self._training_param.last_step

        # some parameters have been move to a class named "training_param" for convenience
        self.epsilon = self._training_param.initial_epsilon

        # now the number of alive frames and total reward depends on the "underlying environment". It is vector instead
        # of scalar
        alive_frame, total_reward = self._init_global_train_loop()
        reward, done = self._init_local_train_loop()
        epoch_num = 0
        self._losses = np.zeros(iterations)
        alive_frames = np.zeros(iterations)
        total_rewards = np.zeros(iterations)
        new_state = None
        self._reset_num = 0
        self._curr_iter_env = 0
        self._max_reward = env.reward_range[1]

        # action types
        # injection, voltage, topology, line, redispatching = action.get_types()
        self.nb_injection = 0
        self.nb_voltage = 0
        self.nb_topology = 0
        self.nb_line = 0
        self.nb_redispatching = 0
        self.nb_do_nothing = 0

        # for non uniform random sampling of the scenarios
        th_size = None
        self._prev_obs_num = 0
        if self.__nb_env == 1:
            # TODO make this available for multi env too
            if _CACHE_AVAILABLE_DEEPQAGENT:
                if isinstance(env.chronics_handler.real_data, MultifolderWithCache):
                    th_size = env.chronics_handler.real_data.cache_size
            if th_size is None:
                th_size = len(env.chronics_handler.real_data.subpaths)

            # number of time step lived per possible scenarios
            if self._time_step_lived is None or self._time_step_lived.shape[0] != th_size:
                self._time_step_lived = np.zeros(th_size, dtype=np.uint64)
            # number of time a given scenario has been played
            if self._nb_chosen is None or self._nb_chosen.shape[0] != th_size:
                self._nb_chosen = np.zeros(th_size, dtype=np.uint)
            # number of time a given scenario has been played
            if self._proba is None or self._proba.shape[0] != th_size:
                self._proba = np.ones(th_size, dtype=np.float64)

        self._prev_id = 0
        # this is for the "limit the episode length" depending on your previous success
        self._total_sucesses = 0
        # update the frequency of action types
        self._nb_updated_act_tensorboard = 0

        with tqdm(total=iterations - training_step, disable=not self.verbose) as pbar:
            while training_step < iterations:
                # reset or build the environment
                initial_state = self._need_reset(env, training_step, epoch_num, done, new_state)

                # Slowly decay the exploration parameter epsilon
                # if self.epsilon > training_param.FINAL_EPSILON:
                self.epsilon = self._training_param.get_next_epsilon(current_step=training_step)

                # then we need to predict the next moves. Agents have been adapted to predict a batch of data
                pm_i, pq_v, act = self._next_move(initial_state, self.epsilon)

                # todo store the illegal / ambiguous / ... actions
                reward, done = self._init_local_train_loop()
                if self.__nb_env == 1:
                    # still the "hack" to have same interface between multi env and env...
                    # yeah it's a pain
                    act = act[0]

                temp_observation_obj, temp_reward, temp_done, info = env.step(act)

                if self.__nb_env == 1:
                    # dirty hack to wrap them into list
                    temp_observation_obj = [temp_observation_obj]
                    temp_reward = np.array([temp_reward], dtype=np.float32)
                    temp_done = np.array([temp_done], dtype=np.bool)
                    info = [info]

                new_state = self._convert_obs_train(temp_observation_obj)
                self._updage_illegal_ambiguous(training_step, info)
                done, reward, total_reward, alive_frame, epoch_num \
                    = self._update_loop(done, temp_reward, temp_done, alive_frame, total_reward, reward, epoch_num)

                # update the replay buffer
                self._store_new_state(initial_state, pm_i, reward, done, new_state)

                # now train the model
                if not self._train_model(training_step):
                    # infinite loss in this case
                    raise RuntimeError("ERROR INFINITE LOSS")

                # Save the network every 1000 iterations
                if training_step % SAVING_NUM == 0 or training_step == iterations - 1:
                    self.save(save_path)

                # save some information to tensorboard
                alive_frames[epoch_num] = np.mean(alive_frame)
                total_rewards[epoch_num] = np.mean(total_reward)
                self._store_action_played_train(training_step, pm_i)

                self._save_tensorboard(training_step, epoch_num, UPDATE_FREQ, total_rewards, alive_frames)
                training_step += 1
                pbar.update(1)

        self.save(save_path)

    # auxiliary functions
    # two below function: to train with multiple environments
    def _convert_obs_train(self, observations):
        """ create the observations that are used for training."""
        if self._obs_as_vect is None:
            size_obs = self.convert_obs(observations[0]).shape[1]
            self._obs_as_vect = np.zeros((self.__nb_env, size_obs), dtype=np.float32)

        for i, obs in enumerate(observations):
            self._obs_as_vect[i, :] = self.convert_obs(obs).reshape(-1)
        return self._obs_as_vect

    def _store_action_played(self, action_int):
        """if activated, this function will store the action taken by the agent."""
        if self.store_action:
            if action_int not in self.dict_action:
                act = self.action_space.all_actions[action_int]
                is_inj, is_volt, is_topo, is_line_status, is_redisp, is_dn = False, False, False, False, False, False
                try:
                    # feature unavailble in grid2op <= 0.9.2
                    is_inj, is_volt, is_topo, is_line_status, is_redisp = act.get_types()
                    is_dn = (not is_inj) and (not is_volt) and (not is_topo) and (not is_line_status) and (not is_redisp)
                except Exception as exc_:
                    pass

                self.dict_action[action_int] = [0, act,
                                                (is_inj, is_volt, is_topo, is_line_status, is_redisp, is_dn)]
                self.dict_action[action_int][0] += 1

                (is_inj, is_volt, is_topo, is_line_status, is_redisp, is_dn) = self.dict_action[action_int][2]
                if is_inj:
                    self.nb_injection += 1
                if is_volt:
                    self.nb_voltage += 1
                if is_topo:
                    self.nb_topology += 1
                if is_line_status:
                    self.nb_line += 1
                if is_redisp:
                    self.nb_redispatching += 1
                if is_dn:
                    self.nb_do_nothing += 1

    def _convert_all_act(self, act_as_integer):
        """this function converts the action given as a list of integer. It ouputs a list of valid grid2op Action"""
        res = []
        for act_id in act_as_integer:
            res.append(self.convert_act(act_id))
            self._store_action_played(act_id)
        return res

    def _load_action_space(self, path):
        """ load the action space in case the model is reloaded"""
        if not os.path.exists(path):
            raise RuntimeError("The model should be stored in \"{}\". But this appears to be empty".format(path))
        try:
            self.action_space.init_converter(
                all_actions=os.path.join(path, "action_space.npy".format(self.name)))
        except Exception as e:
            raise RuntimeError("Impossible to reload converter action space with error \n{}".format(e))

    # utilities for data reading
    def _set_chunk(self, env, nb):
        """
        to optimize the data reading process. See the official grid2op documentation for the effect of setting
        the chunk size for the environment.
        """
        env.set_chunk_size(int(max(100, nb)))

    def _train_model(self, training_step):
        """train the deep q networks."""
        self._training_param.tell_step(training_step)
        if training_step > max(self._training_param.min_observation, self._training_param.minibatch_size) and \
            self._training_param.do_train():
            # train the model
            s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(self._training_param.minibatch_size)
            tf_writer = None
            if self.__graph_saved is False:
                tf_writer = self._tf_writer
            loss = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch,
                                     tf_writer)
            # save learning rate for later
            self._train_lr = self.deep_q._optimizer_model._decayed_lr('float32').numpy()
            self.__graph_saved = True
            if not np.all(np.isfinite(loss)):
                # if the loss is not finite i stop the learning
                return False
            self.deep_q.target_train()
            self._losses[training_step:] = np.sum(loss)
        return True

    def _updage_illegal_ambiguous(self, curr_step, info):
        """update the conunt of illegal and ambiguous actions"""
        self._illegal_actions_per_1000steps[curr_step % 1000] = np.sum([el["is_illegal"] for el in info])
        self._ambiguous_actions_per_1000steps[curr_step % 1000] = np.sum([el["is_ambiguous"] for el in info])

    def _store_action_played_train(self, training_step, action_id):
        """store which action were played, for tensorboard only."""
        which_row = training_step % 1000
        self._actions_per_1000steps[which_row, :] = 0
        self._actions_per_1000steps[which_row, action_id] += 1

    def _fast_forward_env(self, env, time=7*24*60/5):
        """use this functio to skip some time steps when environment is reset."""
        my_int = np.random.randint(0, min(time, env.chronics_handler.max_timestep()))
        env.fast_forward_chronics(my_int)

    def _reset_env_clean_state(self, env):
        """
        reset this environment to a proper state. This should rather be integrated in grid2op. And will probably
        be integrated partially starting from grid2op 1.0.0
        """
        # /!\ DO NOT ATTEMPT TO MODIFY OTHERWISE IT WILL PROBABLY CRASH /!\
        # /!\ THIS WILL BE PART OF THE ENVIRONMENT IN FUTURE GRID2OP RELEASE (>= 1.0.0) /!\
        # AND OF COURSE USING THIS METHOD DURING THE EVALUATION IS COMPLETELY FORBIDDEN
        if self.__nb_env > 1:
            return
        env.current_obs = None
        env.env_modification = None
        env._reset_maintenance()
        env._reset_redispatching()
        env._reset_vectors_and_timings()
        _backend_action = env._backend_action_class()
        _backend_action.all_changed()
        env._backend_action =_backend_action
        env.backend.apply_action(_backend_action)
        _backend_action.reset()
        *_, fail_to_start, info = env.step(env.action_space())
        if fail_to_start:
            # this is happening because not enough care has been taken to handle these problems
            # more care will be taken when this feature will be available in grid2op directly.
            raise Grid2OpException("Impossible to initialize the powergrid, the powerflow diverge at iteration 0. "
                                   "Available information are: {}".format(info))
        env._reset_vectors_and_timings()

    def _need_reset(self, env, observation_num, epoch_num, done, new_state):
        """perform the proper reset of the environment"""
        if self._training_param.step_increase_nb_iter > 0:
            self._max_iter_env(min(max(self._training_param.min_iter,
                                       self._training_param.max_iter_fun(self._total_sucesses)),
                                   self._training_param.max_iter))  # TODO
        self._curr_iter_env += 1
        if new_state is None:
            # it's the first ever loop
            obs = env.reset()
            if self.__nb_env == 1:
                # still hack to have same program interface between multi env and not multi env
                obs = [obs]
            new_state = self._convert_obs_train(obs)
        elif self.__nb_env > 1:
            # in multi env this is automatically handled
            pass
        elif done[0]:
            nb_ts_one_day = 24*60/5
            if False:
                # the 3-4 lines below allow to reuse the loaded dataset and continue further up in the
                try:
                    self._reset_env_clean_state(env)
                    # random fast forward between now and next day
                    self._fast_forward_env(env, time=nb_ts_one_day)
                except (StopIteration, Grid2OpException):
                    env.reset()
                    # random fast forward between now and next week
                    self._fast_forward_env(env, time=7*nb_ts_one_day)

            # update the number of time steps it has live
            ts_lived = observation_num - self._prev_obs_num
            if self._time_step_lived is not None:
                self._time_step_lived[self._prev_id] += ts_lived
            self._prev_obs_num = observation_num
            if self._training_param.oversampling_rate is not None:
                # proba = np.sqrt(1. / (self._time_step_lived +1))
                # # over sampling some kind of "UCB like" stuff
                # # https://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/

                # proba = 1. / (self._time_step_lived + 1)
                self._proba[:] = 1. / (self._time_step_lived ** self._training_param.oversampling_rate + 1)
                self._proba /= np.sum(self._proba)

            _prev_id = self._prev_id
            self._prev_id = None
            if _CACHE_AVAILABLE_DEEPQAGENT:
                if isinstance(env.chronics_handler.real_data, MultifolderWithCache):
                    self._prev_id = env.chronics_handler.real_data.sample_next_chronics(self._proba)
            if self._prev_id is None:
                self._prev_id = _prev_id + 1
                self._prev_id %= self._time_step_lived.shape[0]

            env.reset()
            if self._nb_chosen is not None:
                self._nb_chosen[self._prev_id] += 1

            # random fast forward between now and next week
            if self._training_param.random_sample_datetime_start is not None:
                self._fast_forward_env(env, time=self._training_param.random_sample_datetime_start)

            self._curr_iter_env = 0
            obs = [env.current_obs]
            new_state = self._convert_obs_train(obs)
            if epoch_num % len(env.chronics_handler.real_data.subpaths) == 0:
                # re shuffle the data
                env.chronics_handler.shuffle(lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
        return new_state

    def _init_replay_buffer(self):
        """create and initialized the replay buffer"""
        self.replay_buffer = ReplayBuffer(self._training_param.buffer_size)

    def _store_new_state(self, initial_state, predict_movement_int, reward, done, new_state):
        """store the new state in the replay buffer"""
        # vectorized version of the previous code
        for i_s, pm_i, reward, done, new_state in zip(initial_state, predict_movement_int, reward, done, new_state):
            self.replay_buffer.add(i_s,
                                   pm_i,
                                   reward,
                                   done,
                                   new_state)

    def _max_iter_env(self, new_max_iter):
        """update the number of maximum iteration allowed."""
        self._max_iter_env_ = new_max_iter

    def _next_move(self, curr_state, epsilon):
        pm_i, pq_v = self.deep_q.predict_movement(curr_state, epsilon)
        act = self._convert_all_act(pm_i)
        return pm_i, pq_v, act

    def _init_global_train_loop(self):
        alive_frame = np.zeros(self.__nb_env, dtype=np.int)
        total_reward = np.zeros(self.__nb_env, dtype=np.float32)
        return alive_frame, total_reward

    def _update_loop(self, done, temp_reward, temp_done, alive_frame, total_reward, reward, epoch_num):
        if self.__nb_env == 1:
            # force end of episode at early stage of learning
            if self._curr_iter_env >= self._max_iter_env_:
                temp_done[0] = True
                temp_reward[0] = self._max_reward
                self._total_sucesses += 1

        done = temp_done
        alive_frame[done] = 0
        total_reward[done] = 0.
        self._reset_num += np.sum(done)
        if self._reset_num >= self.__nb_env:
            # increase the "global epoch num" represented by "epoch_num" only when on average
            # all environments are "done"
            epoch_num += 1
            self._reset_num = 0

        total_reward[~done] += temp_reward[~done]
        alive_frame[~done] += 1
        return done, temp_reward, total_reward, alive_frame, epoch_num

    def _init_local_train_loop(self):
        # reward, done = np.zeros(self.nb_process), np.full(self.nb_process, fill_value=False, dtype=np.bool)
        reward = np.zeros(self.__nb_env, dtype=np.float32)
        done = np.full(self.__nb_env, fill_value=False, dtype=np.bool)
        return reward, done

    def _init_deep_q(self, training_param, env):
        """
        This function serves as initializin the neural network.
        """
        if self.deep_q is None:
            self.deep_q = self._nn_archi.make_nn(training_param)
        self.init_obs_extraction(env.observation_space)

    def _save_tensorboard(self, step, epoch_num, UPDATE_FREQ, epoch_rewards, epoch_alive):
        """save all the informations needed in tensorboard."""
        if self._tf_writer is None:
            return

        # Log some useful metrics every even updates
        if step % UPDATE_FREQ == 0 and epoch_num > 0:
            if step % (10 * UPDATE_FREQ) == 0:
                # print the top k scenarios the "hardest" (ie chosen the most number of times
                if self.verbose:
                    top_k = 10
                    if self._nb_chosen is not None:
                        array_ = np.argsort(self._nb_chosen)[-top_k:][::-1]
                        print("hardest scenarios\n{}".format(array_))
                        print("They have been chosen respectively\n{}".format(self._nb_chosen[array_]))
                        # print("Associated proba are\n{}".format(self._proba[array_]))
                        print("The number of timesteps played is\n{}".format(self._time_step_lived[array_]))
                        print("avg (accross all scenarios) number of timsteps played {}"
                              "".format(np.mean(self._time_step_lived)))
                        print("Time alive: {}".format(self._time_step_lived[array_] / (self._nb_chosen[array_] + 1)))
                        print("Avg time alive: {}".format(np.mean(self._time_step_lived / (self._nb_chosen + 1 ))))

            with self._tf_writer.as_default():
                last_alive = epoch_alive[(epoch_num-1)]
                last_reward = epoch_rewards[(epoch_num-1)]

                mean_reward = np.nanmean(epoch_rewards[:epoch_num])
                mean_alive = np.nanmean(epoch_alive[:epoch_num])

                mean_reward_30 = mean_reward
                mean_alive_30 = mean_alive
                mean_reward_100 = mean_reward
                mean_alive_100 = mean_alive

                tmp = self._actions_per_1000steps > 0
                tmp = tmp.sum(axis=0)
                nb_action_taken_last_1000_step = np.sum(tmp > 0)

                nb_illegal_act = np.sum(self._illegal_actions_per_1000steps)
                nb_ambiguous_act = np.sum(self._ambiguous_actions_per_1000steps)

                if epoch_num >= 100:
                    mean_reward_100 = np.nanmean(epoch_rewards[(epoch_num-100):epoch_num])
                    mean_alive_100 = np.nanmean(epoch_alive[(epoch_num-100):epoch_num])

                if epoch_num >= 30:
                    mean_reward_30 = np.nanmean(epoch_rewards[(epoch_num-30):epoch_num])
                    mean_alive_30 = np.nanmean(epoch_alive[(epoch_num-30):epoch_num])

                # to ensure "fair" comparison between single env and multi env
                step_tb = step  # * self.__nb_env
                # if multiply by the number of env we have "trouble" with random exploration at the beginning
                # because it lasts the same number of "real" steps

                # show first the Mean reward and mine time alive (hence the upper case)
                tf.summary.scalar("Mean_alive_30", mean_alive_30, step_tb,
                                  description="Mean reward over the last 30 epochs")
                tf.summary.scalar("Mean_reward_30", mean_reward_30, step_tb,
                                  description="Mean number of timesteps sucessfully manage the last 30 epochs")

                # then it's alpha numerical order, hence the "z_" in front of some information
                tf.summary.scalar("loss", self._losses[step], step_tb,
                                  description="last training loss")

                tf.summary.scalar("last_alive", last_alive, step_tb,
                                  description="last number of timestep during which the agent stayed alive")
                tf.summary.scalar("last_reward", last_reward, step_tb,
                                  description="last reward get by the agent")

                tf.summary.scalar("mean_reward", mean_reward, step_tb)
                tf.summary.scalar("mean_alive", mean_alive, step_tb)

                tf.summary.scalar("mean_reward_100", mean_reward_100, step_tb,
                                  description="Mean reward over the last 100 epochs")
                tf.summary.scalar("mean_alive_100", mean_alive_100, step_tb,
                                  description="Mean number of timesteps sucessfully manage the last 100 epochs")

                tf.summary.scalar("nb_differentaction_taken_1000", nb_action_taken_last_1000_step, step_tb,
                                  description="Number of different actions played the past 1000 steps")
                tf.summary.scalar("nb_illegal_act", nb_illegal_act, step_tb,
                                  description="Number of illegal actions played the past 1000 steps")
                tf.summary.scalar("nb_ambiguous_act", nb_ambiguous_act, step_tb,
                                  description="Number of ambiguous actions played the past 1000 steps")
                tf.summary.scalar("nb_total_success", self._total_sucesses, step_tb,
                                  description="Number of times I reach the end of scenario (no game over)")

                tf.summary.scalar("z_lr", self._train_lr, step_tb,
                                  description="current learning rate")
                tf.summary.scalar("z_epsilon", self.epsilon, step_tb,
                                  description="current epsilon (of the epsilon greedy)")
                tf.summary.scalar("z_max_iter", self._max_iter_env_, step_tb,
                                  description="maximum number of time steps before deciding a scenario is over (=win)")
                tf.summary.scalar("z_total_episode", epoch_num, step_tb,
                                  description="total number of episode played (~number of \"reset\")")

                if self.store_action:
                    nb_ = 10  # reset the frequencies every nb_ saving
                    self._nb_updated_act_tensorboard += UPDATE_FREQ
                    tf.summary.scalar("zz_freq_inj", self.nb_injection / self._nb_updated_act_tensorboard, step_tb)
                    tf.summary.scalar("zz_freq_voltage", self.nb_voltage / self._nb_updated_act_tensorboard, step_tb)
                    tf.summary.scalar("z_freq_topo", self.nb_topology / self._nb_updated_act_tensorboard, step_tb)
                    tf.summary.scalar("z_freq_line_status", self.nb_line / self._nb_updated_act_tensorboard, step_tb)
                    tf.summary.scalar("z_freq_redisp", self.nb_redispatching / self._nb_updated_act_tensorboard, step_tb)
                    tf.summary.scalar("z_freq_do_nothing", self.nb_do_nothing / self._nb_updated_act_tensorboard, step_tb)
                    if step % (nb_ * UPDATE_FREQ) == 0:
                        self.nb_injection = 0
                        self.nb_voltage = 0
                        self.nb_topology = 0
                        self.nb_line = 0
                        self.nb_redispatching = 0
                        self.nb_do_nothing = 0
                        self._nb_updated_act_tensorboard = 0

                if self._time_step_lived is not None:
                    tf.summary.histogram(
                        "timestep_lived", self._time_step_lived, step=step_tb, buckets=None,
                        description="number of time steps lived for all scenarios"
                    )
                if self._nb_chosen is not None:
                    tf.summary.histogram(
                        "nb_chosen", self._nb_chosen, step=step_tb, buckets=None,
                        description="number of times this scenarios has been played"
                    )
