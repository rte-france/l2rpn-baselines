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
    def __init__(self,
                 action_space,
                 nn_archi,
                 name="DeepQAgent",
                 store_action=True,
                 istraining=False,
                 nb_env=1,
                 filter_action_fun=None,
                 verbose=False,
                 **kwargs_converters):
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct, **kwargs_converters)
        self.filter_action_fun = filter_action_fun
        if self.filter_action_fun is not None:
            self.action_space.filter_action(self.filter_action_fun)

        # and now back to the origin implementation
        self.replay_buffer = None
        self.__nb_env = nb_env

        self.deep_q = None
        self.training_param = None
        self.tf_writer = None
        self.name = name
        self.losses = None
        self.graph_saved = False
        self.store_action = store_action
        self.dict_action = {}
        self.istraining = istraining
        self.actions_per_1000steps = np.zeros((1000, self.action_space.size()), dtype=np.int)
        self.illegal_actions_per_1000steps = np.zeros(1000, dtype=np.int)
        self.ambiguous_actions_per_1000steps = np.zeros(1000, dtype=np.int)
        self.epsilon = 1.0

        # for tensorbaord
        self._train_lr = None

        self.reset_num = None

        self.max_iter_env_ = 1000000
        self.curr_iter_env = 0
        self.max_reward = 0.

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
        self.nb_updated_act_tensorboard = None

        # neural network architecture
        self.nn_archi = nn_archi

        # observation tranformers
        self.obs_as_vect = None
        self._tmp_obs = None
        self._indx_obs = None
        self.verbose = verbose

    def init_deep_q(self, training_param):
        if self.deep_q is None:
            self.deep_q = self.nn_archi.make_nn(training_param)

    @staticmethod
    def get_action_size(action_space, filter_fun, kwargs_converters):
        converter = IdToAct(action_space)
        converter.init_converter(**kwargs_converters)
        if filter_fun is not None:
            converter.filter_action(filter_fun)
        return converter.n

    def init_obs_extraction(self, env):
        tmp = np.zeros(0, dtype=np.uint)  # TODO platform independant
        for obs_attr_name in self.nn_archi.get_obs_attr():
            beg_, end_, dtype_ = env.observation_space.get_indx_extract(obs_attr_name)
            tmp = np.concatenate((tmp, np.arange(beg_, end_, dtype=np.uint)))
        self._indx_obs = tmp
        self._tmp_obs = np.zeros((1, tmp.shape[0]), dtype=np.float32)

    # grid2op.Agent interface
    def convert_obs(self, observation):
        obs_as_vect = observation.to_vect()
        # TODO optimize that
        self._tmp_obs[:] = obs_as_vect[self._indx_obs]
        return self._tmp_obs

    def my_act(self, transformed_observation, reward, done=False):
        predict_movement_int, *_ = self.deep_q.predict_movement(transformed_observation, epsilon=0.0)
        res = int(predict_movement_int)
        self._store_action_played(res)
        return res

    # two below function: to train with multiple environments
    def convert_obs_train(self, observations):
        if self.obs_as_vect is None:
            size_obs = self.convert_obs(observations[0]).shape[1]
            self.obs_as_vect = np.zeros((self.__nb_env, size_obs), dtype=np.float32)

        for i, obs in enumerate(observations):
            self.obs_as_vect[i, :] = self.convert_obs(obs).reshape(-1)
        return self.obs_as_vect

    def _store_action_played(self, action_int):
        if self.store_action:
            if action_int not in self.dict_action:
                act = self.action_space.all_actions[action_int]
                is_inj, is_volt, is_topo, is_line_status, is_redisp = act.get_types()
                is_dn = (not is_inj) and (not is_volt) and (not is_topo) and (not is_line_status) and (not is_redisp)
                self.dict_action[action_int] = [0, act, (is_inj, is_volt, is_topo, is_line_status, is_redisp, is_dn)]
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
        # TODO optimize that !
        res = []
        for act_id in act_as_integer:
            res.append(self.convert_act(act_id))
            self._store_action_played(act_id)
        return res

    def load_action_space(self, path):
        # not modified compare to original implementation
        if not os.path.exists(path):
            raise RuntimeError("The model should be stored in \"{}\". But this appears to be empty".format(path))
        try:
            self.action_space.init_converter(
                all_actions=os.path.join(path, "action_space.npy".format(self.name)))
        except Exception as e:
            raise RuntimeError("Impossible to reload converter action space with error \n{}".format(e))

    # baseline interface
    def load(self, path):
        # not modified compare to original implementation
        tmp_me = os.path.join(path, self.name)
        if not os.path.exists(tmp_me):
            raise RuntimeError("The model should be stored in \"{}\". But this appears to be empty".format(tmp_me))
        self.load_action_space(tmp_me)

        # TODO handle case where training param class has been overidden
        self.training_param = TrainingParam.from_json(os.path.join(tmp_me, "training_params.json".format(self.name)))
        self.deep_q = self.nn_archi.make_nn(self.training_param )
        try:
            self.deep_q.load_network(tmp_me, name=self.name)
        except Exception as e:
            raise RuntimeError("Impossible to load the model located at \"{}\" with error \n{}".format(path, e))

        for nm_attr in ["_time_step_lived", "_nb_chosen", "_proba"]:
            conv_path = os.path.join(tmp_me, "{}.npy".format(nm_attr))
            if os.path.exists(conv_path):
                setattr(self, nm_attr, np.load(file=conv_path))
            else:
                raise RuntimeError("Impossible to find the data \"{}.npy\" at \"{}\"".format(nm_attr, tmp_me))

    def save(self, path):
        if path is not None:
            tmp_me = os.path.join(path, self.name)
            if not os.path.exists(tmp_me):
                os.mkdir(tmp_me)
            nm_conv = "action_space.npy"
            conv_path = os.path.join(tmp_me, nm_conv)
            if not os.path.exists(conv_path):
                self.action_space.save(path=tmp_me, name=nm_conv)

            self.training_param.save_as_json(tmp_me, name="training_params.json")
            self.nn_archi.save_as_json(tmp_me, "nn_architecture.json")
            self.deep_q.save_network(tmp_me, name=self.name)

            # TODO save the "oversampling" part, and all the other info
            for nm_attr in ["_time_step_lived", "_nb_chosen", "_proba"]:
                conv_path = os.path.join(tmp_me, "{}.npy".format(nm_attr))
                np.save(arr=getattr(self, nm_attr), file=conv_path)

    # utilities for data reading
    def set_chunk(self, env, nb):
        env.set_chunk_size(int(max(100, nb)))

    def train(self,
              env,
              iterations,
              save_path,
              logdir,
              training_param=None):

        if training_param is None:
            training_param = TrainingParam()

        self._train_lr = training_param.lr

        if self.training_param is None:
            self.training_param = training_param
        else:
            training_param = self.training_param
        self.init_deep_q(self.training_param)

        self._init_replay_buffer()

        # efficient reading of the data (read them by chunk of roughly 1 day
        nb_ts_one_day = 24 * 60 / 5  # number of time steps per day
        self.set_chunk(env, nb_ts_one_day)

        # Create file system related vars
        if save_path is not None:
            save_path = os.path.abspath(save_path)
            os.makedirs(save_path, exist_ok=True)

        if logdir is not None:
            logpath = os.path.join(logdir, self.name)
            self.tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        else:
            logpath = None
            self.tf_writer = None
        UPDATE_FREQ = training_param.update_tensorboard_freq  # update tensorboard every "UPDATE_FREQ" steps
        SAVING_NUM = training_param.save_model_each

        self.init_obs_extraction(env)

        training_step = self.training_param.last_step

        # some parameters have been move to a class named "training_param" for convenience
        self.epsilon = self.training_param.initial_epsilon

        # now the number of alive frames and total reward depends on the "underlying environment". It is vector instead
        # of scalar
        alive_frame, total_reward = self._init_global_train_loop()
        reward, done = self._init_local_train_loop()
        epoch_num = 0
        self.losses = np.zeros(iterations)
        alive_frames = np.zeros(iterations)
        total_rewards = np.zeros(iterations)
        new_state = None
        self.reset_num = 0
        self.curr_iter_env = 0
        self.max_reward = env.reward_range[1]

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
        if _CACHE_AVAILABLE_DEEPQAGENT:
            if isinstance(env.chronics_handler.real_data, MultifolderWithCache):
                th_size = env.chronics_handler.real_data.cache_size
        if th_size is None:
            th_size = len(env.chronics_handler.real_data.subpaths)

        self._prev_obs_num = 0
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
        self.nb_updated_act_tensorboard = 0

        with tqdm(total=iterations - training_step, disable=not self.verbose) as pbar:
            while training_step < iterations:
                # reset or build the environment
                initial_state = self._need_reset(env, training_step, epoch_num, done, new_state)

                # Slowly decay the exploration parameter epsilon
                # if self.epsilon > training_param.FINAL_EPSILON:
                self.epsilon = self.training_param.get_next_epsilon(current_step=training_step)

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
                new_state = self.convert_obs_train(temp_observation_obj)
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
    def _train_model(self, training_step):
        self.training_param.tell_step(training_step)
        if training_step > max(self.training_param.min_observation, self.training_param.minibatch_size) and \
            self.training_param.do_train():
            # train the model
            s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(self.training_param.minibatch_size)
            tf_writer = None
            if self.graph_saved is False:
                tf_writer = self.tf_writer
            loss = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch,
                                     tf_writer)
            # save learning rate for later
            self._train_lr = self.deep_q.optimizer_model._decayed_lr('float32').numpy()
            self.graph_saved = True
            if not np.all(np.isfinite(loss)):
                # if the loss is not finite i stop the learning
                return False
            self.deep_q.target_train()
            self.losses[training_step:] = np.sum(loss)
        return True

    def _updage_illegal_ambiguous(self, curr_step, info):
        self.illegal_actions_per_1000steps[curr_step % 1000] = np.sum([el["is_illegal"] for el in info])
        self.ambiguous_actions_per_1000steps[curr_step % 1000] = np.sum([el["is_ambiguous"] for el in info])

    def _store_action_played_train(self, training_step, action_id):
        which_row = training_step % 1000
        self.actions_per_1000steps[which_row, :] = 0
        self.actions_per_1000steps[which_row, action_id] += 1

    def _fast_forward_env(self, env, time=7*24*60/5):
        my_int = np.random.randint(0, min(time, env.chronics_handler.max_timestep()))
        env.fast_forward_chronics(my_int)

    def _reset_env_clean_state(self, env):
        """

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

        if self.training_param.step_increase_nb_iter > 0:
            self.max_iter_env(min(max(self.training_param.min_iter,
                                      self.training_param.max_iter_fun(self._total_sucesses)),
                                  self.training_param.max_iter))  # TODO
        self.curr_iter_env += 1
        if new_state is None:
            # it's the first ever loop
            obs = env.reset()
            if self.__nb_env == 1:
                # still hack to have same program interface between multi env and not multi env
                obs = [obs]
            new_state = self.convert_obs_train(obs)
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
            self._time_step_lived[self._prev_id] += ts_lived
            self._prev_obs_num = observation_num
            if self.training_param.oversampling_rate is not None:
                # proba = np.sqrt(1. / (self._time_step_lived +1))
                # # over sampling some kind of "UCB like" stuff
                # # https://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/

                # proba = 1. / (self._time_step_lived + 1)
                self._proba[:] = 1. / (self._time_step_lived**self.training_param.oversampling_rate + 1)
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
            self._nb_chosen[self._prev_id] += 1

            # random fast forward between now and next week
            if self.training_param.random_sample_datetime_start is not None:
                self._fast_forward_env(env, time=self.training_param.random_sample_datetime_start)

            self.curr_iter_env = 0
            obs = [env.current_obs]
            new_state = self.convert_obs_train(obs)
            if epoch_num % len(env.chronics_handler.real_data.subpaths) == 0:
                # re shuffle the data
                env.chronics_handler.shuffle(lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
        return new_state

    def _init_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(self.training_param.buffer_size)

    def _store_new_state(self, initial_state, predict_movement_int, reward, done, new_state):
        # vectorized version of the previous code
        for i_s, pm_i, reward, done, new_state in zip(initial_state, predict_movement_int, reward, done, new_state):
            self.replay_buffer.add(i_s,
                                   pm_i,
                                   reward,
                                   done,
                                   new_state)

    def max_iter_env(self, new_max_iter):
        self.max_iter_env_ = new_max_iter

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
            if self.curr_iter_env >= self.max_iter_env_:
                temp_done[0] = True
                temp_reward[0] = self.max_reward
                self._total_sucesses += 1

        done = temp_done
        alive_frame[done] = 0
        total_reward[done] = 0.
        self.reset_num += np.sum(done)
        if self.reset_num >= self.__nb_env:
            # increase the "global epoch num" represented by "epoch_num" only when on average
            # all environments are "done"
            epoch_num += 1
            self.reset_num = 0

        total_reward[~done] += temp_reward[~done]
        alive_frame[~done] += 1
        return done, temp_reward, total_reward, alive_frame, epoch_num

    def _init_local_train_loop(self):
        # reward, done = np.zeros(self.nb_process), np.full(self.nb_process, fill_value=False, dtype=np.bool)
        reward = np.zeros(self.__nb_env, dtype=np.float32)
        done = np.full(self.__nb_env, fill_value=False, dtype=np.bool)
        return reward, done

    def _save_tensorboard(self, step, epoch_num, UPDATE_FREQ, epoch_rewards, epoch_alive):
        if self.tf_writer is None:
            return

        # Log some useful metrics every even updates
        if step % UPDATE_FREQ == 0 and epoch_num > 0:
            if step % (10 * UPDATE_FREQ) == 0:
                # print the top k scenarios the "hardest" (ie chosen the most number of times
                if self.verbose:
                    top_k = 10
                    array_ = np.argsort(self._nb_chosen)[-top_k:][::-1]
                    print("hardest scenarios\n{}".format(array_))
                    print("They have been chosen respectively\n{}".format(self._nb_chosen[array_]))
                    # print("Associated proba are\n{}".format(self._proba[array_]))
                    print("The number of timesteps played is\n{}".format(self._time_step_lived[array_]))
                    print("avg (accross all scenarios) number of timsteps played {}"
                          "".format(np.mean(self._time_step_lived)))
                    print("Time alive: {}".format(self._time_step_lived[array_] / (self._nb_chosen[array_] + 1)))
                    print("Avg time alive: {}".format(np.mean(self._time_step_lived / (self._nb_chosen + 1 ))))
                    # print("avg (accross all scenarios) proba {}"
                    #       "".format(np.mean(self._proba)))
            with self.tf_writer.as_default():
                last_alive = epoch_alive[(epoch_num-1)]
                last_reward = epoch_rewards[(epoch_num-1)]

                mean_reward = np.nanmean(epoch_rewards[:epoch_num])
                mean_alive = np.nanmean(epoch_alive[:epoch_num])

                mean_reward_30 = mean_reward
                mean_alive_30 = mean_alive
                mean_reward_100 = mean_reward
                mean_alive_100 = mean_alive

                tmp = self.actions_per_1000steps > 0
                tmp = tmp.sum(axis=0)
                nb_action_taken_last_1000_step = np.sum(tmp > 0)

                nb_illegal_act = np.sum(self.illegal_actions_per_1000steps)
                nb_ambiguous_act = np.sum(self.ambiguous_actions_per_1000steps)

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
                tf.summary.scalar("loss", self.losses[step], step_tb,
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
                tf.summary.scalar("z_max_iter", self.max_iter_env_, step_tb,
                                  description="maximum number of time steps before deciding a scenario is over (=win)")
                tf.summary.scalar("z_total_episode", epoch_num, step_tb,
                                  description="total number of episode played (~number of \"reset\")")

                if self.store_action:
                    nb_ = 10  # reset the frequencies every nb_ saving
                    self.nb_updated_act_tensorboard += UPDATE_FREQ
                    tf.summary.scalar("zz_freq_inj", self.nb_injection / self.nb_updated_act_tensorboard, step_tb)
                    tf.summary.scalar("zz_freq_voltage", self.nb_voltage / self.nb_updated_act_tensorboard, step_tb)
                    tf.summary.scalar("z_freq_topo", self.nb_topology / self.nb_updated_act_tensorboard, step_tb)
                    tf.summary.scalar("z_freq_line_status", self.nb_line / self.nb_updated_act_tensorboard, step_tb)
                    tf.summary.scalar("z_freq_redisp", self.nb_redispatching / self.nb_updated_act_tensorboard, step_tb)
                    tf.summary.scalar("z_freq_do_nothing", self.nb_do_nothing / self.nb_updated_act_tensorboard, step_tb)
                    if step % (nb_ * UPDATE_FREQ) == 0:
                        self.nb_injection = 0
                        self.nb_voltage = 0
                        self.nb_topology = 0
                        self.nb_line = 0
                        self.nb_redispatching = 0
                        self.nb_do_nothing = 0
                        self.nb_updated_act_tensorboard = 0


                tf.summary.histogram(
                    "timestep_lived", self._time_step_lived, step=step_tb, buckets=None,
                    description="number of time steps lived for all scenarios"
                )
                tf.summary.histogram(
                    "nb_chosen", self._nb_chosen, step=step_tb, buckets=None,
                    description="number of times this scenarios has been played"
                )
