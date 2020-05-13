# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import numpy as np
from abc import abstractmethod
from tqdm import tqdm
import tensorflow as tf

import grid2op
from grid2op.Exceptions import Grid2OpException
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from l2rpn_baselines.utils.ReplayBuffer import ReplayBuffer
from l2rpn_baselines.utils.TrainingParam import TrainingParam
import pdb


class DeepQAgent(AgentWithConverter):
    def __init__(self,
                 action_space,
                 name="DeepQAgent",
                 lr=1e-3,
                 learning_rate_decay_steps=3000,
                 learning_rate_decay_rate=0.99,
                 store_action=False,
                 istraining=False,
                 nb_env=1):
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        # and now back to the origin implementation
        self.replay_buffer = None
        self.__nb_env = nb_env

        self.deep_q = None
        self.training_param = None
        self.tf_writer = None
        self.name = name
        self.losses = None
        self.graph_saved = False
        self.lr = lr
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.store_action = store_action
        self.dict_action = {}
        self.istraining = istraining
        self.actions_per_1000steps = np.zeros((1000, self.action_space.size()), dtype=np.int)
        self.illegal_actions_per_1000steps = np.zeros(1000, dtype=np.int)
        self.ambiguous_actions_per_1000steps = np.zeros(1000, dtype=np.int)
        self.train_lr = lr
        self.epsilon = 1.0

        self.obs_as_vect = None
        self._tmp_obs = None
        self.reset_num = None

    @abstractmethod
    def init_deep_q(self, transformed_observation):
        pass

    # grid2op.Agent interface
    def convert_obs(self, observation):
        if self._tmp_obs is None:
            tmp = np.concatenate((observation.prod_p,
                               observation.load_p,
                               observation.rho,
                               observation.timestep_overflow,
                               observation.line_status,
                               observation.topo_vect,
                               observation.time_before_cooldown_line,
                               observation.time_before_cooldown_sub,
                               )).reshape(1, -1)

            self._tmp_obs = np.zeros((1, tmp.shape[1]), dtype=np.float32)

        # TODO optimize that
        self._tmp_obs[:] = np.concatenate((observation.prod_p,
                               observation.load_p,
                               observation.rho,
                               observation.timestep_overflow,
                               observation.line_status,
                               observation.topo_vect,
                               observation.time_before_cooldown_line,
                               observation.time_before_cooldown_sub,
                               )).reshape(1, -1)
        return self._tmp_obs

    def my_act(self, transformed_observation, reward, done=False):
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)
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
                self.dict_action[action_int] = [0, act]
            self.dict_action[action_int][0] += 1

    def _convert_all_act(self, act_as_integer):
        # TODO optimize that !
        res = []
        for act_id in act_as_integer:
            res.append(self.convert_act(act_id))
        return res

    # baseline interface
    def load(self, path):
        # not modified compare to original implementation
        if not os.path.exists(path):
            raise RuntimeError("The model should be stored in \"{}\". But this appears to be empty".format(path))
        try:
            self.deep_q.load_network(path, name=self.name)
        except Exception as e:
            raise RuntimeError("Impossible to load the model located at \"{}\" with error {}".format(path, e))

    def save(self, path):
        if path is not None:
            self.deep_q.save_network(path, name=self.name)

    # utilities for data reading
    def set_chunk(self, env, nb):
        env.set_chunk_size(int(max(100, nb)))

    def train(self,
              env,
              iterations,
              save_path,
              logdir,
              training_param=TrainingParam()):

        self.training_param = training_param
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
        UPDATE_FREQ = 100  # update tensorboard every "UPDATE_FREQ" steps
        SAVING_NUM = 1000

        training_step = 0

        # some parameters have been move to a class named "training_param" for convenience
        self.epsilon = training_param.INITIAL_EPSILON

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
        with tqdm(total=iterations) as pbar:
            while training_step < iterations:
                # reset or build the environment
                initial_state = self._need_reset(env, training_step, epoch_num, done, new_state)

                # Slowly decay the exploration parameter epsilon
                # if self.epsilon > training_param.FINAL_EPSILON:
                self.epsilon = training_param.get_next_epsilon(current_step=training_step)

                if training_step == 0:
                    # we initialize the NN with the proper shape
                    self.init_deep_q(initial_state)

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
                if not self._train_model(training_param, training_step):
                    # infinite loss in this case
                    print("ERROR INFINITE LOSS")
                    break

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

    # auxiliary functions
    def _train_model(self, training_param, training_step):
        if training_step > max(training_param.MIN_OBSERVATION, training_param.MINIBATCH_SIZE):
            # train the model
            s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(training_param.MINIBATCH_SIZE)
            tf_writer = None
            if self.graph_saved is False:
                tf_writer = self.tf_writer
            loss = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch,
                                     tf_writer)
            # save learning rate for later
            self.train_lr = self.deep_q.optimizer_model._decayed_lr('float32').numpy()
            self.graph_saved = True
            if not np.all(np.isfinite(loss)):
                # if the loss is not finite i stop the learning
                return False
            self.deep_q.target_train()
            self.losses[training_step] = np.sum(loss)
        return True

    def _updage_illegal_ambiguous(self, curr_step, info):
        self.illegal_actions_per_1000steps[curr_step % 1000] = np.sum([el["is_illegal"] for el in info])
        self.ambiguous_actions_per_1000steps[curr_step % 1000] = np.sum([el["is_ambiguous"] for el in info])

    def _store_action_played_train(self, training_step, action_id):
        which_row = training_step % 1000
        self.actions_per_1000steps[which_row, :] = 0
        self.actions_per_1000steps[which_row, action_id] += 1

    def _fast_forward_env(self, env, time=7*24*60/5):
        env.fast_forward_chronics(np.random.randint(0, min(time, env.chronics_handler.max_timestep())))

    def _reset_env_clean_state(self, env):
        """

        """
        # /!\ DO NOT ATTEMPT TO MODIFY OTHERWISE IT WILL PROBABLY CRASH /!\
        # /!\ THIS WILL BE PART OF THE ENVIRONMENT IN FUTURE GRID2OP RELEASE (>= 0.9.0) /!\
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
            # the 3-4 lines below allow to reuse the loaded dataset and continue further up in the
            try:
                self._reset_env_clean_state(env)
                # random fast forward between now and next day
                self._fast_forward_env(env, time=nb_ts_one_day)
            except (StopIteration, Grid2OpException):
                env.reset()
                # random fast forward between now and next week
                self._fast_forward_env(env, time=7*nb_ts_one_day)

            obs = [env.current_obs]
            new_state = self.convert_obs_train(obs)
            if epoch_num % len(env.chronics_handler.real_data.subpaths) == 0:
                # re shuffle the data
                env.chronics_handler.shuffle(lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
        return new_state

    def _init_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(self.training_param.BUFFER_SIZE)

    def _store_new_state(self, initial_state, predict_movement_int, reward, done, new_state):
        # vectorized version of the previous code
        for i_s, pm_i, reward, done, new_state in zip(initial_state, predict_movement_int, reward, done, new_state):
            self.replay_buffer.add(i_s,
                                   pm_i,
                                   reward,
                                   done,
                                   new_state)

    def _next_move(self, curr_state, epsilon):
        pm_i, pq_v = self.deep_q.predict_movement(curr_state, epsilon)
        act = self._convert_all_act(pm_i)
        return pm_i, pq_v, act

    def _init_global_train_loop(self):
        alive_frame = np.zeros(self.__nb_env, dtype=np.int)
        total_reward = np.zeros(self.__nb_env, dtype=np.float32)
        return alive_frame, total_reward

    def _update_loop(self, done, temp_reward, temp_done, alive_frame, total_reward, reward, epoch_num):
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
                tf.summary.scalar("Mean_alive_30", mean_alive_30, step_tb)
                tf.summary.scalar("Mean_reward_30", mean_reward_30, step_tb)
                # then it's alpha numerical order, hence the "z_" in front of some information
                tf.summary.scalar("loss", self.losses[step], step_tb)

                tf.summary.scalar("last_alive", last_alive, step_tb)
                tf.summary.scalar("last_reward", last_reward, step_tb)

                tf.summary.scalar("mean_reward", mean_reward, step_tb)
                tf.summary.scalar("mean_alive", mean_alive, step_tb)

                tf.summary.scalar("mean_reward_100", mean_reward_100, step_tb)
                tf.summary.scalar("mean_alive_100", mean_alive_100, step_tb)

                tf.summary.scalar("nb_differentaction_taken_1000", nb_action_taken_last_1000_step, step_tb)
                tf.summary.scalar("nb_illegal_act", nb_illegal_act, step_tb)
                tf.summary.scalar("nb_ambiguous_act", nb_ambiguous_act, step_tb)

                tf.summary.scalar("z_lr", self.train_lr, step_tb)
                tf.summary.scalar("z_epsilon", self.epsilon, step_tb)