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
                 istraining=False):
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        # and now back to the origin implementation
        self.replay_buffer = None

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

    @abstractmethod
    def init_deep_q(self, transformed_observation):
        pass

    # grid2op.Agent interface
    def convert_obs(self, observation):
        return np.concatenate((observation.prod_p,
                               observation.load_p,
                               observation.rho,
                               observation.timestep_overflow,
                               observation.line_status,
                               observation.topo_vect,
                               observation.time_before_cooldown_line,
                               observation.time_before_cooldown_sub,
                               )).reshape(1, -1)

    def _store_action_played(self, action_int):
        if self.store_action:
            if action_int not in self.dict_action:
                act = self.action_space.all_actions[action_int]
                self.dict_action[action_int] = [0, act]
            self.dict_action[action_int][0] += 1

    def my_act(self, transformed_observation, reward, done=False):
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)
        predict_movement_int, *_ = self.deep_q.predict_movement(transformed_observation.reshape(1, -1), epsilon=0.0)
        res = int(predict_movement_int)
        self._store_action_played(res)
        return res

    # baseline interface
    def load(self, path):
        # not modified compare to original implementation
        if not os.path.exists(path):
            raise RuntimeError("The model should be stored in \"{}\". But this appears to be empty".format(path))
        try:
            self.deep_q.load_network(path, name=self.name)
        except Exception as e:
            raise RuntimeError("Impossible to load the model located at \"{}\"".format(path))

    def save(self, path):
        if path is not None:
            self.deep_q.save_network(path, name=self.name)

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
                temp_observation_obj, temp_reward, temp_done, info = env.step(act)
                new_state = self.convert_obs(temp_observation_obj)

                self._updage_illegal_ambiguous(training_step, info)
                done, reward, total_reward, alive_frame, epoch_num \
                    = self._update_loop(done, temp_reward, temp_done, alive_frame, total_reward, reward, epoch_num)

                self._store_new_state(initial_state, pm_i, reward, done, new_state)

                # now train the model
                if self.replay_buffer.size() > max(training_param.MIN_OBSERVATION, training_param.MINIBATCH_SIZE):
                    # train the model
                    s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(
                        training_param.MINIBATCH_SIZE)
                    tf_writer = None
                    if self.graph_saved is False:
                        tf_writer = self.tf_writer
                    loss = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch,
                                             tf_writer)
                    self.train_lr = self.deep_q.optimizer_model._decayed_lr('float32').numpy()
                    self.graph_saved = True
                    if not np.all(np.isfinite(loss)):
                        # if the loss is not finite i stop the learning
                        print("ERROR INFINITE LOSS")
                        break
                    self.deep_q.target_train()
                    self.losses[training_step] = np.sum(loss)

                # Save the network every 1000 iterations
                if training_step % SAVING_NUM == 0 or training_step == iterations - 1:
                    self.save(save_path)

                # save some information to tensorboard
                alive_frames[epoch_num] = alive_frame
                total_rewards[epoch_num] = total_reward
                self._store_action_played_train(training_step, pm_i)

                self._save_tensorboard(training_step, epoch_num, UPDATE_FREQ, total_rewards, alive_frames)
                training_step += 1
                pbar.update(1)

    # auxiliary functions
    def _updage_illegal_ambiguous(self, curr_step, info):
        self.illegal_actions_per_1000steps[curr_step % 1000] = info["is_illegal"]
        self.ambiguous_actions_per_1000steps[curr_step % 1000] = info["is_ambiguous"]

    def _store_action_played_train(self, training_step, action_id):
        # which_row = int(int(training_step) // 1000)
        which_row = training_step % 1000
        # if self.actions_per_1000steps.shape[0] <= which_row:
        #     self.actions_per_1000steps = np.concatenate((self.actions_per_1000steps,
        #                                                   np.zeros((1, self.action_space.size()), dtype=np.int))
        #                                                  )
        # self.actions_per_1000steps[which_row, action_id] += 1
        self.actions_per_1000steps[which_row, :] = 0
        self.actions_per_1000steps[which_row, action_id] += 1

    def _convert_all_act(self, act_as_integer):
        res = []
        for act_id in act_as_integer:
            res.append(self.convert_act(act_id))
        return res

    def _fast_forward_env(self, env, time=7*24*60/5):
        env.fast_forward_chronics(np.random.randint(0, min(time, env.chronics_handler.max_timestep())))

    def _reset_env_clean_state(self, env):
        """

        """
        # /!\ DO NOT ATTEMPT TO MODIFY OTHERWISE IT WILL PROBABLY CRASH /!\
        # /!\ THIS WILL BE PART OF THE ENVIRONMENT IN FUTURE GRID2OP RELEASE /!\
        # AND OF COURSE USING THIS METHOD DURING THE EVALUATION IS COMPLETELY FORBIDDEN
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
        if done or new_state is None:
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

            obs = env.current_obs
            new_state = self.convert_obs(obs)
            if epoch_num % len(env.chronics_handler.real_data.subpaths) == 0:
                # re shuffle the data
                env.chronics_handler.shuffle(lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])

        return new_state

    def _init_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(self.training_param.BUFFER_SIZE)

    def _store_new_state(self, initial_state, predict_movement_int, reward, done, new_state):
        # vectorized version of the previous code
        self.replay_buffer.add(initial_state.reshape(-1),
                               predict_movement_int.reshape(-1),
                               reward,
                               done,
                               new_state.reshape(-1))

        # # same as before, but looping through the "underlying environment"
        # for sub_env_id in range(self.nb_process):
        #     self.replay_buffer.add(initial_state[sub_env_id],
        #                                  predict_movement_int[sub_env_id],
        #                                  reward[sub_env_id],
        #                                  done[sub_env_id],
        #                                  new_state[sub_env_id])

    def _next_move(self, curr_state, epsilon):
        curr_state_ts = tf.convert_to_tensor(curr_state, dtype=tf.float32)
        pm_i, pq_v = self.deep_q.predict_movement(curr_state_ts, epsilon)
        act = self._convert_all_act(pm_i)
        if len(act) == 1:
            act = act[0]

        # act = self.convert_act(pm_i)
        # # and build the convenient vectors (it was scalars before)
        # predict_movement_int = []
        # predict_q_value = []
        # acts = []
        # for p_id in range(self.nb_process):
        #     predict_movement_int.append(pm_i[p_id])
        #     predict_q_value.append(pq_v[p_id])
        #     # and then we convert it to a valid action
        #     acts.append(self.convert_act(pm_i[p_id]))
        return pm_i, pq_v, act

    def _init_global_train_loop(self):
        # alive_frame = np.zeros(self.nb_process, dtype=np.int)
        # total_reward = np.zeros(self.nb_process, dtype=np.float)
        alive_frame = 0
        total_reward = 0.0
        return alive_frame, total_reward

    def _update_loop(self, done, temp_reward, temp_done, alive_frame, total_reward, reward, epoch_num):
        done = temp_done
        if done:
            alive_frame = 0
            total_reward = 0.
            epoch_num += 1
        else:
            total_reward += temp_reward
            alive_frame += 1
        return done, temp_reward, total_reward, alive_frame, epoch_num
        # we need to handle vectors for "done"
        # reward[~temp_done] += temp_reward[~temp_done]
        #
        # done = done | temp_done
        #
        # # increase of 1 the number of frame alive for relevant "underlying environments"
        # alive_frame[~temp_done] += 1
        # # loop through the environment where a game over was done, and print the results
        # for env_done_idx in np.where(temp_done)[0]:
        #     print("For env with id {}".format(env_done_idx))
        #     print("\tLived with maximum time ", alive_frame[env_done_idx])
        #     print("\tEarned a total of reward equal to ", total_reward[env_done_idx])
        #
        # reward[temp_done] = 0.
        # total_reward[temp_done] = 0.
        # total_reward += reward
        # alive_frame[temp_done] = 0
        # return done, reward, total_reward, alive_frame

    def _init_local_train_loop(self):
        # reward, done = np.zeros(self.nb_process), np.full(self.nb_process, fill_value=False, dtype=np.bool)
        reward = 0.
        done = False
        return reward, done

    def _save_tensorboard(self, step, epoch_num, UPDATE_FREQ, epoch_rewards, epoch_alive):
        if self.tf_writer is None:
            return

        # Log some useful metrics every even updates
        if step % UPDATE_FREQ == 0:
            with self.tf_writer.as_default():
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

                # show first the Mean reward and mine time alive (hence the upper case)
                tf.summary.scalar("Mean_alive_30", mean_alive_30, step)
                tf.summary.scalar("Mean_reward_30", mean_reward_30, step)
                # then it's alpha numerical order, hence the "z_" in front of some information
                tf.summary.scalar("loss", self.losses[step], step)
                tf.summary.scalar("mean_reward", mean_reward, step)
                tf.summary.scalar("mean_alive", mean_alive, step)
                tf.summary.scalar("mean_reward_100", mean_reward_100, step)
                tf.summary.scalar("mean_alive_100", mean_alive_100, step)
                tf.summary.scalar("nb_differentaction_taken_1000", nb_action_taken_last_1000_step, step)
                tf.summary.scalar("nb_illegal_act", nb_illegal_act, step)
                tf.summary.scalar("nb_ambiguous_act", nb_ambiguous_act, step)
                tf.summary.scalar("z_lr", self.train_lr, step)
                tf.summary.scalar("z_epsilon", self.epsilon, step)