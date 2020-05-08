# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import numpy as np
from abc import abstractmethod, ABC
from tqdm import tqdm

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from l2rpn_baselines.utils.ReplayBuffer import ReplayBuffer
from l2rpn_baselines.utils.TrainingParam import TrainingParam


class DeepQAgent(AgentWithConverter):
    def __init__(self,
                 action_space,
                 lr=1e-5):
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)

        # and now back to the origin implementation
        self.replay_buffer = None

        self.deep_q = None
        self.lr = lr
        self.training_param = None
        self.process_buffer = []

    @abstractmethod
    def init_deep_q(self, transformed_observation):
        pass

    # grid2op.Agent interface
    def convert_obs(self, observation):
        return np.concatenate((observation.rho, observation.line_status, observation.topo_vect))

    def my_act(self, transformed_observation, reward, done=False):
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)
        predict_movement_int, *_ = self.deep_q.predict_movement(transformed_observation.reshape(1, -1), epsilon=0.0)
        return int(predict_movement_int)

    # baseline interface
    def load(self, path):
        # not modified compare to original implementation
        self.deep_q.load_network(path)

    def save(self, path):
        if path is not None:
            self.deep_q.save_network(os.path.join(path, self.name))

    def set_chunk(self, nb):
        self.env.set_chunk_size(int(max(100, nb)))

    def shuffle_env_data(self):
        pass

    def train(self,
              env,
              iterations,
              save_path,
              training_param=TrainingParam()):
        self.training_param = training_param
        self._init_replay_buffer()

        # same as in the original implemenation, except the process buffer is now in this class
        observation_num = 0

        # some parameters have been move to a class named "training_param" for convenience
        epsilon = training_param.INITIAL_EPSILON

        # now the number of alive frames and total reward depends on the "underlying environment". It is vector instead
        # of scalar
        alive_frame, total_reward = self._init_global_train_loop()
        reward, done = self._init_local_train_loop()

        with tqdm(total=iterations) as pbar:
            while observation_num < iterations:
                if observation_num % 1000 == 999:
                    print(("Executing loop %d" % observation_num))
                    # for efficient reading of data: at early stage of training, it is advised to load
                    # data by chunk: the model will do game over pretty easily (no need to load all the dataset)
                    tmp = min(10000 * (iterations // observation_num), 10000)
                    self.set_chunk(int(max(10, tmp)))

                curr_state = self._need_reset(observation_num, done)

                if observation_num == 0:
                    # we initialize the NN with the proper shape
                    self.init_deep_q(curr_state)

                # Slowly decay the learning rate
                if epsilon > training_param.FINAL_EPSILON:
                    epsilon -= (training_param.INITIAL_EPSILON - training_param.FINAL_EPSILON) / training_param.EPSILON_DECAY

                initial_state = self._convert_process_buffer()
                self._reset_process_buffer()

                # then we need to predict the next moves. Agents have been adapted to predict a batch of data
                pm_i, pq_v, act = self._next_moves(initial_state, epsilon)

                reward, done = self._init_local_train_loop()
                for i in range(training_param.NUM_FRAMES):
                    temp_observation_obj, temp_reward, temp_done, _ = env.step(act)

                    # we need to handle vectors for "done"
                    reward[~temp_done] += temp_reward[~temp_done]
                    # and then "de stack" the observations coming from different environments
                    self._update_process_buffer(temp_observation_obj)

                    done, reward, total_reward, alive_frame \
                        = self._update_loop(done, temp_done, alive_frame, total_reward, reward)

                self._store_new_state(initial_state, pm_i, reward, done)

                if self.replay_buffer.size() > training_param.MIN_OBSERVATION:
                    s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(
                        training_param.MINIBATCH_SIZE)
                    isfinite = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
                    self.deep_q.target_train()

                    if not isfinite:
                        # if the loss is not finite i stop the learning
                        print("ERROR INFINITE LOSS")
                        break

                # Save the network every 10000 iterations
                if observation_num % 10000 == 9999 or observation_num == iterations - 1:
                    self.save(save_path)
                    print("Saving Network")

                observation_num += 1
                pbar.update(1)

    # auxiliary functions
    def _need_reset(self, observation_num, done):
        if done:
            obs = self.env.reset()
            tmp_obs = self.convert_obs(obs)
            self.process_buffer.append(tmp_obs)
            curr_state = self._convert_process_buffer()
        else:
            #TODO and now what ????
        return WHATTTTT ????

    def _reset_process_buffer(self):
        self.process_buffer = []

    def _init_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(self.training_param.BUFFER_SIZE)

    def _convert_process_buffer(self):
        """Converts the list of NUM_FRAMES images in the process buffer
        into one training sample"""
        # here i simply concatenate the action in case of multiple action in the "buffer"
        if self.training_param.NUM_FRAMES != 1:
            raise RuntimeError("This h_need_resetas not been tested with self.training_param.NUM_FRAMES != 1 for now")
        return np.array([np.concatenate(el) for el in self.process_buffer])

    def _update_process_buffer(self, temp_observation_obj):
        self.process_buffer.append(self.convert_obs(temp_observation_obj))
        # for worker_id, obs in enumerate(temp_observation_obj):
        #     self.process_buffer[worker_id].append(self.convert_obs(temp_observation_obj[worker_id]))

    def _store_new_state(self, initial_state, predict_movement_int, reward, done):
        # vectorized version of the previous code
        new_state = self._convert_process_buffer()
        self.replay_buffer.add(initial_state, predict_movement_int, reward, done, new_state)

        # # same as before, but looping through the "underlying environment"
        # for sub_env_id in range(self.nb_process):
        #     self.replay_buffer.add(initial_state[sub_env_id],
        #                                  predict_movement_int[sub_env_id],
        #                                  reward[sub_env_id],
        #                                  done[sub_env_id],
        #                                  new_state[sub_env_id])

    def _next_move(self, curr_state, epsilon):
        pm_i, pq_v = self.deep_q.predict_movement(curr_state, epsilon)
        act = self.convert_act(pm_i)
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

    def _update_loop(self, done, temp_done, alive_frame, total_reward, reward):
        done = done | temp_done

        # increase of 1 the number of frame alive for relevant "underlying environments"
        alive_frame[~temp_done] += 1
        # loop through the environment where a game over was done, and print the results
        for env_done_idx in np.where(temp_done)[0]:
            print("For env with id {}".format(env_done_idx))
            print("\tLived with maximum time ", alive_frame[env_done_idx])
            print("\tEarned a total of reward equal to ", total_reward[env_done_idx])

        reward[temp_done] = 0.
        total_reward[temp_done] = 0.
        total_reward += reward
        alive_frame[temp_done] = 0
        return done, reward, total_reward, alive_frame

    def _init_local_train_loop(self):
        # reward, done = np.zeros(self.nb_process), np.full(self.nb_process, fill_value=False, dtype=np.bool)
        reward = 0.
        done = False
        return reward, done