# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import json
import copy
import numpy as np
import tensorflow as tf

from grid2op.Parameters import Parameters
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from l2rpn_baselines.SliceRDQN.ExperienceBuffer import ExperienceBuffer
from l2rpn_baselines.SliceRDQN.SliceRDQN_NN import SliceRDQN_NN
from l2rpn_baselines.SliceRDQN.slice_util import *

INITIAL_EPSILON = 0.8
FINAL_EPSILON = 0.05
DECAY_EPSILON = 1024*128
STEP_EPSILON = (INITIAL_EPSILON-FINAL_EPSILON)/DECAY_EPSILON
DISCOUNT_FACTOR = 0.55
REPLAY_BUFFER_SIZE = 1024*8
UPDATE_FREQ = 756
UPDATE_TARGET_HARD_FREQ = -1
UPDATE_TARGET_SOFT_TAU = 0.001
INPUT_BIAS = 3.0

class SliceRDQN(AgentWithConverter):
    def __init__(self,
                 observation_space,
                 action_space,
                 name=__name__,
                 trace_length=1,
                 batch_size=1,
                 is_training=False,
                 lr=1e-5):
        # Call parent constructor
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=IdToAct)

        # Store constructor params
        self.observation_space = observation_space
        self.name = name
        self.trace_length = trace_length
        self.batch_size = batch_size
        self.is_training = is_training
        self.lr = lr

        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []
        self.mem_state = None
        self.carry_state = None

        # Declare training vars
        self.exp_buffer = None
        self.done = False
        self.epoch_rewards = None
        self.epoch_alive = None
        self.Qtarget = None
        self.epsilon = INITIAL_EPSILON

        # Compute dimensions from intial state
        self.action_size = self.action_space.n
        self.observation_shape = shape_obs(self.observation_space)

        # Slices dict
        self.slices = {
            "lines": {
                "indexes": [1,3,4,9,10,11,14,15,18,20,23,24],
                "q_len": lines_q_len(self.action_space)
            },
            "sub": {
                "indexes": [1,2,4,9,10,11,12,14,15,18,20,23,24],
                "q_len": topo_q_len(self.action_space)
            },
            "disp": {
                "indexes": [0,4,7,8,9,10,11,12,14,18,23,24],
                "q_len": disp_q_len(self.action_space)
            }
        }
        self.n_slices = len(self.slices.keys())

        # Load network graph
        self.Qmain = SliceRDQN_NN(self.action_size,
                                  self.observation_shape,
                                  self.slices,
                                  learning_rate = self.lr)
        # Setup training vars if needed
        if self.is_training:
            self._init_training()


    def _init_training(self):
        self.exp_buffer = ExperienceBuffer(REPLAY_BUFFER_SIZE, self.batch_size, self.trace_length)
        self.done = True
        self.epoch_rewards = []
        self.epoch_alive = []
        self.Qtarget = SliceRDQN_NN(self.action_size,
                                    self.observation_shape,
                                    self.slices,
                                    learning_rate = self.lr)

    def _reset_state(self, current_obs):
        # Initial state
        self.obs = current_obs
        self.state = self.convert_obs(self.obs)
        self.done = False
        self.mem_state = np.zeros((self.n_slices, self.Qmain.h_size))
        self.carry_state = np.zeros((self.n_slices, self.Qmain.h_size))

    def _register_experience(self, episode_exp, episode):
        missing_obs = self.trace_length - len(episode_exp)

        if missing_obs > 0: # We are missing exp to make a trace
            exp = episode_exp[0] # Use inital state to fill out
            for missing in range(missing_obs):
                # Use do_nothing action at index 0
                self.exp_buffer.add(exp[0], 0, exp[2], exp[3], exp[4], episode)

        # Register the actual experience
        for exp in episode_exp:
            self.exp_buffer.add(exp[0], exp[1], exp[2], exp[3], exp[4], episode)

    def _save_hyperparameters(self, logpath, env, steps):
        r_instance = env.reward_helper.template_reward
        hp = {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "trace_len": self.trace_length,
            "e_start": INITIAL_EPSILON,
            "e_end": FINAL_EPSILON,
            "e_decay": DECAY_EPSILON,
            "discount": DISCOUNT_FACTOR,
            "buffer_size": REPLAY_BUFFER_SIZE,
            "update_freq": UPDATE_FREQ,
            "update_hard": UPDATE_TARGET_HARD_FREQ,
            "update_soft": UPDATE_TARGET_SOFT_TAU,
            "input_bias": INPUT_BIAS,
            "reward": dict(r_instance)
        }
        hp_filename = "{}-hypers.json".format(self.name)
        hp_path = os.path.join(logpath, hp_filename)
        with open(hp_path, 'w') as fp:
            json.dump(hp, fp=fp, indent=2)

    ## Agent Interface
    def convert_obs(self, observation):
        return convert_obs_pad(observation, bias=INPUT_BIAS)

    def convert_act(self, action):
        return super().convert_act(action)

    def reset(self, observation):
        self._reset_state(observation)

    def my_act(self, state, reward, done=False):
        data_input = np.array(state)
        a, _, m, c = self.Qmain.predict_move(data_input,
                                             self.mem_state,
                                             self.carry_state)
        self.mem_state = m
        self.carry_state = c

        return a

    def load(self, path):
        self.Qmain.load_network(path)
        if self.is_training:
            self.Qmain.update_target_hard(self.Qtarget.model)

    def save(self, path):
        self.Qmain.save_network(path)

    ## Training Procedure
    def train(self, env,
              iterations,
              save_path,
              num_pre_training_steps = 0,
              logdir = "logs"):

        # Loop vars
        num_training_steps = iterations
        num_steps = num_pre_training_steps + num_training_steps
        step = 0
        self.epsilon = INITIAL_EPSILON
        alive_steps = 0
        total_reward = 0
        episode = 0
        episode_exp = []

        # Create file system related vars
        logpath = os.path.join(logdir, self.name)
        os.makedirs(save_path, exist_ok=True)
        modelpath = os.path.join(save_path, self.name + ".tf")
        self.tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        self._save_hyperparameters(save_path, env, num_steps)

        # Training loop
        self._reset_state(env.current_obs)
        while step < num_steps:
            # New episode
            if self.done:
                if episode % 1000 == 0:
                    # shuffle the data every now and then
                    # TODO change the "1000" above
                    env.chronics_handler.shuffle(
                        shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
                new_obs = env.reset() # This shouldn't raise
                self.reset(new_obs)

                new_obs = env.reset() # This shouldn't raise
                self._reset_state(new_obs)
                # Push current episode experience to experience buffer
                self._register_experience(episode_exp, episode)
                # Reset current episode experience
                episode += 1
                episode_exp = []

            if step % 1000 == 0:
                print("Step [{}] -- Dropout [{}]".format(step, self.epsilon))

            # Choose an action
            if step <= num_pre_training_steps:
                a, m, c = self.Qmain.random_move(self.state,
                                                 self.mem_state,
                                                 self.carry_state)
            elif len(episode_exp) < self.trace_length:
                a, m, c = self.Qmain.random_move(self.state,
                                                 self.mem_state,
                                                 self.carry_state)
                a = 0
            else:
                a, _, m, c = self.Qmain.bayesian_move(self.state,
                                                      self.mem_state,
                                                      self.carry_state,
                                                      self.epsilon)

            # Update LSTM state
            self.mem_state = m
            self.carry_state = c

            # Convert it to a valid action
            act = self.convert_act(a)
            # Execute action
            new_obs, reward, self.done, info = env.step(act)
            new_state = self.convert_obs(new_obs)

            # Save to current episode experience
            episode_exp.append((self.state, a, reward, self.done, new_state))

            # Train when pre-training is over
            if step >= num_pre_training_steps:
                training_step = step - num_pre_training_steps
                # Slowly decay dropout rate
                if self.epsilon > FINAL_EPSILON:
                    self.epsilon -= STEP_EPSILON
                if self.epsilon < FINAL_EPSILON:
                    self.epsilon = FINAL_EPSILON

                # Perform training at given frequency
                if step % UPDATE_FREQ == 0 and self.exp_buffer.can_sample():
                    # Sample from experience buffer
                    batch = self.exp_buffer.sample()
                    # Perform training
                    self._batch_train(batch, training_step, step)
                    # Update target network towards primary network
                    if UPDATE_TARGET_SOFT_TAU > 0:
                        tau = UPDATE_TARGET_SOFT_TAU
                        self.Qmain.update_target_soft(self.Qtarget.model, tau)

                # Every UPDATE_TARGET_HARD_FREQ trainings
                # update target completely
                if UPDATE_TARGET_HARD_FREQ > 0 and \
                   step % (UPDATE_FREQ * UPDATE_TARGET_HARD_FREQ) == 0:
                    self.Qmain.update_target_hard(self.Qtarget.model)

            total_reward += reward
            if self.done:
                self.epoch_rewards.append(total_reward)
                self.epoch_alive.append(alive_steps)
                print("Survived [{}] steps".format(alive_steps))
                print("Total reward [{}]".format(total_reward))
                alive_steps = 0
                total_reward = 0
            else:
                alive_steps += 1

            # Save the network every 1000 iterations
            if step > 0 and step % 1000 == 0:
                self.save(modelpath)

            # Iterate to next loop
            step += 1
            self.obs = new_obs
            self.state = new_state

        # Save model after all steps
        self.save(modelpath)

    def _batch_train(self, batch, training_step, step):
        """Trains network to fit given parameters"""
        Q = np.zeros((self.batch_size, self.action_size))
        batch_mem = np.zeros((self.batch_size, self.n_slices, self.Qmain.h_size))
        batch_carry = np.zeros((self.batch_size, self.n_slices, self.Qmain.h_size))

        input_shape = (self.batch_size, self.trace_length) + self.observation_shape
        m_data = np.vstack(batch[:, 0])
        m_data = m_data.reshape(input_shape)
        t_data = np.vstack(batch[:, 4])
        t_data = t_data.reshape(input_shape)
        q_input = [copy.deepcopy(batch_mem), copy.deepcopy(batch_carry), copy.deepcopy(m_data)]
        q1_input = [copy.deepcopy(batch_mem), copy.deepcopy(batch_carry), copy.deepcopy(t_data)]
        q2_input = [copy.deepcopy(batch_mem), copy.deepcopy(batch_carry), copy.deepcopy(t_data)]

        # Batch predict
        self.Qmain.trace_length.assign(self.trace_length)
        self.Qmain.dropout_rate.assign(0.0)
        self.Qtarget.trace_length.assign(self.trace_length)
        self.Qtarget.dropout_rate.assign(0.0)

        # Save the graph just the first time
        if training_step == 0:
            tf.summary.trace_on()

        # T batch predict
        Q, _, _ = self.Qmain.model.predict(q_input, batch_size = self.batch_size)

        ## Log graph once and disable graph logging
        if training_step == 0:
            with self.tf_writer.as_default():
                tf.summary.trace_export(self.name + "-graph", step)

        # T+1 batch predict
        Q1, _, _ = self.Qmain.model.predict(q1_input, batch_size = self.batch_size)
        Q2, _, _ = self.Qtarget.model.predict(q2_input, batch_size = self.batch_size)

        # Compute batch Double Q update to Qtarget
        for i in range(self.batch_size):
            idx = i * (self.trace_length - 1)
            doubleQ = Q2[i, np.argmax(Q1[i])]
            a = batch[idx][1]
            r = batch[idx][2]
            d = batch[idx][3]
            Q[i, a] = r
            if d == False:
                Q[i, a] += DISCOUNT_FACTOR * doubleQ

        # Batch train
        batch_x = [batch_mem, batch_carry, m_data]
        batch_y = [Q, batch_mem, batch_carry]
        loss = self.Qmain.model.train_on_batch(batch_x, batch_y)
        loss = loss[0]

        print("loss =", loss)
        with self.tf_writer.as_default():
            mean_reward = np.mean(self.epoch_rewards)
            mean_alive = np.mean(self.epoch_alive)
            if len(self.epoch_rewards) >= 100:
                mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                mean_alive_100 = np.mean(self.epoch_alive[-100:])
            else:
                mean_reward_100 = mean_reward
                mean_alive_100 = mean_alive
            tf.summary.scalar("mean_reward", mean_reward, step)
            tf.summary.scalar("mean_alive", mean_alive, step)
            tf.summary.scalar("mean_reward_100", mean_reward_100, step)
            tf.summary.scalar("mean_alive_100", mean_alive_100, step)
            tf.summary.scalar("loss", loss, step)
