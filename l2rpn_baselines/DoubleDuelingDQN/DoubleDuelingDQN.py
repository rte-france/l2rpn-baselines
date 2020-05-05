# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import json
import math
import numpy as np
import tensorflow as tf
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
from l2rpn_baselines.DoubleDuelingDQN.prioritized_replay_buffer import PrioritizedReplayBuffer

LR_DECAY_STEPS = 1024*32
LR_DECAY_RATE = 0.95
INITIAL_EPSILON = 0.99
FINAL_EPSILON = 0.001
DECAY_EPSILON = 1024*32
DISCOUNT_FACTOR = 0.99
PER_CAPACITY = 1024*64
PER_ALPHA = 0.7
PER_BETA = 0.5
UPDATE_FREQ = 64
UPDATE_TARGET_HARD_FREQ = 16
UPDATE_TARGET_SOFT_TAU = -1


class DoubleDuelingDQN(AgentWithConverter):
    def __init__(self,
                 observation_space,
                 action_space,
                 name=__name__,
                 num_frames=4,
                 is_training=False,
                 batch_size=32,
                 lr=1e-5):
        # Call parent constructor
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=IdToAct)
        self.obs_space = observation_space
        
        # Store constructor params
        self.name = name
        self.num_frames = num_frames
        self.is_training = is_training
        self.batch_size = batch_size
        self.lr = lr
        
        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []
        self.frames = []

        # Declare training vars
        self.per_buffer = None
        self.done = False
        self.frames2 = None
        self.epoch_rewards = None
        self.epoch_alive = None
        self.Qtarget = None
        self.epsilon = 0.0

        # Compute dimensions from intial spaces
        self.observation_size = self.obs_space.size_obs()
        self.action_size = self.action_space.size()

        # Load network graph
        self.Qmain = DoubleDuelingDQN_NN(self.action_size,
                                         self.observation_size,
                                         num_frames=self.num_frames,
                                         learning_rate=self.lr,
                                         learning_rate_decay_steps=LR_DECAY_STEPS,
                                         learning_rate_decay_rate=LR_DECAY_RATE)
        # Setup training vars if needed
        if self.is_training:
            self._init_training()

    def _init_training(self):
        self.epsilon = INITIAL_EPSILON
        self.frames2 = []
        self.epoch_rewards = []
        self.epoch_alive = []
        self.per_buffer = PrioritizedReplayBuffer(PER_CAPACITY, PER_ALPHA)
        self.Qtarget = DoubleDuelingDQN_NN(self.action_size,
                                           self.observation_size,
                                           num_frames = self.num_frames)

    def _reset_state(self, current_obs):
        # Initial state
        self.obs = current_obs
        self.state = self.convert_obs(self.obs)
        self.done = False

    def _reset_frame_buffer(self):
        # Reset frame buffers
        self.frames = []
        if self.is_training:
            self.frames2 = []

    def _save_current_frame(self, state):
        self.frames.append(state.copy())
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)

    def _save_next_frame(self, next_state):
        self.frames2.append(next_state.copy())
        if len(self.frames2) > self.num_frames:
            self.frames2.pop(0)

    def _adaptive_epsilon_decay(self, step):
        ada_div = DECAY_EPSILON / 10.0
        step_off = step + ada_div
        ada_eps = INITIAL_EPSILON * -math.log10((step_off + 1) / (DECAY_EPSILON + ada_div))
        ada_eps_up_clip = min(INITIAL_EPSILON, ada_eps)
        ada_eps_low_clip = max(FINAL_EPSILON, ada_eps_up_clip)
        return ada_eps_low_clip
            
    def _save_hyperparameters(self, logpath, env, steps):
        r_instance = env.reward_helper.template_reward
        hp = {
            "lr": self.lr,
            "lr_decay_steps": LR_DECAY_STEPS,
            "lr_decay_rate": LR_DECAY_RATE,
            "batch_size": self.batch_size,
            "stack_frames": self.num_frames,
            "iter": steps,
            "e_start": INITIAL_EPSILON,
            "e_end": FINAL_EPSILON,
            "e_decay": DECAY_EPSILON,
            "discount": DISCOUNT_FACTOR,
            "per_alpha": PER_ALPHA,
            "per_beta": PER_BETA,
            "per_capacity": PER_CAPACITY,
            "update_freq": UPDATE_FREQ,
            "update_hard": UPDATE_TARGET_HARD_FREQ,
            "update_soft": UPDATE_TARGET_SOFT_TAU,
            "reward": dict(r_instance)
        }
        hp_filename = "{}-hypers.json".format(self.name)
        hp_path = os.path.join(logpath, hp_filename)
        with open(hp_path, 'w') as fp:
            json.dump(hp, fp=fp, indent=2)

    ## Agent Interface
    def convert_obs(self, observation):
        li_vect=  []
        for el in observation.attr_list_vect:
            v = observation._get_array_from_attr_name(el).astype(np.float32)
            v_fix = np.nan_to_num(v)
            v_norm = np.linalg.norm(v_fix)
            if v_norm > 1e6:
                v_res = (v_fix / v_norm) * 10.0
            else:
                v_res = v_fix
            li_vect.append(v_res)
        return np.concatenate(li_vect)

    def convert_act(self, action):
        return super().convert_act(action)

    ## Baseline Interface
    def reset(self, observation):
        self._reset_state(observation)
        self._reset_frame_buffer()

    def my_act(self, state, reward, done=False):
        # Register current state to stacking buffer
        self._save_current_frame(state)
        # We need at least num frames to predict
        if len(self.frames) < self.num_frames:
            return 0 # Do nothing
        # Infer with the last num_frames states
        a, _ = self.Qmain.predict_move(np.array(self.frames))
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
              logdir = "logs-train"):
        # Make sure we can fill the experience buffer
        if num_pre_training_steps < self.batch_size * self.num_frames:
            num_pre_training_steps = self.batch_size * self.num_frames

        # Loop vars
        num_training_steps = iterations
        num_steps = num_pre_training_steps + num_training_steps
        step = 0
        self.epsilon = INITIAL_EPSILON
        alive_steps = 0
        total_reward = 0
        self.done = True

        # Create file system related vars
        logpath = os.path.join(logdir, self.name)
        os.makedirs(save_path, exist_ok=True)
        modelpath = os.path.join(save_path, self.name + ".h5")
        self.tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        self._save_hyperparameters(save_path, env, num_steps)
        
        # Training loop
        while step < num_steps:
            # Init first time or new episode
            if self.done:
                new_obs = env.reset() # This shouldn't raise
                # Random fast forward somewhere in the day
                #ff_rand = np.random.randint(0, 12*24) 
                #env.fast_forward_chronics(ff_rand)
                # Reset internal state
                #new_obs = env.current_obs
                self.reset(new_obs)
            if step % 1000 == 0:
                print("Step [{}] -- Random [{}]".format(step, self.epsilon))

            # Save current observation to stacking buffer
            self._save_current_frame(self.state)

            # Choose an action
            if step <= num_pre_training_steps:
                a = self.Qmain.random_move()
            elif np.random.rand(1) < self.epsilon:
                a = self.Qmain.random_move()
            elif len(self.frames) < self.num_frames:
                a = 0 # Do nothing
            else:
                a, _ = self.Qmain.predict_move(np.array(self.frames))

            # Convert it to a valid action
            act = self.convert_act(a)
            # Execute action
            new_obs, reward, self.done, info = env.step(act)
            new_state = self.convert_obs(new_obs)
            if info["is_illegal"] or info["is_ambiguous"] or \
               info["is_dispatching_illegal"] or info["is_illegal_reco"]:
                print (a, info)

            # Save new observation to stacking buffer
            self._save_next_frame(new_state)

            # Save to experience buffer
            if len(self.frames2) == self.num_frames:
                self.per_buffer.add(np.array(self.frames),
                                    a, reward,
                                    np.array(self.frames2),
                                    self.done)

            # Perform training when we have enough experience in buffer
            if step >= num_pre_training_steps:
                training_step = step - num_pre_training_steps
                # Decay chance of random action
                self.epsilon = self._adaptive_epsilon_decay(training_step)

                # Perform training at given frequency
                if step % UPDATE_FREQ == 0 and len(self.per_buffer) >= self.batch_size:
                    # Perform training
                    self._batch_train(training_step, step)

                    if UPDATE_TARGET_SOFT_TAU > 0.0:
                        # Update target network towards primary network
                        self.Qmain.update_target_soft(self.Qtarget.model, tau=UPDATE_TARGET_SOFT_TAU)

                # Every UPDATE_TARGET_HARD_FREQ trainings, update target completely
                if UPDATE_TARGET_HARD_FREQ > 0 and step % (UPDATE_FREQ * UPDATE_TARGET_HARD_FREQ) == 0:
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
            # Make new obs the current obs
            self.obs = new_obs
            self.state = new_state

        # Save model after all steps
        self.save(modelpath)

    def _batch_train(self, training_step, step):
        """Trains network to fit given parameters"""

        # Sample from experience buffer
        sample_batch = self.per_buffer.sample(self.batch_size, PER_BETA)
        s_batch = sample_batch[0]
        a_batch = sample_batch[1]
        r_batch = sample_batch[2]
        s2_batch = sample_batch[3]
        d_batch = sample_batch[4]
        w_batch = sample_batch[5]
        idx_batch = sample_batch[6]

        Q = np.zeros((self.batch_size, self.action_size))

        # Reshape frames to 1D
        input_size = self.observation_size * self.num_frames
        input_t = np.reshape(s_batch, (self.batch_size, input_size))
        input_t_1 = np.reshape(s2_batch, (self.batch_size, input_size))

        # Save the graph just the first time
        if training_step == 0:
            tf.summary.trace_on()

        # T Batch predict
        Q = self.Qmain.model.predict(input_t, batch_size = self.batch_size)

        ## Log graph once and disable graph logging
        if training_step == 0:
            with self.tf_writer.as_default():
                tf.summary.trace_export(self.name + "-graph", step)

        # T+1 batch predict
        Q1 = self.Qmain.model.predict(input_t_1, batch_size = self.batch_size)
        Q2 = self.Qtarget.model.predict(input_t_1, batch_size = self.batch_size)

        # Compute batch Qtarget using Double DQN
        for i in range(self.batch_size):
            doubleQ = Q2[i, np.argmax(Q1[i])]
            Q[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                Q[i, a_batch[i]] += DISCOUNT_FACTOR * doubleQ

        # Batch train
        loss = self.Qmain.train_on_batch(input_t, Q, w_batch)

        # Update PER buffer
        priorities = self.Qmain.batch_sq_error
        # Can't be zero, no upper limit
        priorities = np.clip(priorities, a_min=1e-8, a_max=None)
        self.per_buffer.update_priorities(idx_batch, priorities)

        # Log some useful metrics every even updates
        if step % (UPDATE_FREQ * 2) == 0:
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
                tf.summary.scalar("lr", self.Qmain.train_lr, step)

            print("loss =", loss)
