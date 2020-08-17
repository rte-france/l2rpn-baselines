# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os

from grid2op.Agent import BaseAgent
from l2rpn_baselines.SAC.SAC_NN import SAC_NN
from l2rpn_baselines.SAC.SAC_Obs import *
from l2rpn_baselines.utils.ReplayBuffer import ReplayBuffer

class SAC_Agent(BaseAgent):
    def __init__(self,
                 observation_space,
                 action_space,
                 nn_config,
                 name="SAC",
                 training=False,
                 verbose=False):
        super().__init__(action_space)

        self.name = name
        self.observation_space = observation_space
        self.observation_shape = self.nn_observation_shape(observation_space)
        self.action_shape = self.nn_action_shape(action_space)
        self.nn = SAC_NN(self.observation_shape,
                         self.action_shape,
                         nn_config,
                         training=training)
        self.verbose = verbose

        # Declare None vars for training mode
        self.replay_buffer = None
        self.done = False
        self.current_state = None
        self.next_state = None
        self.current_obs = None
        self.next_obs = None

    #####
    ## Grid2op <-> NN converters
    #####
    def nn_observation_shape(self, observation_space):
        obs_size = sac_size_obs(observation_space)
        return (obs_size,)

    def nn_action_shape(self, action_space):
        return (action_space.dim_topo,)

    def observation_grid2op_to_nn(self, observation_grid2op):
        return sac_convert_obs(observation_grid2op)

    def danger(self, observation_grid2op):
        return np.any(observation_grid2op.rho > 0.95)

    def set_target(self, nn_target):
        self.has_target = True
        self.target_topo = np.zeros_like(nn_target, dtype=int)
        self.target_topo[nn_target > 0.0] = 1
        self.target_topo += 1
        print ("New target = ", self.target_topo)
    
    def consume_target(self, observation_grid2op):
        current = observation_grid2op.topo_vect
        target = self.target_topo
        
        act_v = np.array(observation_grid2op.topo_vect)

        for sub_id in range(observation_grid2op.n_sub):
            sub_start = np.sum(observation_grid2op.sub_info[:sub_id])
            sub_end = observation_grid2op.sub_info[sub_id]

            if np.any(current[sub_start:sub_end] != target[sub_start:sub_end]):
                act_v[sub_start:sub_end] = target[sub_start:sub_end]
                break

        if np.all(act_v == current):
            self.has_target = False
            self.target_topo = None


        # Convert to NN format
        action_nn = np.array(act_v)
        action_nn -= 2
        action_nn[action_nn == 0] = 1
        
        # Do Nothing for unchanged elements
        act_v[act_v == current] = 0

        action_grid2op = self.action_space({"set_bus": act_v})
        return action_grid2op, action_nn

    ####
    ## grid2op.BaseAgent interface
    ####
    def reset(self, observation_grid2op):
        self.has_target = False
        self.target_topo = None

    def act(self, observation_grid2op, reward, done=False):
        obs_nn = observation_grid2op_to_nn(observation_grid2op)
        action_grid2op, _ = self._act(observation_grid2op, obs_nn)
        return action_grid2op

    def _act(self, observation_grid2op, observation_nn):
        act_grid2op = None
        act_nn = None
        if self.has_target:
            act_grid2op, act_nn = self.consume_target(observation_grid2op)
        elif self.danger(observation_grid2op):
            obs_nn = tf.reshape(observation_nn,
                                shape=(1, observation_nn.shape[0]))
            nn_target = self.nn.predict(obs_nn)[0]
            self.set_target(nn_target)
            act_grid2op, act_nn = self.consume_target(observation_grid2op)
        else:
            act_grid2op = self.action_space({})

        return act_grid2op, act_nn
            

    ###
    ## Baseline train
    ###
    def checkpoint(self, save_path, update_step):
        ckpt_name = "{}-{:04d}".format(self.name, update_step)
        self.nn.save_network(save_path, name=ckpt_name)

    def train(self, env, iterations, save_path, logs_path, train_cfg):
        # Init training vars
        replay_buffer = ReplayBuffer(train_cfg.replay_buffer_size)
        target_step = 0
        update_step = 0
        step = 0

        # Init gym vars
        done = True
        obs = None
        obs_next = None
        obs_nn = None
        obs_nn_next = None
        reward = 0.0
        info = {}

        # Init logger
        logpath = os.path.join(logs_path, self.name)
        logger = tf.summary.create_file_writer(logpath, name=self.name)

        # Do iterations updates
        while update_step < iterations:
            if done:
                obs = env.reset()
                self.reset(obs)
                obs_nn = self.observation_grid2op_to_nn(obs)
                done = False

            act_grid2op, act_nn = self._act(obs, obs_nn)
            obs_next, reward, done, info = env.step(act_grid2op)
            obs_nn_next = self.observation_grid2op_to_nn(obs_next)

            if act_nn is not None:                
                replay_buffer.add(obs_nn, act_nn, reward, done, obs_nn_next)
                target_step += 1

                if replay_buffer.size() >= train_cfg.batch_size:
                    batch = replay_buffer.sample(train_cfg.batch_size)
                    losses = self.nn.train(*batch)
                    update_step += 1

                    with logger.as_default():
                        tf.summary.scalar("1-loss_q1", losses[0], update_step)
                        tf.summary.scalar("2-loss_q2", losses[1], update_step)
                        tf.summary.scalar("3-loss_policy", losses[2],
                                          update_step)
                        tf.summary.scalar("4-loss_alpha", losses[3],
                                          update_step)
                        tf.summary.scalar("5-alpha", self.nn.alpha,
                                          update_step)

                    if update_step % train_cfg.save_freq == 0:
                        self.checkpoint(save_path, update_step)

            obs = obs_next
            obs_nn = obs_nn_next
            step += 1

        # Save after all training steps
        self.nn.save_network(save_path, name=self.name)
