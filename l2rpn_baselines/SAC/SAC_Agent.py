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
from l2rpn_baselines.utils import TensorboardLogger

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
                         training=training,
                         verbose=verbose)
        self.training = training
        self.verbose = verbose

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
        if np.any(observation_grid2op.rho > 0.90):
            return True
        action_grid2op = self.action_space({})
        _, _, done, _ = observation_grid2op.simulate(action_grid2op)
        return done

    def clear_target(self):
        self.has_target = False
        self.target_topo = None
        self.target_nn = None

    def set_target(self, nn_target):
        self.has_target = True
        self.target_nn = np.array(nn_target)
        self.target_topo = np.zeros_like(nn_target, dtype=int)
        self.target_topo[self.target_nn > 0.0] = 1
        self.target_topo += 1
        if self.verbose:
            print ("New target")
            s_beg = 0
            s_end = self.observation_space.sub_info[0]
            for sub_id in range(self.observation_space.n_sub):
                print (sub_id, self.target_topo[s_beg:s_end], sep="\t")
                s_beg = s_end
                s_end += self.observation_space.sub_info[sub_id]

    def return_target(self, observation_grid2op):
        current = observation_grid2op.topo_vect
        act_v = np.zeros_like(current)

        sub_ids = np.arange(observation_grid2op.n_sub)
        self.space_prng.shuffle(sub_ids)
        for sub_id in sub_ids:
            sub_start = np.sum(observation_grid2op.sub_info[:sub_id])
            sub_end = sub_start + observation_grid2op.sub_info[sub_id]
            if np.any(current[sub_start:sub_end] != 1):
                act_v[sub_start:sub_end] = 1
                break

        action_grid2op = self.action_space({"set_bus": act_v})
        return action_grid2op
        
    def consume_target(self, observation_grid2op):
        current = observation_grid2op.topo_vect
        target = self.target_topo
        target_nn = self.target_nn

        act_v = np.array(observation_grid2op.topo_vect)

        sub_ids = np.arange(observation_grid2op.n_sub)
        self.space_prng.shuffle(sub_ids)
        for sub_id in sub_ids:
            sub_start = np.sum(observation_grid2op.sub_info[:sub_id])
            sub_end = sub_start + observation_grid2op.sub_info[sub_id]
            if np.any(current[sub_start:sub_end] != target[sub_start:sub_end]):
                act_v[sub_start:sub_end] = target[sub_start:sub_end]
                break

        act_nn = np.array(act_v, dtype=np.float32)
        act_nn[act_nn == 1] = -1.0
        act_nn[act_nn == 2] = 1.0

        # Clean up if consumed completely
        if np.all(target == act_v):
            print ("Consumed target")
            self.clear_target()

        # Adjust grid2op action
        # Do Nothing for unchanged elements
        act_v[act_v == current] = 0

        action_grid2op = self.action_space({"set_bus": act_v})
        print ("Consume step", action_grid2op)
        return action_grid2op, act_nn

    ####
    ## grid2op.BaseAgent interface
    ####
    def reset(self, observation_grid2op):
        self.clear_target()

    def act(self, observation_grid2op, reward, done=False):
        obs_nn = self.observation_grid2op_to_nn(observation_grid2op)
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
            nn_target = self.nn.predict(obs_nn, self.training)[0]
            self.set_target(nn_target)
            act_grid2op, act_nn = self.consume_target(observation_grid2op)
        else:
            act_grid2op = self.return_target(observation_grid2op)

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
        logger = TensorboardLogger(self.name, logpath)
        episode_steps = 0
        episode_rewards_sum = 0.0
        episode_illegal = 0

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

            if info["is_illegal"]:
                episode_illegal += 1
                print ("Illegal", act_grid2op)
            if done:
                print ("Game over", info)
            episode_steps += 1
            episode_rewards_sum += reward

            if act_nn is not None:
                replay_buffer.add(obs_nn, act_nn, reward, done, obs_nn_next)
                target_step += 1

                if target_step % train_cfg.update_freq == 0 and \
                   replay_buffer.size() >= train_cfg.batch_size and \
                   replay_buffer.size() >= train_cfg.min_replay_buffer_size:
                    batch = replay_buffer.sample(train_cfg.batch_size)
                    losses = self.nn.train(*batch)
                    update_step += 1

                    if update_step % train_cfg.log_freq == 0:
                        logger.scalar("001-loss_q1", losses[0])
                        logger.scalar("002-loss_q2", losses[1])
                        logger.scalar("003-loss_policy", losses[2])
                        logger.scalar("004-loss_alpha", losses[3])
                        logger.scalar("005-alpha", self.nn.alpha)
                        logger.write(update_step)

                    if update_step % train_cfg.save_freq == 0:
                        self.checkpoint(save_path, update_step)

            obs = obs_next
            obs_nn = obs_nn_next
            step += 1

            if done:
                logger.mean_scalar("010-steps", episode_steps, 10)
                logger.mean_scalar("100-steps", episode_steps, 100)
                logger.mean_scalar("011-illegal", episode_illegal, 10)
                logger.mean_scalar("101-illegal", episode_illegal, 100)
                logger.mean_scalar("012-rewardsum", episode_rewards_sum, 10)
                logger.mean_scalar("102-rewardsum", episode_rewards_sum, 100)
                if self.verbose:
                    print("Global step:\t{:08d}".format(step))
                    print("Update step:\t{:08d}".format(update_step))
                    print("Episode steps:\t{:08d}".format(episode_steps))
                    print("Rewards sum:\t{:.2f}".format(episode_rewards_sum))

                episode_steps = 0
                episode_rewards_sum = 0.0
                episode_illegal = 0

        # Save after all training steps
        self.nn.save_network(save_path, name=self.name)
