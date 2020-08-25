# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
from itertools import compress

from grid2op.Parameters import Parameters
from grid2op.Agent import BaseAgent

from l2rpn_baselines.SAC.SAC_NN import SAC_NN
from l2rpn_baselines.SAC.SAC_Obs import *
from l2rpn_baselines.SAC.SAC_ReplayBuffer import SAC_ReplayBuffer
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
        self.impact_shape = self.nn_impact_shape(action_space)
        self.nn = SAC_NN(self.observation_shape,
                         self.action_shape,
                         self.impact_shape,
                         nn_config,
                         training=training,
                         verbose=verbose)
        self.training = training
        self.verbose = verbose
        self.sample = self.training
        self._build_sub_maps()

    def _build_sub_maps(self):
        self.sub_iso_pos = []
        self.sub_l_pos = []

        for sub_id in range(self.action_space.n_sub):
            sub_loads = np.where(self.action_space.load_to_subid == sub_id)[0]
            sub_gens = np.where(self.action_space.gen_to_subid == sub_id)[0]
            sub_lor = np.where(self.action_space.line_or_to_subid == sub_id)[0]
            sub_lex = np.where(self.action_space.line_ex_to_subid == sub_id)[0]

            sub_loads_pos = self.action_space.load_pos_topo_vect[sub_loads]
            sub_gens_pos = self.action_space.gen_pos_topo_vect[sub_gens]
            sub_iso_pos = np.concatenate([sub_loads_pos, sub_gens_pos])

            sub_lor_pos = self.action_space.line_or_pos_topo_vect[sub_lor]
            sub_lex_pos = self.action_space.line_ex_pos_topo_vect[sub_lex]
            sub_l_pos = np.concatenate([sub_lor_pos, sub_lex_pos])

            self.sub_iso_pos.append(sub_iso_pos)
            self.sub_l_pos.append(sub_l_pos)

    #####
    ## Grid2op <-> NN converters
    #####
    def nn_observation_shape(self, observation_space):
        obs_size = sac_size_obs(observation_space)
        return (obs_size,)

    def nn_action_shape(self, action_space):
        return (action_space.dim_topo,)

    def nn_impact_shape(self, action_space):
        return (action_space.n_sub,)

    def observation_grid2op_to_nn(self, observation_grid2op):
        return sac_convert_obs(observation_grid2op)

    def clear_target(self):
        self.has_target = False
        self.target_grid2op = None
        self.target_nn = None
        self.target_act_grid2op = []
        self.target_act_nn = []
        self.target_impact_nn = []
        self.target_act_sub = []

    def get_target(self,
                   observation_grid2op,
                   observation_nn):
        if self.verbose:
            sub_fmt = "{:<5}{:<20}{:<20}"
            print (sub_fmt.format("Id:", "Current:",  "Target:"))

        # Get new target
        # Reshape to batch_size 1 for inference
        obs_nn = tf.reshape(observation_nn, (1,) + self.observation_shape)
        self.target_nn, self.impact_nn = self.nn.predict(obs_nn, self.sample)
        self.target_nn = self.target_nn[0]
        self.impact_nn = self.impact_nn[0]
        self.target_grid2op = np.ones_like(self.target_nn, dtype=int)
        self.target_grid2op[self.target_nn > 0.0] = 2

        # Compute forward actions using impact
        sub_idxs = tf.argsort(self.impact_nn, direction='DESCENDING')
        for sub_id in sub_idxs:
            sub_size = self.observation_space.sub_info[sub_id]
            sub_start = np.sum(self.observation_space.sub_info[:sub_id])
            sub_end = sub_start + sub_size

            # Skip small substations
            if sub_size < 4:
                continue

            # Force connectivity on bus 1
            self.target_grid2op[self.sub_l_pos[sub_id][0]] = 1

            # Avoid isolation
            sub_iso_pos = self.sub_iso_pos[sub_id]
            len_iso = len(sub_iso_pos)
            sub_l_pos = self.sub_l_pos[sub_id]
            if len_iso > 0:
                sub_lines = self.target_grid2op[sub_l_pos]
                bus2_lines_disabled = np.all(sub_lines == 1)
                sub_iso = self.target_grid2op[sub_iso_pos]
                bus2_iso_used = np.any(sub_iso == 2)
                if bus2_lines_disabled and bus2_iso_used:
                    self.target_grid2op[sub_iso_pos] = 1

            # Show grid2op target in verbose mode
            if self.verbose:
                sub_fmt = "{:<5}{:<20}{:<20}"
                sub_target = self.target_grid2op[sub_start:sub_end]
                sub_current = observation_grid2op.topo_vect[sub_start:sub_end]
                sub_log = sub_fmt.format(sub_id.numpy(),
                                         str(sub_current),
                                         str(sub_target))
                print (sub_log)

            # Compute grid2op action set bus
            act_v = np.zeros_like(observation_grid2op.topo_vect)
            act_v[sub_start:sub_end] = self.target_grid2op[sub_start:sub_end]
            act_grid2op = self.action_space({"set_bus": act_v})
            self.target_act_grid2op.append(act_grid2op)

            # Substation NN target
            act_nn = np.zeros_like(self.target_nn)
            act_nn[sub_start:sub_end] = self.target_nn[sub_start:sub_end]
            self.target_act_nn.append(act_nn)

            # Substation NN impact
            impact_nn = np.full(self.impact_nn.shape, -1.0)
            impact_nn[sub_id] = self.impact_nn[sub_id]
            self.target_impact_nn.append(impact_nn)

            # Store sub id
            self.target_act_sub.append(sub_id)

        self.has_target = True


    def prune_target(self, observation_grid2op):
        if self.has_target is False:
            return

        # Filter init: Keep all actions
        prune_filter = np.ones(len(self.target_act_sub), dtype=bool)
        # If sub is already in target position, filter out
        for i, sub_id in enumerate(self.target_act_sub):
            act_grid2op = self.target_act_grid2op[i]
            sub_start = np.sum(self.observation_space.sub_info[:sub_id])
            sub_end = sub_start + self.observation_space.sub_info[sub_id]
            act_v = act_grid2op._set_topo_vect[sub_start:sub_end]
            current_v = observation_grid2op.topo_vect[sub_start:sub_end]
            if np.all(act_v == current_v):
                prune_filter[i] = False

        # Apply filter
        self.target_act_grid2op = list(compress(self.target_act_grid2op,
                                                prune_filter))
        self.target_act_nn = list(compress(self.target_act_nn,
                                           prune_filter))
        self.target_impact_nn = list(compress(self.target_impact_nn,
                                              prune_filter))
        self.target_act_sub = list(compress(self.target_act_sub,
                                            prune_filter))

        if len(self.target_act_grid2op) == 0:
            # Consumed completely
            self.clear_target()
            
    def consume_target(self):
        a_grid2op = self.target_act_grid2op.pop(0)
        a_nn = self.target_act_nn.pop(0)
        i_nn = self.target_impact_nn.pop(0)
        sub_id = self.target_act_sub.pop(0)

        if len(self.target_act_grid2op) == 0:
            # Consumed completely
            self.clear_target()

        return a_grid2op, a_nn, i_nn

    ####
    ## grid2op.BaseAgent interface
    ####
    def reset(self, observation_grid2op):
        self.clear_target()

    def act(self, observation_grid2op, reward, done=False):
        obs_nn = self.observation_grid2op_to_nn(observation_grid2op)
        action_grid2op, _, _ = self._act(observation_grid2op, obs_nn)
        return action_grid2op

    def danger(self, observation_grid2op, action_grid2op):
        # Will fail, get a new target
        _, _, done, _ = observation_grid2op.simulate(action_grid2op)
        if done:
            return True

        # Get a target to solve overflows
        if self.has_target is False and \
           np.any(observation_grid2op.rho > 0.95):
            return True

        # Play the action
        return False
    
    def _act(self, observation_grid2op, observation_nn):
        a_grid2op = None
        a_nn = None
        i_nn = None

        self.prune_target(observation_grid2op)
        if self.has_target:
            a_grid2op, a_nn, i_nn = self.consume_target()
            print ("Continue target: ", a_grid2op)
        else:
            a_grid2op = self.action_space({})

        if self.danger(observation_grid2op, a_grid2op):
            self.clear_target()
            self.get_target(observation_grid2op, observation_nn)
            self.prune_target(observation_grid2op)
            if self.has_target:
                a_grid2op, a_nn, i_nn = self.consume_target()
                print ("Start target: ", a_grid2op)
            else:
                a_grid2op = self.action_space({})
                a_nn = None
                i_nn = None

        return a_grid2op, a_nn, i_nn

    def _step(self, env, observation_grid2op, observation_nn):
        dn = self.action_space({})
        if self.danger(observation_grid2op, dn):
            self.clear_target()
            self.get_target(observation_grid2op, observation_nn)
            self.prune_target(observation_grid2op)
            if self.has_target:
                target_nn = self.target_nn
                impact_nn = self.impact_nn
                done = False
                reward = 0.0
                info = {}
                while done is False and self.has_target:
                    a_grid2op, a_nn, i_nn = self.consume_target()
                    print ("Apply:", a_grid2op)
                    obs, r, done, info = env.step(a_grid2op)
                    reward += r
                return obs, reward, done, info, target_nn, impact_nn
        # Not in danger: DN
        obs, reward, done, info = env.step(dn)
        return obs, reward, done, info, None, None

    ###
    ## Baseline train
    ###
    def checkpoint(self, save_path, update_step):
        ckpt_name = "{}-{:04d}".format(self.name, update_step)
        self.nn.save_network(save_path, name=ckpt_name)

    def train_cv(self, env, current_step, total_step):
        params = Parameters()
        difficulty = "unchanged"
        if current_step == 0:
            params.NO_OVERFLOW_DISCONNECTION = True
            params.NB_TIMESTEP_OVERFLOW_ALLOWED = 9999
            params.NB_TIMESTEP_COOLDOWN_SUB = 0
            params.NB_TIMESTEP_COOLDOWN_LINE = 0
            params.HARD_OVERFLOW_THRESHOLD = 9999
            params.NB_TIMESTEP_RECONNECTION = 0
            difficulty = "0"
            env.parameters = params
        elif current_step == int(total_step * 0.05):
            params.NO_OVERFLOW_DISCONNECTION = False
            params.NB_TIMESTEP_OVERFLOW_ALLOWED = 6
            params.NB_TIMESTEP_COOLDOWN_SUB = 0
            params.NB_TIMESTEP_COOLDOWN_LINE = 0
            params.HARD_OVERFLOW_THRESHOLD = 3.0
            params.NB_TIMESTEP_RECONNECTION = 1
            difficulty = "1"
            self.rpbf.clear()
            env.parameters = params
        elif current_step == int(total_step * 0.1):
            params.NO_OVERFLOW_DISCONNECTION = False
            params.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
            params.NB_TIMESTEP_COOLDOWN_SUB = 1
            params.NB_TIMESTEP_COOLDOWN_LINE = 1
            params.HARD_OVERFLOW_THRESHOLD = 2.5
            params.NB_TIMESTEP_RECONNECTION = 6
            difficulty = "2"
            self.rpbf.clear()
            env.parameters = params
        elif current_step == int(total_step * 0.2):
            params.NO_OVERFLOW_DISCONNECTION = False
            params.NB_TIMESTEP_OVERFLOW_ALLOWED = 2
            params.NB_TIMESTEP_COOLDOWN_SUB = 3
            params.NB_TIMESTEP_COOLDOWN_LINE = 3
            params.HARD_OVERFLOW_THRESHOLD = 2.0
            params.NB_TIMESTEP_RECONNECTION = 12
            difficulty = "competition"
            self.rpbf.clear()
            env.parameters = params
        return difficulty
        
    def train(self, env, iterations, save_path, logs_path, train_cfg):
        # Init training vars
        replay_buffer = SAC_ReplayBuffer(train_cfg.replay_buffer_size)
        self.rpbf = replay_buffer
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

        if self.verbose:
            print ("Training for {}".format(iterations))
            print (train_cfg.to_dict())
        
        # Do iterations updates
        while update_step < iterations:
            # Curriculum training
            difficulty = self.train_cv(env, update_step, iterations)

            # New episode
            if done:
                obs = env.reset()
                self.reset(obs)
                obs_nn = self.observation_grid2op_to_nn(obs)
                done = False

            stepped = self._step(env, obs, obs_nn)
            (obs_next, reward, done, info, act_nn, impact_nn) = stepped
            #act_grid2op, act_nn, impact_nn  = self._act(obs, obs_nn)
            #obs_next, reward, done, info = env.step(act_grid2op)
            obs_nn_next = self.observation_grid2op_to_nn(obs_next)

            if info["is_illegal"]:
                episode_illegal += 1
                #print ("Illegal", act_grid2op)
            if done:
                print ("Game over", info)
            episode_steps += 1
            episode_rewards_sum += reward

            if act_nn is not None:
                replay_buffer.add(obs_nn, act_nn, impact_nn,
                                  reward, done, obs_nn_next)
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
                    print("Difficulty:\t{}".format(difficulty))
                    print("Buffer size:\t{}".format(replay_buffer.size()))

                episode_steps = 0
                episode_rewards_sum = 0.0
                episode_illegal = 0

        # Save after all training steps
        self.nn.save_network(save_path, name=self.name)
