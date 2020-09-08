# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import sys
import json
from itertools import compress
import tensorflow as tf

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

        self._precompute()

        self.observation_shape = self.nn_observation_shape(observation_space)
        self.bridge_shape = self.nn_bridge_shape(observation_space)
        self.split_shape = self.nn_split_shape(observation_space)
        self.action_shape = self.nn_action_shape(action_space)
        self.impact_shape = self.nn_impact_shape(action_space)
        self.nn = SAC_NN(self.observation_shape,
                         self.bridge_shape,
                         self.split_shape,
                         self.action_shape,
                         self.impact_shape,
                         nn_config,
                         training=training,
                         verbose=verbose)
        self.training = training
        self.verbose = verbose
        self.threshold = 0.33
        self.target_max_recurs = 2

    def stdout(self, *print_args):
        if self.verbose:
            print (*print_args, file=sys.stdout)

    def stderr(self, *print_args):
        print (*print_args, file=sys.stderr)

    def _precompute(self):
        self.sub_iso_pos = []
        self.sub_l_pos = []
        self.sub_grid2op = []
        self.sub_info = []
        self.sub_topo_mask = []

        # Precompute elements positions for each substation
        for sub_id in range(self.action_space.n_sub):
            if self.action_space.sub_info[sub_id] < 4:
                continue

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
            self.sub_grid2op.append(sub_id)

            sub_start = np.sum(self.action_space.sub_info[:sub_id])
            sub_end = sub_start + self.action_space.sub_info[sub_id]
            sub_pos = np.arange(sub_start, sub_end)
            # Remove first line extremity on current sub
            sub_pos = sub_pos[sub_pos != sub_l_pos[0]]
            self.sub_topo_mask.append(sub_pos)

        self.sub_grid2op = np.array(self.sub_grid2op)
        self.sub_topo_mask = np.concatenate(self.sub_topo_mask)

    #####
    ## Grid2op <-> NN converters
    #####
    def nn_observation_shape(self, observation_space):
        obs_size = sac_size_obs(observation_space)
        return (obs_size,)

    def nn_bridge_shape(self, observation_space):
        bridge_size = observation_space.n_line
        return (bridge_size,)

    def nn_split_shape(self, observation_space):
        split_size = observation_space.n_sub
        return (split_size,)

    def nn_action_shape(self, action_space):
        action_size = len(self.sub_topo_mask) #action_space.dim_topo
        return (action_size,)

    def nn_impact_shape(self, action_space):
        impact_size = len(self.sub_grid2op) #observation_space.n_sub
        return (impact_size,)

    def observation_grid2op_to_nn(self, observation_grid2op):
        obs = sac_convert_obs(observation_grid2op)
        bridge = sac_bridge_obs(observation_grid2op)
        split = sac_split_obs(observation_grid2op)
        return [obs, bridge, split]

    def clear_target(self):
        self.has_target = False
        self.target_grid2op = None
        self.target_nn = None
        self.impact_nn = None
        self.target_act_grid2op = []
        self.target_act_sub = []

    def get_target(self,
                   observation_grid2op,
                   observation_nn,
                   bridge_nn,
                   split_nn,
                   sample=False):

        # Get new target
        # Reshape to batch_size 1 for inference
        o_nn = tf.reshape(observation_nn, (1,) + self.observation_shape)
        b_nn = tf.reshape(bridge_nn, (1,) + self.bridge_shape)
        s_nn = tf.reshape(split_nn, (1,) + self.split_shape)
        self.target_nn, self.impact_nn = self.nn.predict(o_nn,
                                                         b_nn,
                                                         s_nn,
                                                         sample)
        self.target_nn = self.target_nn[0].numpy()
        self.impact_nn = self.impact_nn[0].numpy()
        self.target_grid2op = np.ones(self.action_space.dim_topo, dtype=int)
        target_bus_2 = np.zeros_like(self.target_nn, dtype=bool)
        target_bus_2[self.target_nn > self.threshold] = True
        self.target_grid2op[self.sub_topo_mask[target_bus_2]] = 2        
        #target_disc = np.zeros_like(self.target_nn, dtype=bool)
        #target_disc[self.target_nn < -self.threshold] = True
        #self.target_grid2op[self.sub_topo_mask[target_disc]] = -1

        sub_fmt = "{:<5}{:<25}{:<25}"
        self.stdout(sub_fmt.format("Id:", "Current:",  "Target:"))

        # Compute forward actions using impact
        nn_sub_idxs = np.argsort(self.impact_nn)
        for nn_sub_id in nn_sub_idxs:
            sub_id = self.sub_grid2op[nn_sub_id]
            sub_size = self.observation_space.sub_info[sub_id]
            sub_start = np.sum(self.observation_space.sub_info[:sub_id])
            sub_end = sub_start + sub_size

            # Force connectivity to bus 1 on first line of current sub
            force_line_id = self.sub_l_pos[nn_sub_id][0]
            self.target_grid2op[force_line_id] = 1

            # Avoid load/gen disconnection
            sub_iso_pos = self.sub_iso_pos[nn_sub_id]
            disc = self.target_grid2op[sub_iso_pos]
            disc[disc == -1] = 1
            self.target_grid2op[sub_iso_pos] = disc

            # Avoid isolation
            sub_iso_pos = self.sub_iso_pos[nn_sub_id]
            len_iso = len(sub_iso_pos)
            sub_l_pos = self.sub_l_pos[nn_sub_id]
            if len_iso > 0:
                sub_lines = self.target_grid2op[sub_l_pos]
                bus2_lines_disabled = np.all(sub_lines[sub_lines != -1] == 1)
                sub_iso = self.target_grid2op[sub_iso_pos]
                bus2_iso_used = np.any(sub_iso == 2)
                if bus2_lines_disabled and bus2_iso_used:
                    self.target_grid2op[sub_iso_pos] = 1

            # Show grid2op target in verbose mode
            sub_range = np.arange(sub_start, sub_end)
            sub_range = sub_range[sub_range != force_line_id]
            sub_target = self.target_grid2op[sub_range]
            sub_current = observation_grid2op.topo_vect[sub_range]
            sub_log = sub_fmt.format(sub_id, str(sub_current), str(sub_target))
            self.stdout(sub_log)

            # Compute grid2op action set bus
            act_v = np.zeros_like(observation_grid2op.topo_vect)
            act_v[sub_start:sub_end] = self.target_grid2op[sub_start:sub_end]
            act_grid2op = self.action_space({"set_bus": act_v})
            self.target_act_grid2op.append(act_grid2op)

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
        self.target_act_sub = list(compress(self.target_act_sub,
                                            prune_filter))

        if len(self.target_act_grid2op) == 0:
            # Consumed completely
            self.clear_target()

    def consume_target(self):
        a_grid2op = self.target_act_grid2op.pop(0)
        sub_id = self.target_act_sub.pop(0)

        if len(self.target_act_grid2op) == 0:
            # Consumed completely
            self.clear_target()

        return a_grid2op

    ####
    ## grid2op.BaseAgent interface
    ####
    def reset(self, observation_grid2op):
        self.clear_target()

    def danger(self, observation_grid2op, action_grid2op):
        # Adapted Geirina 2019 WCCI strategy

        # Get a target to solve overflows
        if self.has_target is False and \
           np.any(observation_grid2op.rho > 0.95):
            return True

        # Will fail, get a new target
        _, _, done, _ = observation_grid2op.simulate(action_grid2op)
        if done:
            return True

        # Play the action
        return False

    def act(self, observation_grid2op, reward, done=False):
        nn_in = self.observation_grid2op_to_nn(observation_grid2op)
        (obs_nn, bridge_nn, split_nn) = nn_in
        action_grid2op, _, _ = self._act(observation_grid2op,
                                         obs_nn,
                                         bridge_nn,
                                         split_nn)
        return action_grid2op

    def _act(self,
             observation_grid2op,
             observation_nn,
             bridge_nn,
             split_nn,
             sample=False,
             recurs=0):
        a_grid2op = self.action_space({})
        a_nn = None
        i_nn = None

        # We just computed a target, if it fails, will need to sample
        if self.has_target and recurs >= 1:
            sample = True

        # Consume target action if not the same as current state
        self.prune_target(observation_grid2op)
        if self.has_target:
            a_nn = self.target_nn
            i_nn = self.impact_nn
            a_grid2op = self.consume_target()
            self.stdout("Consume target: ", a_grid2op)

        # Too many tries, go with it
        if recurs > self.target_max_recurs:
            self.stdout("Target Max recurs")
            return a_grid2op, a_nn, i_nn

        # Get a new target if we are in danger with current action
        if self.danger(observation_grid2op, a_grid2op):
            self.stdout("Target danger: Sample=", sample)
            self.clear_target()
            self.get_target(observation_grid2op,
                            observation_nn,
                            bridge_nn,
                            split_nn,
                            sample)
            recurs += 1
            return self._act(observation_grid2op,
                             observation_nn,
                             bridge_nn,
                             split_nn,
                             sample, recurs)
        else: # Not in danger, play current action
            return a_grid2op, a_nn, i_nn

    ###
    ## Baseline train
    ###
    def checkpoint(self, save_path, update_step):
        ckpt_name = "{}-{:04d}".format(self.name, update_step)
        self.nn.save_network(save_path, name=ckpt_name)

    def train_cv(self, env, current_step, total_step, difficulty):
        params = Parameters()
        if current_step == 0:
            params.NO_OVERFLOW_DISCONNECTION = True
            params.NB_TIMESTEP_OVERFLOW_ALLOWED = 9999
            params.NB_TIMESTEP_COOLDOWN_SUB = 0
            params.NB_TIMESTEP_COOLDOWN_LINE = 0
            params.HARD_OVERFLOW_THRESHOLD = 9999
            params.NB_TIMESTEP_RECONNECTION = 0
            difficulty = "0"
            env.parameters = params
        elif difficulty != "1" and current_step == int(total_step * 0.05):
            params.NO_OVERFLOW_DISCONNECTION = False
            params.NB_TIMESTEP_OVERFLOW_ALLOWED = 6
            params.NB_TIMESTEP_COOLDOWN_SUB = 0
            params.NB_TIMESTEP_COOLDOWN_LINE = 0
            params.HARD_OVERFLOW_THRESHOLD = 3.0
            params.NB_TIMESTEP_RECONNECTION = 1
            difficulty = "1"
            self.rpbf.clear()
            env.parameters = params
        elif difficulty != "2" and current_step == int(total_step * 0.1):
            params.NO_OVERFLOW_DISCONNECTION = False
            params.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
            params.NB_TIMESTEP_COOLDOWN_SUB = 1
            params.NB_TIMESTEP_COOLDOWN_LINE = 1
            params.HARD_OVERFLOW_THRESHOLD = 2.5
            params.NB_TIMESTEP_RECONNECTION = 6
            difficulty = "2"
            self.rpbf.clear()
            env.parameters = params
        elif difficulty != "competition" and \
             current_step == int(total_step * 0.2):
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
        self.target_max_recurs = 20
        target_step = 0
        update_step = 0
        step = 0
        episode = 0

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
        tested = False

        # Copy configs in save path
        os.makedirs(save_path, exist_ok=True)
        cfg_path = os.path.join(save_path, "train.json")
        with open(cfg_path, 'w+') as cfp:
            json.dump(train_cfg.to_dict(), cfp, indent=4)
        cfg_path = os.path.join(save_path, "nn.json")
        with open(cfg_path, 'w+') as cfp:
            json.dump(self.nn._cfg.to_dict(), cfp, indent=4)

        self.stdout("Training for {} iterations".format(iterations))
        self.stdout(train_cfg.to_dict())
        difficulty = "None"
        
        # Do iterations updates
        while update_step < iterations:
            if train_cfg.cv:
                # Curriculum training
                difficulty = self.train_cv(env, update_step,
                                           iterations, difficulty)
            else:
                difficulty = "default"

            # New episode
            if done:
                tested = False
                obs = env.reset()
                self.reset(obs)
                done = False
                episode += 1

            # Operate
            nn_in = self.observation_grid2op_to_nn(obs)
            (o_nn, b_nn, s_nn) = nn_in # Obs, bridge, split
            a_grid2op, a_nn, i_nn = self._act(obs, o_nn, b_nn, s_nn, True)
            obs_next, reward, done, info = env.step(a_grid2op)
            s = 1

            if info["is_illegal"]:
                episode_illegal += 1
            if done and \
               info["exception"] is not None and \
               len(info["exception"]) != 0:
                self.stdout("Game over", info)
            elif done:
                self.stdout("Episode success ?!")

            episode_steps += s
            episode_rewards_sum += reward

            # Learn
            if a_nn is not None:
                nn_next = self.observation_grid2op_to_nn(obs_next)
                target_step += 1
                # Save transition to replay buffer
                replay_buffer.add(nn_in, a_nn, i_nn, reward, done, nn_next)
                # Train / Update
                if target_step % train_cfg.update_freq == 0 and \
                   replay_buffer.size() >= train_cfg.batch_size and \
                   replay_buffer.size() >= train_cfg.min_replay_buffer_size:
                    batch = replay_buffer.sample(train_cfg.batch_size)
                    delay = (update_step % 2 == 0)
                    losses = self.nn.train(*batch, delayed=delay)
                    update_step += 1

                    # Save weights sometimes
                    if update_step % train_cfg.save_freq == 0:
                        self.checkpoint(save_path, update_step)

                    # Run test configuration sometimes
                    if update_step % train_cfg.test_freq == 0:
                        self.train_test(env, logger, update_step)
                        done = True
                        tested = True

                    # Tensorboard logging
                    if update_step % train_cfg.log_freq == 0:
                        logger.scalar("001-loss_q1", losses[0])
                        logger.scalar("002-loss_q2", losses[1])
                        if losses[2] is not None:
                            logger.scalar("003-loss_policy", losses[2])
                        if losses[3] is not None:
                            logger.scalar("004-loss_alpha", losses[3])
                        logger.scalar("005-alpha", self.nn.alpha)
                        logger.write(update_step)

            step += s
            obs = obs_next

            # Episode metrics logging
            if done and not tested:
                logger.mean_scalar("010-steps", episode_steps, 10)
                logger.mean_scalar("100-steps", episode_steps, 100)
                logger.mean_scalar("011-illegal", episode_illegal, 10)
                logger.mean_scalar("101-illegal", episode_illegal, 100)
                logger.mean_scalar("012-rewardsum", episode_rewards_sum, 10)
                logger.mean_scalar("102-rewardsum", episode_rewards_sum, 100)
                self.stdout("Global step:\t{:08d}".format(step))
                self.stdout("Update step:\t{:08d}".format(update_step))
                self.stdout("Episode steps:\t{:08d}".format(episode_steps))
                self.stdout("Rewards sum:\t{:.2f}".format(episode_rewards_sum))
                self.stdout("Difficulty:\t{}".format(difficulty))
                self.stdout("Buffer size:\t{}".format(replay_buffer.size()))
                self.stdout("Episode:\t{}".format(episode))

                # Episode metrics reset
                episode_steps = 0
                episode_rewards_sum = 0.0
                episode_illegal = 0

        # Save after all training steps
        self.nn.save_network(save_path, name=self.name)

    def train_test(self, env, logger, update_step):
        # Cache current training scenario
        save_id = env.chronics_handler._prev_cache_id

        # Run test on a few scenarios
        test_v = []
        for scenario_id in [0, 42, 128, 666, 954]:
            env.set_id(scenario_id)

            obs = env.reset()
            self.reset(obs)
            done = False
            r = 0.0
            info = None
            test_step = 0

            while done is False:
                action_grid2op = self.act(obs, r, done)
                obs, r, done, info = env.step(action_grid2op)
                test_step += 1

            self.stdout("Test scenario {}: {}".format(scenario_id, test_step))
            test_v.append(test_step)

        logger.scalar("test", np.mean(test_v))

        # Restore training scenario
        env.chronics_handler.tell_id(save_id)
        env.reset()
