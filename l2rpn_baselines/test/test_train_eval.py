# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

# test that the baselines can be imported
import os
import unittest
import warnings
import tempfile

import grid2op
from l2rpn_baselines.utils import TrainingParam, NNParam
from l2rpn_baselines.DeepQSimple import train as train_dqn
from l2rpn_baselines.DeepQSimple import evaluate as eval_dqn
from l2rpn_baselines.DuelQSimple import train as train_d3qn
from l2rpn_baselines.DuelQSimple import evaluate as eval_d3qn
from l2rpn_baselines.SAC import train as train_sac
from l2rpn_baselines.SAC import evaluate as eval_sac
from l2rpn_baselines.DuelQLeapNet import train as train_leap
from l2rpn_baselines.DuelQLeapNet import evaluate as eval_leap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestDeepQSimple(unittest.TestCase):
    def test_train_eval(self):
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tmp_dir = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                             "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                             "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]

            # neural network architecture
            observation_size = NNParam.get_obs_size(env, li_attr_obs_X)
            sizes = [100, 50, 10]  # sizes of each hidden layers
            kwargs_archi = {'observation_size': observation_size,
                            'sizes': sizes,
                            'activs': ["relu" for _ in sizes],  # all relu activation function
                            "list_attr_obs": li_attr_obs_X}

            kwargs_converters = {"all_actions": None,
                                 "set_line_status": False,
                                 "change_bus_vect": True,
                                 "set_topo_vect": False
                                 }
            nm_ = "AnneOnymous"
            train_dqn(env,
                      name=nm_,
                      iterations=100,
                      save_path=tmp_dir,
                      load_path=None,
                      logs_dir=tmp_dir,
                      nb_env=1,
                      training_param=tp,
                      verbose=False,
                      kwargs_converters=kwargs_converters,
                      kwargs_archi=kwargs_archi)

            baseline_2 = eval_dqn(env,
                                  name=nm_,
                                  load_path=tmp_dir,
                                  logs_path=tmp_dir,
                                  nb_episode=1,
                                  nb_process=1,
                                  max_steps=30,
                                  verbose=False,
                                  save_gif=False)


class TestDuelQSimple(unittest.TestCase):
    def test_train_eval(self):
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tmp_dir = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                             "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                             "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]

            # neural network architecture
            observation_size = NNParam.get_obs_size(env, li_attr_obs_X)
            sizes = [100, 50, 10]  # sizes of each hidden layers
            kwargs_archi = {'observation_size': observation_size,
                            'sizes': sizes,
                            'activs': ["relu" for _ in sizes],  # all relu activation function
                            "list_attr_obs": li_attr_obs_X}

            kwargs_converters = {"all_actions": None,
                                 "set_line_status": False,
                                 "change_bus_vect": True,
                                 "set_topo_vect": False
                                 }
            nm_ = "AnneOnymous"
            train_d3qn(env,
                      name=nm_,
                      iterations=100,
                      save_path=tmp_dir,
                      load_path=None,
                      logs_dir=tmp_dir,
                      nb_env=1,
                      training_param=tp,
                      verbose=False,
                      kwargs_converters=kwargs_converters,
                      kwargs_archi=kwargs_archi)

            baseline_2 = eval_d3qn(env,
                                  name=nm_,
                                  load_path=tmp_dir,
                                  logs_path=tmp_dir,
                                  nb_episode=1,
                                  nb_process=1,
                                  max_steps=30,
                                  verbose=False,
                                  save_gif=False)


class TestSAC(unittest.TestCase):
    def test_train_eval(self):
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tmp_dir = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                             "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                             "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]

            # neural network architecture
            observation_size = NNParam.get_obs_size(env, li_attr_obs_X)
            sizes_q = [100, 50, 10]  # sizes of each hidden layers
            sizes_v = [100, 100]  # sizes of each hidden layers
            sizes_pol = [100, 10]  # sizes of each hidden layers
            kwargs_archi = {'observation_size': observation_size,
                            'sizes': sizes_q,
                            'activs': ["relu" for _ in range(len(sizes_q))],
                            "list_attr_obs": li_attr_obs_X,
                            "sizes_value": sizes_v,
                            "activs_value": ["relu" for _ in range(len(sizes_v))],
                            "sizes_policy": sizes_pol,
                            "activs_policy": ["relu" for _ in range(len(sizes_pol))]
                            }

            kwargs_converters = {"all_actions": None,
                                 "set_line_status": False,
                                 "change_bus_vect": True,
                                 "set_topo_vect": False
                                 }
            nm_ = "AnneOnymous"
            train_sac(env,
                      name=nm_,
                      iterations=100,
                      save_path=tmp_dir,
                      load_path=None,
                      logs_dir=tmp_dir,
                      nb_env=1,
                      training_param=tp,
                      verbose=False,
                      kwargs_converters=kwargs_converters,
                      kwargs_archi=kwargs_archi)

            baseline_2 = eval_sac(env,
                                  name=nm_,
                                  load_path=tmp_dir,
                                  logs_path=tmp_dir,
                                  nb_episode=1,
                                  nb_process=1,
                                  max_steps=30,
                                  verbose=False,
                                  save_gif=False)


class TestLeapNet(unittest.TestCase):
    def test_train_eval(self):
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tmp_dir = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                             "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                             "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]

            # neural network architecture
            li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                             "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                             "time_before_cooldown_sub", "timestep_overflow", "line_status", "rho"]
            li_attr_obs_Tau = ["rho", "line_status"]
            sizes = [800, 800, 800, 494, 494, 494]

            x_dim = NNParam.get_obs_size(env, li_attr_obs_X)
            tau_dims = [NNParam.get_obs_size(env, [el]) for el in li_attr_obs_Tau]

            kwargs_archi = {'sizes': sizes,
                            'activs': ["relu" for _ in sizes],
                            'x_dim': x_dim,
                            'tau_dims': tau_dims,
                            'tau_adds': [0.0 for _ in range(len(tau_dims))],
                            'tau_mults': [1.0 for _ in range(len(tau_dims))],
                            "list_attr_obs": li_attr_obs_X,
                            "list_attr_obs_tau": li_attr_obs_Tau
                            }

            kwargs_converters = {"all_actions": None,
                                 "set_line_status": False,
                                 "change_bus_vect": True,
                                 "set_topo_vect": False
                                 }
            nm_ = "AnneOnymous"
            train_leap(env,
                       name=nm_,
                       iterations=100,
                       save_path=tmp_dir,
                       load_path=None,
                       logs_dir=tmp_dir,
                       nb_env=1,
                       training_param=tp,
                       verbose=False,
                       kwargs_converters=kwargs_converters,
                       kwargs_archi=kwargs_archi)

            baseline_2 = eval_leap(env,
                                   name=nm_,
                                   load_path=tmp_dir,
                                   logs_path=tmp_dir,
                                   nb_episode=1,
                                   nb_process=1,
                                   max_steps=30,
                                   verbose=False,
                                   save_gif=False)


if __name__ == "__main__":
    unittest.main()