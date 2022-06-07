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
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import grid2op

from l2rpn_baselines.utils import TrainingParam, NNParam, make_multi_env
try:
    from l2rpn_baselines.DeepQSimple import train as train_dqn
    from l2rpn_baselines.DeepQSimple import evaluate as eval_dqn
    has_DeepQSimple = None
except ImportError as exc_:
    has_DeepQSimple = exc_
try:
    from l2rpn_baselines.DuelQSimple import train as train_d3qs
    from l2rpn_baselines.DuelQSimple import evaluate as eval_d3qs
    has_DuelQSimple = None
except ImportError as exc_:
    has_DuelQSimple = exc_
try:
    from l2rpn_baselines.SACOld import train as train_sacold
    from l2rpn_baselines.SACOld import evaluate as eval_sacold
    has_SACOld = None
except ImportError as exc_:
    has_SACOld = exc_
try:
    from l2rpn_baselines.DuelQLeapNet import train as train_leap
    from l2rpn_baselines.DuelQLeapNet import evaluate as eval_leap
    has_DuelQLeapNet = None
except ImportError as exc_:
    has_DuelQLeapNet = exc_
try:
    from l2rpn_baselines.LeapNetEncoded import train as train_leapenc
    from l2rpn_baselines.LeapNetEncoded import evaluate as eval_leapenc
    has_LeapNetEncoded = None
except ImportError as exc_:
    has_LeapNetEncoded = exc_
try:
    from l2rpn_baselines.DoubleDuelingDQN import train as train_d3qn
    from l2rpn_baselines.DoubleDuelingDQN import evaluate as eval_d3qn
    from l2rpn_baselines.DoubleDuelingDQN import DoubleDuelingDQNConfig as d3qn_cfg
    has_DoubleDuelingDQN = None
except ImportError as exc_:
    has_DoubleDuelingDQN = exc_
try:
    from l2rpn_baselines.DoubleDuelingRDQN import train as train_rqn
    from l2rpn_baselines.DoubleDuelingRDQN import evaluate as eval_rqn
    from l2rpn_baselines.DoubleDuelingRDQN import DoubleDuelingRDQNConfig as rdqn_cfg
    has_DoubleDuelingRDQN = None
except ImportError as exc_:
    has_DoubleDuelingRDQN = exc_
try:
    from l2rpn_baselines.SliceRDQN import train as train_srqn
    from l2rpn_baselines.SliceRDQN import evaluate as eval_srqn
    from l2rpn_baselines.SliceRDQN import SliceRDQN_Config as srdqn_cfg
    has_SliceRDQN = None
except ImportError as exc_:
    has_SliceRDQN = exc_
try:
    from l2rpn_baselines.ExpertAgent import evaluate as eval_expert
    has_ExpertAgent = None
except ImportError as exc_:
    has_ExpertAgent = exc_
    has_SliceRDQN = exc_
    
try:
    from l2rpn_baselines.PPO_RLLIB import train as train_ppo_rllib
    from l2rpn_baselines.PPO_RLLIB import evaluate as eval_ppo_rllib
    has_ppo_rllib = None
except ImportError as exc_:
    has_ppo_rllib = exc_
    
try:
    from l2rpn_baselines.PPO_SB3 import train as train_ppo_sb3
    from l2rpn_baselines.PPO_SB3 import evaluate as eval_ppo_sb3
    has_ppo_sb3 = None
except ImportError as exc_:
    has_ppo_sb3 = exc_


class TestDeepQSimple(unittest.TestCase):
    def setUp(self) -> None:
        if has_DeepQSimple is not None:
            raise ImportError(f"TestDuelQSimple is not available with error:\n{has_DeepQSimple}")

    def test_train_eval(self):
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tp.min_observation = 32
        tmp_dir = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            li_attr_obs_X = ["prod_p", "load_p", "rho"]

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

    def test_train_eval_multiprocess(self):
        # test only done for this baselines because the feature is coded in base class in DeepQAgent
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tp.min_observation = 32
        tmp_dir = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_init = grid2op.make("rte_case5_example", test=True)
            env = make_multi_env(env_init=env_init, nb_env=2)
            li_attr_obs_X = ["prod_p", "load_p", "rho"]

            # neural network architecture
            observation_size = NNParam.get_obs_size(env_init, li_attr_obs_X)
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
                      training_param=tp,
                      verbose=False,
                      kwargs_converters=kwargs_converters,
                      kwargs_archi=kwargs_archi)

            baseline_2 = eval_dqn(env_init,
                                  name=nm_,
                                  load_path=tmp_dir,
                                  logs_path=tmp_dir,
                                  nb_episode=1,
                                  nb_process=1,
                                  max_steps=30,
                                  verbose=False,
                                  save_gif=False)

    def test_train_eval_multimix(self):
        # test only done for this baselines because the feature is coded in base class in DeepQAgent
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tp.min_observation = 32
        tmp_dir = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_neurips_2020_track2", test=True)
            li_attr_obs_X = ["prod_p", "load_p", "rho"]

            # neural network architecture
            observation_size = NNParam.get_obs_size(env, li_attr_obs_X)
            sizes = [100, 50, 10]  # sizes of each hidden layers
            kwargs_archi = {'observation_size': observation_size,
                            'sizes': sizes,
                            'activs': ["relu" for _ in sizes],  # all relu activation function
                            "list_attr_obs": li_attr_obs_X}

            kwargs_converters = {"all_actions": None,
                                 "set_line_status": False,
                                 "change_bus_vect": False,
                                 "set_topo_vect": False
                                 }
            nm_ = "AnneOnymous"
            train_dqn(env,
                      name=nm_,
                      iterations=100,
                      save_path=tmp_dir,
                      load_path=None,
                      logs_dir=tmp_dir,
                      training_param=tp,
                      verbose=False,
                      kwargs_converters=kwargs_converters,
                      kwargs_archi=kwargs_archi)

            for mix in env:
                baseline_2 = eval_dqn(mix,
                                      name=nm_,
                                      load_path=tmp_dir,
                                      logs_path=tmp_dir,
                                      nb_episode=1,
                                      nb_process=1,
                                      max_steps=30,
                                      verbose=False,
                                      save_gif=False)

    def test_train_eval_multi(self):
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tp.min_observation = 32
        tmp_dir = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_init = grid2op.make("rte_case5_example", test=True)
            env = make_multi_env(env_init, 2)

            li_attr_obs_X = ["prod_p", "load_p", "rho"]

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
                      training_param=tp,
                      verbose=False,
                      kwargs_converters=kwargs_converters,
                      kwargs_archi=kwargs_archi)

            baseline_2 = eval_dqn(env_init,
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
        if has_DuelQSimple is not None:
            raise ImportError(f"TestDuelQSimple is not available with error:\n{has_DuelQSimple}")
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tp.min_observation = 32
        tmp_dir = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            li_attr_obs_X = ["prod_p", "load_p", "rho"]

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
            train_d3qs(env,
                      name=nm_,
                      iterations=100,
                      save_path=tmp_dir,
                      load_path=None,
                      logs_dir=tmp_dir,
                      training_param=tp,
                      verbose=False,
                      kwargs_converters=kwargs_converters,
                      kwargs_archi=kwargs_archi)

            baseline_2 = eval_d3qs(env,
                                   name=nm_,
                                   load_path=tmp_dir,
                                   logs_path=tmp_dir,
                                   nb_episode=1,
                                   nb_process=1,
                                   max_steps=30,
                                   verbose=False,
                                   save_gif=False)


class TestSACOld(unittest.TestCase):
    def test_train_eval(self):
        if has_SACOld is not None:
            raise ImportError(f"TestSACOld is not available with error:\n{has_SACOld}")
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tp.min_observation = 32
        tmp_dir = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            li_attr_obs_X = ["prod_p", "load_p", "rho"]

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
            train_sacold(env,
                      name=nm_,
                      iterations=100,
                      save_path=tmp_dir,
                      load_path=None,
                      logs_dir=tmp_dir,
                      training_param=tp,
                      verbose=False,
                      kwargs_converters=kwargs_converters,
                      kwargs_archi=kwargs_archi)

            baseline_2 = eval_sacold(env,
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
        if has_DuelQLeapNet is not None:
            raise ImportError(f"TestLeapNet is not available with error:\n{has_DuelQLeapNet}")
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tp.min_observation = 32
        tmp_dir = tempfile.mkdtemp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            # neural network architecture
            li_attr_obs_X = ["prod_p", "load_p", "rho"]
            li_attr_obs_Tau = ["line_status"]
            sizes = [100, 50, 10]

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


class TestLeapNetEncoded(unittest.TestCase):
    def test_train_eval(self):
        tp = TrainingParam()
        tp.buffer_size = 100
        tp.minibatch_size = 8
        tp.update_freq = 32
        tp.min_observation = 32
        tmp_dir = tempfile.mkdtemp()
        if has_LeapNetEncoded is not None:
            raise ImportError(f"TestLeapNetEncoded is not available with error:\n{has_LeapNetEncoded}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            kwargs_converters = {"all_actions": None,
                                 "set_line_status": False,
                                 "change_line_status": True,
                                 "change_bus_vect": False,
                                 "set_topo_vect": False,
                                 "redispacth": False
                                 }

            # nn architecture
            li_attr_obs_X = ["prod_p", "prod_v", "load_p", "load_q"]
            li_attr_obs_input_q = ["time_before_cooldown_line",
                                   "time_before_cooldown_sub",
                                   "actual_dispatch",
                                   "target_dispatch",
                                   "day_of_week",
                                   "hour_of_day",
                                   "minute_of_hour",
                                   "rho"]
            li_attr_obs_Tau = ["line_status", "timestep_overflow"]
            list_attr_gm_out = ["a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v"] + li_attr_obs_X

            kwargs_archi = {'sizes': [],
                            'activs': [],
                            'x_dim': -1,

                            "list_attr_obs": li_attr_obs_X,
                            "list_attr_obs_tau": li_attr_obs_Tau,
                            "list_attr_obs_x": li_attr_obs_X,
                            "list_attr_obs_input_q": li_attr_obs_input_q,
                            "list_attr_obs_gm_out": list_attr_gm_out,

                            'dim_topo': env.dim_topo,

                            "sizes_enc": (10, 10, 10, 10),
                            "sizes_main": (50, ),
                            "sizes_out_gm": (50,),
                            "sizes_Qnet": (50, 50, )
                            }
            nm_ = "AnneOnymous"
            train_leapenc(env,
                       name=nm_,
                       iterations=100,
                       save_path=tmp_dir,
                       load_path=None,
                       logs_dir=tmp_dir,
                       training_param=tp,
                       verbose=False,
                       kwargs_converters=kwargs_converters,
                       kwargs_archi=kwargs_archi)

            baseline_2 = eval_leapenc(env,
                                   name=nm_,
                                   load_path=tmp_dir,
                                   logs_path=tmp_dir,
                                   nb_episode=1,
                                   nb_process=1,
                                   max_steps=30,
                                   verbose=False,
                                   save_gif=False)


class TestD3QN(unittest.TestCase):
    def test_train_eval(self):
        tmp_dir = tempfile.mkdtemp()
        if has_DoubleDuelingDQN is not None:
            raise ImportError(f"TestD3QN is not available with error:\n{has_DoubleDuelingDQN}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            nm_ = "test_D3QN"

            d3qn_cfg.INITIAL_EPISLON = 1.0
            d3qn_cfg.FINAL_EPISLON = 0.01
            d3qn_cfg.EPISLON_DECAY = 20
            d3qn_cfg.UPDATE_FREQ = 16
            
            train_d3qn(env,
                       name=nm_,
                       iterations=100,
                       save_path=tmp_dir,
                       load_path=None,
                       logs_path=tmp_dir,
                       learning_rate=1e-4,
                       verbose=False,
                       num_pre_training_steps=32,
                       num_frames=4,
                       batch_size=8)

            model_path = os.path.join(tmp_dir, nm_ + ".h5")
            eval_res = eval_d3qn(env,
                                 load_path=model_path,
                                 logs_path=tmp_dir,
                                 nb_episode=1,
                                 nb_process=1,
                                 max_steps=10,
                                 verbose=False,
                                 save_gif=False)

            assert eval_res is not None


class TestRDQN(unittest.TestCase):
    def test_train_eval(self):
        tmp_dir = tempfile.mkdtemp()
        if has_DoubleDuelingRDQN is not None:
            raise ImportError(f"TestRDQN is not available with error:\n{has_DoubleDuelingRDQN}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            nm_ = "test_RDQN"
            rdqn_cfg.INITIAL_EPISLON = 1.0
            rdqn_cfg.FINAL_EPISLON = 0.01
            rdqn_cfg.EPISLON_DECAY = 20
            rdqn_cfg.UPDATE_FREQ = 16

            train_rqn(env,
                      name=nm_,
                      iterations=100,
                      save_path=tmp_dir,
                      load_path=None,
                      logs_path=tmp_dir,
                      learning_rate=1e-4,
                      verbose=False,
                      num_pre_training_steps=16,
                      batch_size=8)

            model_path = os.path.join(tmp_dir, nm_ + ".tf")
            eval_res = eval_rqn(env,
                                load_path=model_path,
                                logs_path=tmp_dir,
                                nb_episode=1,
                                nb_process=1,
                                max_steps=10,
                                verbose=False,
                                save_gif=False)

            assert eval_res is not None


class TestSRDQN(unittest.TestCase):
    def test_train_eval(self):
        tmp_dir = tempfile.mkdtemp()
        if has_SliceRDQN is not None:
            raise ImportError(f"TestSRDQN is not available with error:\n{has_SliceRDQN}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case5_example", test=True)
            nm_ = "test_SRDQN"
            srdqn_cfg.INITIAL_EPISLON = 1.0
            srdqn_cfg.FINAL_EPISLON = 0.01
            srdqn_cfg.EPISLON_DECAY = 20
            srdqn_cfg.UPDATE_FREQ = 16

            train_srqn(env,
                       name=nm_,
                       iterations=100,
                       save_path=tmp_dir,
                       load_path=None,
                       logs_path=tmp_dir,
                       learning_rate=1e-4,
                       verbose=False,
                       num_pre_training_steps=32,
                       batch_size=8)

            model_path = os.path.join(tmp_dir, nm_ + ".tf")
            eval_res = eval_srqn(env,
                                 load_path=model_path,
                                 logs_path=tmp_dir,
                                 nb_episode=1,
                                 nb_process=1,
                                 max_steps=10,
                                 verbose=False,
                                 save_gif=False)

            assert eval_res is not None


class TestExpertAgent(unittest.TestCase):
    def test_train_eval(self):
        if has_ExpertAgent is not None:
            raise ImportError(f"TestExpertAgent is not available with error:\n{has_ExpertAgent}")

        env = grid2op.make("l2rpn_neurips_2020_track1", True)
        res = eval_expert(env, grid="IEEE118_3")
        assert res is not None


class TestPPOSB3(unittest.TestCase):
    def test_train_eval(self):
        tmp_dir = tempfile.mkdtemp()
        if has_ppo_sb3 is not None:
            raise ImportError(f"PPO_SB3 is not available with error:\n{has_ppo_sb3}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True)
            nm_ = "TestPPOSB3"

            train_ppo_sb3(env,
                       name=nm_,
                       iterations=10,
                       save_path=tmp_dir,
                       load_path=None,
                       logs_dir=tmp_dir,
                       learning_rate=1e-4,
                       verbose=False)
            
            agent, eval_res = eval_ppo_sb3(env,
                                 load_path=tmp_dir,
                                 name=nm_,
                                 logs_path=tmp_dir,
                                 nb_episode=1,
                                 nb_process=1,
                                 max_steps=10,
                                 verbose=False,
                                 save_gif=False)
            assert eval_res is not None


class TestPPORLLIB(unittest.TestCase):
    def test_train_eval(self):
        tmp_dir = tempfile.mkdtemp()
        if has_ppo_rllib is not None:
            raise ImportError(f"PPO_RLLIB is not available with error:\n{has_ppo_rllib}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True)
            nm_ = "TestPPORLLIB"

            train_ppo_rllib(env,
                       name=nm_,
                       iterations=1,
                       save_path=tmp_dir,
                       load_path=None,
                       learning_rate=1e-4,
                       verbose=False)
            
            agent, eval_res = eval_ppo_rllib(env,
                                 load_path=tmp_dir,
                                 name=nm_,
                                 logs_path=tmp_dir,
                                 nb_episode=1,
                                 nb_process=1,
                                 max_steps=10,
                                 verbose=False,
                                 save_gif=False)
            assert eval_res is not None


if __name__ == "__main__":
    unittest.main()
