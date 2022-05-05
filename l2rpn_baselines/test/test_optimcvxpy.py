# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from pickle import FALSE
import unittest
import warnings
import numpy as np

import grid2op
from grid2op.Action import PlayableAction
from grid2op.Parameters import Parameters

from lightsim2grid import LightSimBackend

from l2rpn_baselines.OptimCVXPY.optimCVXPY import OptimCVXPY

import pdb

class TestOptimCVXPY(unittest.TestCase):
    def _aux_check_type(self, act, line_status=False, redisp=True):
        # return
        types = act.get_types()
        injection, voltage, topology, line, redispatching, storage, curtailment = types
        assert not injection
        assert not voltage
        assert not topology
        if line_status:
            assert line
        else:
            assert not line
        if redisp:
            assert redispatching
        else:
            assert not redispatching
        assert storage
        assert curtailment
    
    def _aux_create_env_setup(self, param=None):
        if param is None:
            param = Parameters()
            param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage",
                                backend=LightSimBackend(),
                                action_class=PlayableAction,
                                param=param,
                                test=True)
        env.set_id(2)
        env.seed(0)
        env.reset()
        env.fast_forward_chronics(215)
        return env
        
    def test_unsafe(self):
        env = self._aux_create_env_setup()
        agent = OptimCVXPY(env.action_space, env, rho_danger=0., margin_th_limit=0.85,
                           alpha_por_error=0.)
        
        obs, reward, done, info = env.step(env.action_space())
        # max rhos of the 3 following step if I do nothing
        max_rhos = [1.0063555, 1.0104821, 1.0110041]
        
        act = agent.act(obs, None, None)
        self._aux_check_type(act)
        obs, reward, done, info = env.step(act)
        assert not info["exception"]
        assert not done
        assert obs.rho.max() < 1.0, f"{obs.rho.max()} >= 1.0"
        assert obs.rho.max() < max_rhos[0], f"{obs.rho.max()} >= {max_rhos[0]}"
        
        act = agent.act(obs, None, None)
        self._aux_check_type(act)
        obs, reward, done, info = env.step(act)
        assert not info["exception"]
        assert not done
        assert obs.rho.max() < 1.0, f"{obs.rho.max()} >= 1.0"
        assert obs.rho.max() < max_rhos[1], f"{obs.rho.max()} >= {max_rhos[1]}"
        
        act = agent.act(obs, None, None)
        self._aux_check_type(act)
        obs, reward, done, info = env.step(act)
        assert not info["exception"]
        assert not done
        assert obs.rho.max() < 1.0, f"{obs.rho.max()} >= 1.0"
        assert obs.rho.max() < max_rhos[2], f"{obs.rho.max()} >= {max_rhos[2]}"
        
    def test_unsafe_linedisc(self):
        env = self._aux_create_env_setup()
        agent = OptimCVXPY(env.action_space, env, rho_danger=0., margin_th_limit=0.85, alpha_por_error=0.)
        
        l_id_disc = 4
        obs, reward, done, info = env.step(env.action_space({"set_line_status": [(l_id_disc, -1)]}))
        assert not done
        assert obs.rho[l_id_disc] <= 1e-6, f"{obs.rho[l_id_disc]} > 1e-6"
        
        # max rhos of the 3 following step if I do nothing
        max_rhos = [1.006486, 1.0111672, 1.0115097]        
        
        act = agent.act(obs, None, None)
        assert agent.flow_computed[l_id_disc] <= 1e-6, f"{agent.flow_computed[l_id_disc]} > 1e-6"
        self._aux_check_type(act)
        obs, reward, done, info = env.step(act)
        assert not info["exception"]
        assert not done
        assert obs.rho.max() < 1.0, f"{obs.rho.max()} >= 1.0"
        assert obs.rho.max() < max_rhos[0], f"{obs.rho.max()} >= {max_rhos[0]}"
        
        act = agent.act(obs, None, None)
        assert agent.flow_computed[l_id_disc] <= 1e-6, f"{agent.flow_computed[l_id_disc]} > 1e-6"
        self._aux_check_type(act)
        obs, reward, done, info = env.step(act)
        assert not info["exception"]
        assert not done
        assert obs.rho.max() < 1.0, f"{obs.rho.max()} >= 1.0"
        assert obs.rho.max() < max_rhos[1], f"{obs.rho.max()} >= {max_rhos[1]}"
        
        act = agent.act(obs, None, None)
        assert agent.flow_computed[l_id_disc] <= 1e-6, f"{agent.flow_computed[l_id_disc]} > 1e-6"
        self._aux_check_type(act)
        obs, reward, done, info = env.step(act)
        assert not info["exception"]
        assert not done
        assert obs.rho.max() < 1.0, f"{obs.rho.max()} >= 1.0"
        assert obs.rho.max() < max_rhos[2], f"{obs.rho.max()} >= {max_rhos[2]}"
    
    def test_safe_do_reco(self):
        env = self._aux_create_env_setup()
        agent = OptimCVXPY(env.action_space,
                           env,
                           rho_safe=9.5,
                           rho_danger=10.,
                           margin_th_limit=0.9,
                           alpha_por_error=0.)
        
        l_id_disc = 4
        obs, reward, done, info = env.step(env.action_space({"set_line_status": [(l_id_disc, -1)]}))
        assert not done
        act = agent.act(obs, None, None)
        types = act.get_types()
        injection, voltage, topology, line, redispatching, storage, curtailment = types
        assert line
    
    def test_safe_dont_reco_cooldown(self):
        param = Parameters()
        param.NB_TIMESTEP_COOLDOWN_LINE = 3
        param.NO_OVERFLOW_DISCONNECTION = True
        env = self._aux_create_env_setup(param=param)
        agent = OptimCVXPY(env.action_space,
                           env,
                           rho_safe=9.5,
                           rho_danger=10.,
                           margin_th_limit=10.,
                           alpha_por_error=0.)
        
        l_id_disc = 4
        # a cooldown applies, agent does not reconnect it
        obs, reward, done, info = env.step(env.action_space({"set_line_status": [(l_id_disc, -1)]}))
        assert not done
        act = agent.act(obs, None, None)
        types = act.get_types()
        injection, voltage, topology, line, redispatching, storage, curtailment = types
        assert not line
        # still a cooldown
        obs, reward, done, info = env.step(env.action_space())
        assert not done
        act = agent.act(obs, None, None)
        types = act.get_types()
        injection, voltage, topology, line, redispatching, storage, curtailment = types
        assert not line
        # still a cooldown
        obs, reward, done, info = env.step(env.action_space())
        assert not done
        act = agent.act(obs, None, None)
        types = act.get_types()
        injection, voltage, topology, line, redispatching, storage, curtailment = types
        assert not line
        
        # no more cooldown, it should reconnect it
        obs, reward, done, info = env.step(env.action_space())
        assert not done
        act = agent.act(obs, None, None)
        types = act.get_types()
        injection, voltage, topology, line, redispatching, storage, curtailment = types
        assert line
        obs, reward, done, info = env.step(act)
        assert obs.line_status[l_id_disc]
    
    def test_safe_setback_redisp(self):
        env = self._aux_create_env_setup()
        agent = OptimCVXPY(env.action_space,
                           env,
                           rho_safe=9.5,
                           rho_danger=10.,
                           margin_th_limit=10.0,
                           weight_storage_target=0.,
                           alpha_por_error=0.
                           )
        act_prev = env.action_space()
        act_prev.redispatch = [3.0, 4.0, 0.0, 0.0, 0.0, -7.0]
        obs, reward, done, info = env.step(act_prev)
        disp_ref = 1.0 * obs.actual_dispatch
        assert not done
        act = agent.act(obs, None, None)
        obs, reward, done, info = env.step(act)
        assert not done
        # now check that it has set back the redispatching to a closer value to the reference
        assert np.sum(obs.actual_dispatch**2) < np.sum(disp_ref**2)  
        
    def test_safe_setback_storage(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        param.ACTIVATE_STORAGE_LOSS = False  # otherwise it makes tests more complicated
        env = self._aux_create_env_setup()
        agent = OptimCVXPY(env.action_space,
                           env,
                           rho_safe=9.5,
                           rho_danger=10.,
                           margin_th_limit=10.0,
                           weight_redisp_target=0.,
                           alpha_por_error=0.
                           )
        act_prev = env.action_space()
        act_prev.storage_p = [4.9, -9.9]
        obs, reward, done, info = env.step(act_prev)
        obs, reward, done, info = env.step(act_prev)
        obs, reward, done, info = env.step(act_prev)
        obs, reward, done, info = env.step(act_prev)
        for i in range(5): # more than 5 iterations and I got an error due to rounding
            obs_before = obs.copy()
            act = agent.act(obs_before, None, None)
            obs, reward, done, info = env.step(act)
            assert not info["exception"]
            assert not done
            assert (np.sum((obs.storage_charge - 0.5 * obs.storage_Emax)**2) <= 
                    np.sum((obs_before.storage_charge - 0.5 * obs.storage_Emax)**2)), f"error at iteration {i}"
        
        env = self._aux_create_env_setup()
        act_prev = env.action_space()
        act_prev.storage_p = [4.9, 9.9]
        obs, reward, done, info = env.step(act_prev)
        obs, reward, done, info = env.step(act_prev)
        obs, reward, done, info = env.step(act_prev)
        obs, reward, done, info = env.step(act_prev)
        for i in range(5): # more than 5 iterations and I got an error due to rounding
            obs_before = obs.copy()
            agent._DEBUG = True
            act = agent.act(obs_before, None, None)
            obs, reward, done, info = env.step(act)
            assert not info["exception"], f"error at iteration {i}"
            assert not done, f"error at iteration {i}"
            assert (np.sum((obs.storage_charge - 0.5 * obs.storage_Emax)**2) <= 
                    np.sum((obs_before.storage_charge - 0.5 * obs.storage_Emax)**2)), f"error at iteration {i}"
    def test_run_dc(self):
        env = self._aux_create_env_setup()
        agent = OptimCVXPY(env.action_space,
                           env,
                           alpha_por_error=0.,
                           )
        obs = env.get_obs()
        agent.flow_computed[:] = np.NaN
        agent.run_dc(obs)
        assert np.all(np.isfinite(agent.flow_computed))
        init_flow = 1.0 * agent.flow_computed
        
        obs, reward, done, info = env.step(env.action_space())
        agent.flow_computed[:] = np.NaN
        agent.run_dc(obs)
        assert np.all(np.isfinite(agent.flow_computed))
        after_flow = 1.0 * agent.flow_computed
        assert np.all(init_flow != after_flow)
    
    def test_without_storage(self):
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(env_name, test=True)
        obs = env.reset()
        agent = OptimCVXPY(env.action_space, env, alpha_por_error=0.)
        conv = agent.run_dc(obs)
        assert conv
        act = agent.act(obs, 1.0, False)
        obs, reward, done, info = env.step(act)
        act = agent.act(obs, 1.0, False)
        
        
if __name__ == '__main__':
    unittest.main()