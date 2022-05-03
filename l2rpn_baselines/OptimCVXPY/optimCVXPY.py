# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from typing import Optional
import logging
import warnings
import cvxpy as cp
import numpy as np

import grid2op
from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from grid2op.Action import PlayableAction, ActionSpace, BaseAction
from grid2op.Backend import PandaPowerBackend
from grid2op.Observation import BaseObservation
from lightsim2grid import LightSimBackend
from lightsim2grid.gridmodel import init

import pdb

# TODO: "predictive control"
# TODO: no flow in constraints but in objective function
# TODO: reuse previous computations
class OptimCVXPY(BaseAgent):
    """
    This agent choses its action by resolving, at each `agent.act(...)` call an optimization routine
    that is then converted to a grid2op action.

    It has 3 main behaviours:

    - `safe grid`: when the grid is safe, it tries to get back to an "original" state. It will
    gradually cancel all past redispatching and curtailment action and aim at a storage state
    of charge close to `0.5 * Emax` for all storage units. If the grid is safe this agent can
    also take some actions to reconnect powerlines.
    - `unsafe grid`: when the grid is unsafe, it tries to set it back to a "safe" state (all flows
    below their thermal limit) by optimizing storage units, curtailment and redispatching only.
    (This agent does not perform topological actions in this state)
    - `intermediate grid`: in this state the agent does nothing. This state is mainly present
    to avoid this agent to "oscillate" between safe and unsafe states.

    The "behaviours" in which the agent is in depends on the maximum power flow (in percent)
    of the grid. If the maximum power flow is below a certain threshold (`rho_safe`), the agent is in the
    "safe grid" state. If the maximum power flow is above a certain threshold (`rho_danger`), the agent is in
    "unsafe grid" state.

    Have a look at the documentation for more details about the optimization problems
    solved in each case.
    
    
    Parameters
    ----------
    action_space : `grid2op.Action.ActionSpace`
        The action space of the environment.
        
    _powerlines_x: `cp.Parameter`	
        The reactance of each powerline / transformer in the network given in per unit !
    
    _margin_th_limit: `cp.Parameter`
        In the "unsafe state" this agent will try to minimize the thermal limit violation.
        
        A "thermal limit violation" is defined as having a flow (in dc) above 
        `margin_th_limit * thermal_limit_mw`.
        
        The model is particularly sensitive to this parameter.
        
    rho_danger: `float`
        If any `obs.rho` is above `rho_danger`, then the agent will use the
        "unsafe grid" optimization routine and try to apply curtailment,
        redispatching and action on storage unit to set back the grid into a safe state.
        
    rho_safe: `float`
        If all `obs.rho` are below `rho_safe`, then the agent will use the
        "safe grid" optimization routine and try to set back the grid into
        a reference state.
    
    nb_max_bus: `int`	
        Maximum number of buses allowed in the powergrid.
    
    _penalty_curtailment: ` cp.Parameter`

    _penalty_redispatching: `cp.Parameter`

    _penalty_storag: `cp.Parameter`
        
    bus_or: `cp.Parameter`
    
    bus_ex: `cp.Parameter`
    
    bus_load: `cp.Parameter`
    
    bus_gen: `cp.Parameter`
    
    bus_storage: `cp.Parameter`
        
    load_per_bus: `cp.Parameter`
    
    gen_per_bus: `cp.Parameter`
        
    redisp_up: `cp.Parameter`
    
    redisp_down: `cp.Parameter`
        
    curtail_down: `cp.Parameter`
    
    curtail_up: `cp.Parameter`
    
    storage_down: `cp.Parameter`
    
    storage_up: `cp.Parameter`
        
    th_lim_mw: `cp.Parameter`
    
    flow_computed: `np.ndarray`
    
    margin_rounding: `float`
    
    margin_sparse: `float`
    
    logger: `logging.Logger`
        A logger to log information about the optimization process.
    
    """
    SOLVER_TYPES = [cp.SCS, cp.OSQP, cp.SCIPY]
    
    def __init__(self,
                 action_space : ActionSpace,
                 env : Environment,
                 lines_x_pu: Optional[np.array]=None,
                 margin_th_limit: float=0.9,
                 rho_danger: float=0.95,
                 rho_safe: float=0.85,
                 penalty_curtailment: float=0.1,
                 penalty_redispatching: float=0.03,
                 penalty_storage: float=0.3,
                 margin_rounding: float=0.01,
                 margin_sparse: float=1e-4,
                 logger : Optional[logging.Logger]=None) -> None:
        """Initialize this class

        Parameters
        ----------
        action_space : `grid2op.Action.ActionSpace`
            The action space of the environment.
            
        env: `grid2op.Environment.Environment`:
            The environment in which the agent evolves.
            
            If `lines_x_pu` is not provided, then this agent will attempt to read the
            reactance of each powerlines and transformer from the environment backend.
            
        lines_x_pu: `np.ndarray`	
            The reactance of each powerline / transformer in the network.
            
            It is optional and if it's not provided, then the reactance will be read from the
            environment.
        
        margin_th_limit: `float`
            In the "unsafe state" this agent will try to minimize the thermal limit violation.
            
            A "thermal limit violation" is defined as having a flow (in dc) above 
            `margin_th_limit * thermal_limit_mw`.
            
            The model is particularly sensitive to this parameter.
            
        rho_danger: `float`
            If any `obs.rho` is above `rho_danger`, then the agent will use the
            "unsafe grid" optimization routine and try to apply curtailment,
            redispatching and action on storage unit to set back the grid into a safe state.
            
        rho_safe: `float`
            If all `obs.rho` are below `rho_safe`, then the agent will use the
            "safe grid" optimization routine and try to set back the grid into
            a reference state.
        
        penalty_curtailment: `float`
            The cost of applying a curtailment in the objective function.
        
            Default value is 0.1.
            
        penalty_redispatching: `float`
            The cost of applying a redispatching in the objective function.
            
            Default value is 0.03.
        
        penalty_storage: `float`
            The cost of applying a storage in the objective function.
              
            Default value is 0.3.
                   
        margin_rounding: `float`
            A margin taken to avoid rounding issues that could lead to infeasible
            actions due to "redispatching above max_ramp_up" for example.
            
        margin_sparse: `float`
            A margin taken when converting the output of the optimization routine
            to grid2op actions: if some values are below this value, then they are
            set to zero.
            
        logger: `logging.Logger`
            A logger to log information about the optimization process.
    
        Raises
        ------
        ValueError
            If you provide a `lines_x_pu` that is not of the same size as the number of powerlines
            
        RuntimeError
            In case the lines reactance are not provided and cannot 
            be inferred from the environment.
            
        """
        BaseAgent.__init__(self, action_space)
        self._margin_th_limit: cp.Parameter = cp.Parameter(value=margin_th_limit,
                                                           nonneg=True)
        self._penalty_curtailment: cp.Parameter = cp.Parameter(value=penalty_curtailment,
                                                               nonneg=True)
        self._penalty_redispatching: cp.Parameter = cp.Parameter(value=penalty_redispatching,
                                                                 nonneg=True)
        self._penalty_storage: cp.Parameter = cp.Parameter(value=penalty_storage,
                                                           nonneg=True)
        
        self.margin_rounding: float = float(margin_rounding)
        self.margin_sparse: float = float(margin_sparse)
        self.rho_danger: float = float(rho_danger)
        self.rho_safe: float = float(rho_safe)
        
        if lines_x_pu is not None:
            powerlines_x = 1.0 * np.array(lines_x_pu).astype(float)
        elif isinstance(env.backend, LightSimBackend): 
            # read the powerline x (reactance) from
            # lightsim grid
            powerlines_x = np.array(
                [float(el.x_pu) for el in env.backend._grid.get_lines()] + 
                [float(el.x_pu) for el in env.backend._grid.get_trafos()]) 
        elif isinstance(env.backend, PandaPowerBackend):
            # read the powerline x (reactance) from
            # pandapower grid
            pp_net = env.backend._grid
            grid_model = init(pp_net) 
            powerlines_x = np.array(
                [float(el.x_pu) for el in grid_model.get_lines()] + 
                [float(el.x_pu) for el in grid_model.get_trafos()]) 
        else:
            # no powerline information available
            raise RuntimeError(f"Unkown backend type: {type(env.backend)}. If you want to use "
                               "OptimCVXPY, you need to provide the reactance of each powerline / "
                               "transformer in per unit in the `lines_x` parameter.")
        if powerlines_x.shape[0] != env.n_line:
            raise ValueError("The number of lines in the grid is not the same as the number "
                                "of lines in provided lines_x")
        if np.any(powerlines_x <= 0.):
            raise ValueError("All powerline reactance must be strictly positive")
        
        self._powerlines_x: cp.Parameter = cp.Parameter(shape=powerlines_x.shape,
                                                        value=1.0 * powerlines_x,
                                                        pos=True)
        
        # TODO replace all below with sparse matrices
        # to be able to change the topology more easily
        self.nb_max_bus: int = 2 * env.n_sub
        self.bus_or: cp.Parameter = cp.Parameter(shape=env.n_line,
                                                 value=env.line_or_to_subid,
                                                 integer=True)
        self.bus_ex: cp.Parameter = cp.Parameter(shape=env.n_line,
                                                 value=env.line_ex_to_subid,
                                                 integer=True)
        self.bus_load: cp.Parameter = cp.Parameter(shape=env.n_load,
                                                   value=env.load_to_subid,
                                                   integer=True)
        self.bus_gen: cp.Parameter = cp.Parameter(shape=env.n_gen,
                                                  value=env.gen_to_subid,
                                                  integer=True)
        self.bus_storage: cp.Parameter = cp.Parameter(shape=env.n_storage,
                                                      value=env.storage_to_subid,
                                                      integer=True)
        
        this_zeros_ = np.zeros(self.nb_max_bus)
        self.load_per_bus: cp.Parameter = cp.Parameter(shape=self.nb_max_bus,
                                                       value=1.0 * this_zeros_,
                                                        nonneg=True)
        self.gen_per_bus: cp.Parameter = cp.Parameter(shape=self.nb_max_bus,
                                                      value=1.0 * this_zeros_,
                                                      nonneg=True)
        
        self.redisp_up: cp.Parameter = cp.Parameter(shape=self.nb_max_bus,
                                                    value=1.0 * this_zeros_,
                                                    nonneg=True)
        self.redisp_down: cp.Parameter = cp.Parameter(shape=self.nb_max_bus,
                                                      value=1.0 * this_zeros_,
                                                      nonneg=True)
        
        self.curtail_down: cp.Parameter = cp.Parameter(shape=self.nb_max_bus,
                                                       value=1.0 * this_zeros_,
                                                       nonneg=True)
        self.curtail_up: cp.Parameter = cp.Parameter(shape=self.nb_max_bus,
                                                     value=1.0 * this_zeros_,
                                                     nonneg=True)
        
        self.storage_down: cp.Parameter = cp.Parameter(shape=self.nb_max_bus,
                                                       value=1.0 * this_zeros_,
                                                       nonneg=True)
        self.storage_up: cp.Parameter = cp.Parameter(shape=self.nb_max_bus,
                                                     value=1.0 * this_zeros_,
                                                     nonneg=True)
        
        self._th_lim_mw: cp.Parameter = cp.Parameter(shape=env.n_line,
                                                     value=env.get_thermal_limit(),
                                                     nonneg=True)
        
        if logger is None:
            self.logger: logging.Logger = logging.getLogger(__name__)
            self.logger.disabled = False
        else:
            self.logger: logging.Logger = logger.getChild("OptimCVXPY")

        self.flow_computed = np.zeros(env.n_line, dtype=float)
        self.flow_computed[:] = np.NaN
        
    @property
    def margin_th_limit(self) -> cp.Parameter:
        return self._margin_th_limit
    
    @margin_th_limit.setter
    def margin_th_limit(self, val: float):
        self._margin_th_limit = float(val)
        
    @property
    def penalty_curtailment(self) -> cp.Parameter:
        return self._penalty_curtailment
    
    @penalty_curtailment.setter
    def penalty_curtailment(self, val: float):
        self._penalty_curtailment = float(val)
        
    @property
    def penalty_redispatching(self) -> cp.Parameter:
        return self._penalty_redispatching
    
    @penalty_redispatching.setter
    def penalty_redispatching(self, val: float):
        self._penalty_redispatching = float(val)
        
    @property
    def penalty_storage(self) -> cp.Parameter:
        return self._penalty_storage
    
    @penalty_storage.setter
    def penalty_storage(self, val: float):
        self._penalty_storage = float(val)
        
    def _update_topo_param(self, obs: BaseObservation):
        tmp_ = 1 * obs.line_or_to_subid
        tmp_ [obs.line_or_bus == 2] += obs.n_sub
        self.bus_or.value[:] = tmp_
        tmp_ = 1 * obs.line_ex_to_subid
        tmp_ [obs.line_ex_bus == 2] += obs.n_sub
        self.bus_ex.value[:] = tmp_
        
        # "disconnect" in the model the line disconnected
        # it should be equilavent to connect them all (at both side) to the slack
        self.bus_ex.value [(obs.line_or_bus == -1) | (obs.line_ex_bus == -1)] = 0
        self.bus_or.value [(obs.line_or_bus == -1) | (obs.line_ex_bus == -1)] = 0
         
        tmp_ = obs.load_to_subid
        tmp_[obs.load_bus == 2] += obs.n_sub
        self.bus_load.value[:] = tmp_
        
        tmp_ = obs.gen_to_subid
        tmp_[obs.gen_bus == 2] += obs.n_sub
        self.bus_gen.value[:] = tmp_
        
        tmp_ = obs.storage_to_subid
        tmp_[obs.storage_bus == 2] += obs.n_sub
        self.bus_storage.value[:] = tmp_
        
    def _update_th_lim_param(self, obs: BaseObservation):
        self._th_lim_mw.value[:] =  (0.001 * obs.thermal_limit)**2 * obs.v_or **2 * 3. - obs.q_or**2
        self._th_lim_mw.value[:] = np.sqrt(self._th_lim_mw.value)
        # TODO what if it's negative !
        
    def _update_inj_param(self, obs: BaseObservation):
        self.load_per_bus.value[:] = 0.
        self.gen_per_bus.value[:] = 0.
        load_p = 1.0 * obs.load_p
        load_p *= (obs.gen_p.sum() - obs.storage_power.sum()) / load_p.sum() 
        for bus_id in range(self.nb_max_bus):
            self.load_per_bus.value[bus_id] += load_p[self.bus_load.value == bus_id].sum()
            self.load_per_bus.value[bus_id] += obs.storage_power[self.bus_storage.value == bus_id].sum()
            self.gen_per_bus.value[bus_id] += obs.gen_p[self.bus_gen.value == bus_id].sum()

    def _add_redisp_const(self, obs: BaseObservation, bus_id: int):
        # add the constraint on the redispatching
        self.redisp_up.value[bus_id] = obs.gen_margin_up[self.bus_gen.value == bus_id].sum()
        self.redisp_down.value[bus_id] = obs.gen_margin_down[self.bus_gen.value == bus_id].sum()
    
    def _add_storage_const(self, obs: BaseObservation, bus_id: int):
        # limit in MW
        stor_down = obs.storage_max_p_prod[self.bus_storage.value == bus_id].sum()
        # limit due to energy (if almost empty)
        stor_down = np.minimum(stor_down,
                                obs.storage_charge[self.bus_storage.value == bus_id].sum() * (60. / obs.delta_time) 
                                )
        self.storage_down.value[bus_id] = stor_down
        
        # limit in MW
        stor_up = obs.storage_max_p_absorb[self.bus_storage.value == bus_id].sum()
        # limit due to energy (if almost full)
        stor_up = np.minimum(stor_up,
                                (obs.storage_Emax - obs.storage_charge)[self.bus_storage.value == bus_id].sum() * (60. / obs.delta_time) 
                                )
        self.storage_up.value[bus_id] = stor_up
            
    def _update_constraints_param_unsafe(self, obs: BaseObservation):
        tmp_ = 1.0 * obs.gen_p
        tmp_[~obs.gen_renewable] = 0.
        
        for bus_id in range(self.nb_max_bus):
            # redispatching
            self._add_redisp_const(obs, bus_id) 
            
            # curtailment
            self.curtail_down.value[bus_id] = 0.
            self.curtail_up.value[bus_id] = tmp_[(self.bus_gen.value == bus_id) & obs.gen_renewable].sum()
            
            # storage
            self._add_storage_const(obs, bus_id)
            
        self._remove_margin_rounding()
        
    def _remove_margin_rounding(self):
        self.storage_down.value[self.storage_down.value > self.margin_rounding] -= self.margin_rounding
        self.storage_up.value[self.storage_up.value > self.margin_rounding] -= self.margin_rounding
        self.curtail_down.value[self.curtail_down.value > self.margin_rounding] -= self.margin_rounding
        self.curtail_up.value[self.curtail_up.value > self.margin_rounding] -= self.margin_rounding
        self.redisp_up.value[self.redisp_up.value > self.margin_rounding] -= self.margin_rounding
        self.redisp_down.value[self.redisp_down.value > self.margin_rounding] -= self.margin_rounding
        
    def _validate_param_values(self):
        self.storage_down._validate_value(self.storage_down.value)
        self.storage_up._validate_value(self.storage_up.value)
        self.curtail_down._validate_value(self.curtail_down.value)
        self.curtail_up._validate_value(self.curtail_up.value)
        self.redisp_up._validate_value(self.redisp_up.value)
        self.redisp_down._validate_value(self.redisp_down.value)
        self._th_lim_mw._validate_value(self._th_lim_mw.value)
         
    def update_parameters(self, obs: BaseObservation, unsafe: bool = True):
        ## update the topology information
        self._update_topo_param(obs)
        
        ## update the thermal limit
        self._update_th_lim_param(obs)
        
        ## update the load / gen bus injected values
        self._update_inj_param(obs)

        ## update the constraints parameters
        if unsafe:
            self._update_constraints_param_unsafe(obs)
        else:
            self._update_constraints_param_safe(obs)
        
        # check that all parameters have correct values
        # for example non negative values for non negative parameters
        self._validate_param_values()
    
    def _aux_compute_kcl(self, inj_bus, f_or):
        KCL_eq = []
        for bus_id in range(self.nb_max_bus):
            tmp = inj_bus[bus_id]
            if np.any(self.bus_or.value == bus_id):
                tmp +=  cp.sum(f_or[self.bus_or.value == bus_id])
            if np.any(self.bus_ex.value == bus_id):
                tmp -=  cp.sum(f_or[self.bus_ex.value == bus_id])
            KCL_eq.append(tmp)
        return KCL_eq
    
    def _mask_theta_zero(self):
        theta_is_zero = np.full(self.nb_max_bus, True, bool)
        theta_is_zero[self.bus_or.value] = False
        theta_is_zero[self.bus_ex.value] = False
        theta_is_zero[self.bus_load.value] = False
        theta_is_zero[self.bus_gen.value] = False
        theta_is_zero[self.bus_storage.value] = False
        theta_is_zero[0] = True  # slack bus
        return theta_is_zero
        
    def compute_optimum_unsafe(self):
        # variables
        theta = cp.Variable(shape=self.nb_max_bus)  # at each bus
        curtailment_mw = cp.Variable(shape=self.nb_max_bus)  # at each bus
        storage = cp.Variable(shape=self.nb_max_bus)  # at each bus
        redispatching = cp.Variable(shape=self.nb_max_bus)  # at each bus
        
        # usefull quantities
        f_or = cp.multiply(1. / self._powerlines_x , (theta[self.bus_or.value] - theta[self.bus_ex.value]))
        inj_bus = (self.load_per_bus + storage) - (self.gen_per_bus + redispatching - curtailment_mw)
        energy_added = cp.sum(curtailment_mw) + cp.sum(storage) - cp.sum(redispatching)
        
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)
        theta_is_zero = self._mask_theta_zero()
        
        # constraints
        constraints = ( # slack bus
                        [theta[theta_is_zero] == 0] + 
                        
                        # KCL
                        [el == 0 for el in KCL_eq] +
                        
                        # limit redispatching to possible values
                        [redispatching <= self.redisp_up, redispatching >= -self.redisp_down] +
                        # limit curtailment
                        [curtailment_mw <= self.curtail_up, curtailment_mw >= -self.curtail_down] +
                        # limit storage
                        [storage <= self.storage_up, storage >= -self.storage_down] +
                        
                        # bus and generator variation should sum to 0. (not sure it's mandatory)
                        [energy_added == 0]
                      )
        
        # objective
        # cost = cp.norm1(gp_var) + cp.norm1(lp_var)
        cost = ( self._penalty_curtailment * cp.sum_squares(curtailment_mw) + 
                 self._penalty_storage * cp.sum_squares(storage) +
                 self._penalty_redispatching * cp.sum_squares(redispatching) +
                 cp.sum_squares(cp.pos(cp.abs(f_or) - self._margin_th_limit * self._th_lim_mw))
        )
        
        # solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob)
        
        if has_converged:
            self.flow_computed[:] = f_or.value
            res = (curtailment_mw.value, storage.value, redispatching.value)
        else:
            self.logger.error(f"Problem with the optimization for all tested solvers ({type(self).SOLVER_TYPES})")
            self.flow_computed[:] = np.NaN
            tmp_ = np.zeros(shape=self.nb_max_bus)
            res = (1.0 * tmp_, 1.0 * tmp_, 1.0 * tmp_)
        
        return  res
    
    def _solve_problem(self, prob, solver_type=None):
        """
        try different solvers until one finds a good solution...
        Not pretty at all...
        """
        if solver_type is None:
            for solver_type in type(self).SOLVER_TYPES:
                res = self._solve_problem(prob, solver_type=solver_type)
                if res:
                    self.logger.info(f"Solver {solver_type} has converged. Stopping there.")
                    return True
            return False
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                tmp_ = prob.solve(solver=solver_type)
                
            if np.isfinite(tmp_):
                return True
            else:
                self.logger.warning(f"Problem with the optimization for {solver_type}, infinite value")
                raise cp.error.SolverError("Infinite value")

        except cp.error.SolverError as exc_:
            self.logger.warning(f"Problem with the optimization for {solver_type}: {exc_}")
            return False
            
    def _clean_vect(self, curtailment, storage, redispatching):
        """remove the value too small and set them at 0."""
        curtailment[np.abs(curtailment) <= self.margin_sparse] = 0.
        storage[np.abs(storage) <= self.margin_sparse] = 0.
        redispatching[np.abs(redispatching) <= self.margin_sparse] = 0.
        
    def to_grid2op(self,
                   obs,
                   curtailment: np.ndarray,
                   storage: np.ndarray,
                   redispatching: np.ndarray) -> BaseAction:
        self._clean_vect(curtailment, storage, redispatching)
        
        act = self.action_space()
        
        # storage
        storage_ = np.zeros(shape=act.n_storage)
        storage_[:] = storage[self.bus_storage.value]
        # TODO what is multiple storage on a single bus ?
        act.storage_p = storage_
        
        # curtailment
        # becarefull here, the curtailment is given by the optimizer
        # in the amount of MW you remove, grid2op
        # expects a maximum value
        curtailment_ = np.zeros(shape=act.n_gen) -1.
        gen_curt = obs.gen_renewable & (obs.gen_p > 0.1)
        idx_gen = self.bus_gen.value[gen_curt]
        tmp_ = curtailment[idx_gen]
        modif_gen_optim = tmp_ != 0.
        gen_p = 1.0 * obs.gen_p
        aux_ = curtailment_[gen_curt]
        aux_[modif_gen_optim] = (gen_p[gen_curt][modif_gen_optim] - 
                                 tmp_[modif_gen_optim] * 
                                 gen_p[gen_curt][modif_gen_optim] / 
                                 self.gen_per_bus.value[idx_gen][modif_gen_optim]
        )
        aux_[~modif_gen_optim] = -1.
        curtailment_[gen_curt] = aux_
        curtailment_[~gen_curt] = -1.
        act.curtail_mw = curtailment_
        
        # redispatching
        redisp_ = np.zeros(obs.n_gen)
        gen_redi = obs.gen_redispatchable & (obs.gen_p > 0.1)
        idx_gen = self.bus_gen.value[gen_redi]
        tmp_ = redispatching[idx_gen]
        redisp_[gen_redi] = tmp_ *  gen_p[gen_redi] / self.gen_per_bus.value[idx_gen]
        redisp_[~gen_redi] = 0.
        act.redispatch = redisp_
        return act
    
    def _update_constraints_param_safe(self, obs):
        tmp_ = 1.0 * obs.gen_p
        tmp_[~obs.gen_renewable] = 0.
        
        for bus_id in range(self.nb_max_bus):
            # redispatching
            self._add_redisp_const(obs, bus_id) 
            
            # storage
            self._add_storage_const(obs, bus_id)
            
            # curtailment
            # self.curtail_down.value[bus_id] = 0.
            # self.curtail_up.value[bus_id] = tmp_[(self.bus_gen.value == bus_id) & obs.gen_renewable].sum()
            
        self._remove_margin_rounding()
    
    def compute_optimum_safe(self, obs: BaseObservation, l_id=None):
        if l_id is not None:
            # TODO why reconnecting it on busbar 1 ?
            self.bus_ex.value[l_id] = obs.line_ex_to_subid[l_id]
            self.bus_or.value[l_id] = obs.line_or_to_subid[l_id]
        
        # variables
        theta = cp.Variable(shape=self.nb_max_bus)  # at each bus
        curtailment_mw = cp.Variable(shape=self.nb_max_bus)  # at each bus
        storage = cp.Variable(shape=self.nb_max_bus)  # at each bus
        redispatching = cp.Variable(shape=self.nb_max_bus)  # at each bus
        
        #  stuff to put elsewhere (TODO)
        past_dispatch = cp.Parameter(shape=self.nb_max_bus,
                                     value=np.zeros(self.nb_max_bus)
                                     )  # at each bus
        for bus_id in range(self.nb_max_bus):
            past_dispatch.value[bus_id] = obs.target_dispatch[self.bus_gen.value == bus_id].sum()
        past_state_of_charge = cp.Parameter(shape=self.nb_max_bus,
                                            value=np.zeros(self.nb_max_bus),
                                            nonneg=True
                                            )  # at each bus
        for bus_id in range(self.nb_max_bus):
            past_state_of_charge.value[bus_id] = obs.storage_charge[self.bus_storage.value == bus_id].sum()
        
        # TODO put that in constructor with possibility to modify it !
        SoC = np.zeros(shape=self.nb_max_bus)
        for bus_id in range(self.nb_max_bus):
            SoC[bus_id] = 0.5 * obs.storage_Emax[self.bus_storage.value == bus_id].sum()
        storage_target = cp.Parameter(shape=self.nb_max_bus,
                                      value=1.0 * SoC,
                                      nonneg=True)
        
        # usefull quantities
        f_or = cp.multiply(1. / self._powerlines_x , (theta[self.bus_or.value] - theta[self.bus_ex.value]))
        inj_bus = (self.load_per_bus + storage) - (self.gen_per_bus + redispatching - curtailment_mw)
        energy_added = cp.sum(curtailment_mw) + cp.sum(storage) - cp.sum(redispatching)
        
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)
        theta_is_zero = self._mask_theta_zero()
        
        dispatch_after_this = past_dispatch + redispatching
        state_of_charge_after = past_state_of_charge + storage / (60. / obs.delta_time)
        
        # constraints
        constraints = ( # slack bus
                        [theta[theta_is_zero] == 0] + 
                        
                        # KCL
                        [el == 0 for el in KCL_eq] +
                        
                        # I impose here that the flows are bellow the limits
                        [f_or <= self._margin_th_limit * self._th_lim_mw] +
                        [f_or >= -self._margin_th_limit * self._th_lim_mw] +
                        
                        # limit redispatching to possible values
                        [redispatching <= self.redisp_up, redispatching >= -self.redisp_down] +
                        # limit curtailment
                        [curtailment_mw <= self.curtail_up, curtailment_mw >= -self.curtail_down] +
                        # limit storage
                        [storage <= self.storage_up, storage >= -self.storage_down] +
                        
                        # bus and generator variation should sum to 0. (not sure it's mandatory)
                        [energy_added == 0]
                      )
        
        # TODO (in ctor) redisp_target
        # TODO (in ctor) curtail_target
        
        # objective
        # cost = cp.norm1(gp_var) + cp.norm1(lp_var)
        cost = ( self._penalty_curtailment * cp.sum_squares(curtailment_mw) + 
                 self._penalty_storage * cp.sum_squares(storage) +
                 self._penalty_redispatching * cp.sum_squares(redispatching) +
                 cp.sum_squares(dispatch_after_this)  +
                 cp.sum_squares(state_of_charge_after - storage_target)
        )
        
        # solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob)
        
        if has_converged:
            self.flow_computed[:] = f_or.value
            res = (curtailment_mw.value, storage.value, redispatching.value)
        else:
            self.logger.error(f"Problem with the optimization for all tested solvers ({type(self).SOLVER_TYPES})")
            self.flow_computed[:] = np.NaN
            tmp_ = np.zeros(shape=self.nb_max_bus)
            res = (1.0 * tmp_, 1.0 * tmp_, 1.0 * tmp_)
        
        return  res
    
    def act(self,
            obs: BaseObservation,
            reward: float,
            done: bool) -> BaseAction:
        
        self.flow_computed[:] = np.NaN
        if obs.rho.max() > self.rho_danger:
            # I attempt to make the grid more secure
            
            # update the observation
            self.update_parameters(obs)
            # solve the problem
            curtailment, storage, redispatching = self.compute_optimum_unsafe()
            # get back the grid2op representation
            act = self.to_grid2op(obs, curtailment, storage, redispatching)
        elif obs.rho.max() < self.rho_safe:
            # I attempt to get back to a more robust state (reconnect powerlines,
            # storage state of charge close to the target state of charge,
            # redispatching close to 0.0 etc.)
            act = self.action_space()
            
            can_be_reco = (obs.time_before_cooldown_line == 0) & (~obs.line_status)
            l_id = None
            if np.any(can_be_reco):
                # powerlines are not in cooldown
                # I attempt to reconnect one of them (first one in the list)
                l_id = np.where(can_be_reco)[0][0]
                # TODO optimization to chose the "best" line to reconnect
                act.line_set_status = [(l_id, +1)]
            
            # TODO
            self.update_parameters(obs, unsafe=False)
            curtailment, storage, redispatching = self.compute_optimum_safe(obs, l_id)
            # get back the grid2op representation
            act = self.to_grid2op(obs, curtailment, storage, redispatching)
        else:
            # I do nothing between rho_danger and rho_safe
            act = self.action_space()
        return act
        
if __name__ == "__main__":
    pass
