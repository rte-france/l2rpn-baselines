.. currentmodule:: l2rpn_baselines.OptimCVXPY

OptimCVXPY: A example implementation of an agent based on an optimizer
=======================================================================

.. note::
    This "baseline" uses optimization and only performs (using optimization) actions on
    continuous variables.

    If you want a more general agent, you can use the:
    
    - :mod:`l2rpn_baselines.ExpertAgent.ExpertAgent` to perform actions on discrete variables 
      (especially the topology) using some heuristics
    - `grid2op_milp_agent <https://github.com/rte-france/grid2op-milp-agent>`_ that also uses
      an optimization package (in this case google "or-tools") to perform topological
      actions. The integration of this baseline in l2rpn-baselines is in progress.

Description
-----------
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

This agent adopts the DC approximation in its optimization routine. In the 
current formulation, it is "greedy" and does not "look ahead", though it would be possible.

safe grid
++++++++++
TODO: explain the optimization problem solved!

unsafe grid
++++++++++++
The goal in this case is to get back  in a safe state as quickly as possible.

To that end, the agent will minimize the violation of thermal limit. To avoid undesired behaviour
where the agent would do too much redispatching / curtailment / storage (by saturating 
its constraints for example) you have the possibility also to add a penalty of such in
the optimization problem with the parameters `penalty_curtailment`, `penalty_redispatching` and	
`penalty_storage`.

Agent class
-----------
This agent does not train, it is only a closed system analysis to help decision making to solve an overload.
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.OptimCVXPY import OptimCVXPY
    from l2rpn_baselines.OptimCVXPY import evaluate

.. automodule:: l2rpn_baselines.OptimCVXPY.OptimCVXPY
    :members:
    :autosummary:
