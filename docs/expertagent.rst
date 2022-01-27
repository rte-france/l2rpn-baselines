.. currentmodule:: l2rpn_baselines.ExpertAgent

ExpertAgent: A example implementation of using ExpertOpForGrid for empirical overflow solving
=============================================================================================

Description
-----------
This Agent uses a greedy algorithm to try to solve an overflow by trying many different topological actions.

To use this agent, please follow the install steps in ExpertAgent/READ_ME.md : https://github.com/rte-france/l2rpn-baselines/tree/master/l2rpn_baselines/ExpertAgent/READ_ME.md

You can find the documentation on the algorithm and it's usage on : https://github.com/marota/ExpertOp4Grid

The readthedocs documentation can be found here : https://expertop4grid.readthedocs.io/en/latest/

Agent class
-----------
This agent does not train, it is only a closed system analysis to help decision making to solve an overload.
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.ExpertAgent import ExpertAgent
    from l2rpn_baselines.ExpertAgent import eval_expertagent

.. automodule:: l2rpn_baselines.ExpertAgent.ExpertAgent
    :members:
    :autosummary:
