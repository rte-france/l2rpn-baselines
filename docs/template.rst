How to contribute to l2rpn baselines
=====================================

Description
-----------
A Baseline is a :class:`grid2op.Agent.BaseAgent` with a few more methods that allows to easily load / write and train
it.

I can then be used as any grid2op Agent, for example in a runner or doing the "while" open gym loop.

Compared to bare grid2op Agent, baselines have 3 more methods:
- :func:`Template.load`: to load the agent, if applicable
- :func:`Template.save`: to save the agent, if applicable
- :func:`Template.train`: to train the agent, if applicable

The method :func:`Template.reset` is already present in grid2op but is emphasized here. It is called
by a runner at the beginning of each episode with the first observation.

The method :func:`Template.act` is also present in grid2op, of course. It the main method of the baseline,
that receives an observation (and a reward and flag that says if an episode is over or not) an return a valid
action.

**NB** the "real" instance of environment on which the baseline will be evaluated will be built AFTER the creation
of the baseline. The parameters of the real environment on which the baseline will be assessed will belong to the
same class than the argument used by the baseline. This means that if a baseline is built with a grid2op
environment "env", this environment will not be modified in any manner, all it's internal variable will not
change etc. This is done to prevent cheating.

Implementation Example
-----------------------
.. automodule:: l2rpn_baselines.Template
    :members:
    :autosummary:
