.. l2rpn-baselines documentation master file, created by
   sphinx-quickstart on Thu Jun  4 16:14:17 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

============================================
Welcome to l2rpn-baselines's documentation!
============================================

In this documentation we expose first what is this package about and how to contribute, and then which baselines
are already implemented in this package.

.. toctree::
   :maxdepth: 2
   :caption: How to contribute

   template
   donothing

Some RL implementation examples
---------------------------------

Lots of reinforcement learning algorithms are already implemented by state of
the art libraries heavily maintained and updated. 

We highly recommend to use such packages if you would like to apply reinforcement
learning to the power grid control problem.

.. toctree::
   :maxdepth: 1
   :caption: Some RL implementation examples

   ppo_rllib
   ppo_stable_baselines
   external_contributions


Expert systems and optimizers
------------------------------

In this section, we grouped up some noticeable contributions for the powergrid control 
problem. 

These solutions comes either from past top performers of the l2rpn competitions, or
from custom implementation of some published research performing well
in some environment.

.. toctree::
   :maxdepth: 1
   :caption: Expert systems and optimizers

   expertagent
   optimcvxpy


Legacy implementations
---------------------------

.. note::
   Most of the codes below are legacy code that will not be updated and contains
   (most likely) lots of bugs, inefficiencies and "not so great" code.

   It's totally fine to use them if you want to dive deep into implementation.
   For most usage however, we strongly encourage you to check out the
   :class:`l2rpn_baselines.PPO_SB3.PPO_SB3` or the 
   :class:`l2rpn_baselines.PPO_RLLIB.PPO_RLLIB`.

For more "in depth" look at what is possible to do, we also wrote some 
custom implementation of some reinforcement learning algorithms.

We do not necessarily recommend to have a deep look at these packages. However,
you can check them out if you need some inspiration of what can be done by
using grid2op more closely that through the gym interface.

.. toctree::
   :maxdepth: 1
   :caption: Legacy implementations

   utils
   deepqsimple
   doubleduelingdqn
   duelqsimple
   duelqleapnet
   doubleduelingrdqn
   leapnetencoded
   sacold

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
