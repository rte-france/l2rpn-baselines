.. l2rpn-baselines documentation master file, created by
   sphinx-quickstart on Thu Jun  4 16:14:17 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to l2rpn-baselines's documentation!
===========================================

In this documentation we expose first what is this package about and how to contribute, and then which baselines
are already implemented in this package.

How to contribute
------------------

.. toctree::
   :maxdepth: 2

   template
   donothing

Baseline already Available
---------------------------

These are the "baselines" that are available. Please note that each of these baselines
is provided as an example of what can be achieved with grid2op.

It can serve a possible implementation for a usecase. At the moment, we do not provide
baseline with hyper parameters tuned that performs correctly.

.. toctree::
   :maxdepth: 2

   utils
   deepqsimple
   doubleduelingdqn
   duelqsimple
   expertagent
   ppo_stable_baselines


More advanced baselines
------------------------

.. toctree::
   :maxdepth: 2

   duelqleapnet
   doubleduelingrdqn
   leapnetencoded
   external_contributions


Deprecated baselines
---------------------------

.. toctree::
   :maxdepth: 2

   sacold


Contributions
-------------

TODO

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
