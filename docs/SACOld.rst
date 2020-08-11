SAC: Soft Actor Critic
=========================

This baseline comes from the paper:
`Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_


Description
-----------
This module proposes an implementation of the SAC algorithm.

**This is an old implementation that is probably not correct, it was included out of
backward compatibility with earlier version (< 0.5.0) of this package**

An example to train this model is available in the train function :ref:`Example-sacold`.

Exported class
--------------
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.SACOld import train, evaluate, SACOld

.. automodule:: l2rpn_baselines.SACOld
    :members:
    :autosummary:

Other non exported class
------------------------
These classes need to be imported, if you want to import them with (non exhaustive list):
.. code-block:: python

    from l2rpn_baselines.SACOld.SACOld_NN import SACOld_NN
    from l2rpn_baselines.SACOld.SACOld_NNParam import SACOld_NNParam


.. autoclass:: l2rpn_baselines.SACOld.SACOld_NN.SACOld_NN
    :members:
    :autosummary:

.. autoclass:: l2rpn_baselines.SACOld.SACOld_NNParam.SACOld_NNParam
    :members:
    :autosummary:
