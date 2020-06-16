SAC: Soft Actor Critic
=========================

This baseline comes from the paper:
`Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_


Description
-----------
This module proposes an implementation of the SAC algorithm.

An example to train this model is available in the train function :ref:`Example-sac`.

Exported class
--------------
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.SAC import train, evaluate, SAC

.. automodule:: l2rpn_baselines.SAC
    :members:
    :autosummary:

Other non exported class
------------------------
These classes need to be imported, if you want to import them with (non exhaustive list):
.. code-block:: python

    from l2rpn_baselines.SAC.SAC_NN import SAC_NN
    from l2rpn_baselines.SAC.SAC_NNParam import SAC_NNParam


.. autoclass:: l2rpn_baselines.SAC.SAC_NN.SAC_NN
    :members:
    :autosummary:

.. autoclass:: l2rpn_baselines.SAC.SAC_NNParam.SAC_NNParam
    :members:
    :autosummary:
