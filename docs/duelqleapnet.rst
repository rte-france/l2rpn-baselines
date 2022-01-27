.. currentmodule:: l2rpn_baselines.DuelQLeapNet

DuelQLeapNet: D3QN with LeapNet
================================

TODO reference the original papers `ESANN Paper <https://hal.archives-ouvertes.fr/hal-02268886>`_
`Leap Net <https://www.sciencedirect.com/science/article/abs/pii/S0925231220305051>`_

That has now be implemented as a github repository `Leap Net Github <https://github.com/BDonnot/leap_net>`_

Description
-----------
The Leap is a type of neural network that has showed really good performances on the predictions of flows on
powerlines based on the injection and the topology.

In this baseline, we use this very same architecture to model the Q function. The D3QN RL method is used.

An example to train this model is available in the train function :ref:`Example-leapnet`.

Exported class
--------------
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.DuelQLeapNet import train, evaluate, DuelQLeapNet

.. automodule:: l2rpn_baselines.DuelQLeapNet
    :members:
    :autosummary:

Other non exported class
------------------------
These classes need to be imported, if you want to import them with (non exhaustive list):
.. code-block:: python

    from l2rpn_baselines.DuelQLeapNet.DuelQLeapNet_NN import DuelQLeapNet_NN
    from l2rpn_baselines.DuelQLeapNet.LeapNet_NNParam import LeapNet_NNParam


.. autoclass:: l2rpn_baselines.DuelQLeapNet.DuelQLeapNet_NN.DuelQLeapNet_NN
    :members:
    :autosummary:

.. autoclass:: l2rpn_baselines.DuelQLeapNet.LeapNet_NNParam.LeapNet_NNParam
    :members:
    :autosummary:
