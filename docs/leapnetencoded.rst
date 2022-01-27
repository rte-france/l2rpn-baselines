.. currentmodule:: l2rpn_baselines.LeapNetEncoded

LeapNetEncoded: D3QN on a state encoded by a leap net
======================================================

TODO reference the original papers `ESANN Paper <https://hal.archives-ouvertes.fr/hal-02268886>`_
`Leap Net <https://www.sciencedirect.com/science/article/abs/pii/S0925231220305051>`_

That has now be implemented as a github repository `Leap Net Github <https://github.com/BDonnot/leap_net>`_

Description
-----------
The Leap is a type of neural network that has showed really good performances on the predictions of flows on
powerlines based on the injection and the topology.

In this baseline, we use this very same architecture to model encode the powergrid state (at a given
step).

Then this embedding of the powergrid is used by a neural network (that can be a regular network or
a leap net) that parametrized the Q function.

An example to train this model is available in the train function :ref:`Example-leapnetenc`.

Exported class
--------------
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.LeapNetEncoded import train, evaluate, LeapNetEncoded

.. automodule:: l2rpn_baselines.LeapNetEncoded
    :members:
    :autosummary:

Other non exported class
------------------------
These classes need to be imported, if you want to import them with (non exhaustive list):

.. code-block:: python

    from l2rpn_baselines.LeapNetEncoded.leapNetEncoded_NN import LeapNetEncoded_NN
    from l2rpn_baselines.LeapNetEncoded.leapNetEncoded_NNParam import LeapNetEncoded_NNParam


.. autoclass:: l2rpn_baselines.LeapNetEncoded.leapNetEncoded_NN.LeapNetEncoded_NN
    :members:
    :autosummary:

.. autoclass:: l2rpn_baselines.LeapNetEncoded.leapNetEncoded_NNParam.LeapNetEncoded_NNParam
    :members:
    :autosummary:
