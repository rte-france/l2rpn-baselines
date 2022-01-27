.. currentmodule:: l2rpn_baselines.DoubleDuelingRDQN

DoubleDuelingRDQN: A example implementation of Recurrent DoubleQ Network
========================================================================

Description
-----------
This module serves as an concrete example on how to implement a recurrent D3QN baseline.
This baseline is of type Recurrent Double Duelling Deep Q Network, as in Duelling Q, DoubleQ update and recurrent neural network.

It's main purpose is to provide an example of this network type running with Grid2Op. However, don't expect to obtain state of the art results.


Agent class
------------------------
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.DoubleDuelingRDQN import DoubleDuelingRDQN
    from l2rpn_baselines.DoubleDuelingRDQN import train
    from l2rpn_baselines.DoubleDuelingRDQN import evaluate

.. automodule:: l2rpn_baselines.DoubleDuelingRDQN.DoubleDuelingRDQN
    :members:
    :autosummary:

Configuration
------------------------
Training a model requires tweaking many hyperparameters, these can be found in a specific class attributes:

.. code-block:: python

    from l2rpn_baselines.DoubleDuelingRDQN import DoubleDuelingRDQNConfig

    # Set hyperparameters before training
    DoubleDuelingRDQNConfig.LR = 1e-5
    DoubleDuelingRDQNConfig.TRACE_LENGTH = 12

.. automodule:: l2rpn_baselines.DoubleDuelingRDQN.DoubleDuelingRDQNConfig
    :members:
    :undoc-members:

Internal classes
------------------------
The neural network model is defined in a separate class.
You may want to import it manually:

.. code-block:: python

    from l2rpn_baselines.DoubleDuelingRDQN.DoubleDuelingRDQN_NN import DoubleDuelingRDQN_NN


.. autoclass:: l2rpn_baselines.DoubleDuelingRDQN.DoubleDuelingRDQN_NN.DoubleDuelingRDQN_NN
    :members:
    :autosummary:
