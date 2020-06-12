DoubleDuelingDQN: A example implementation of Double Duelling Deep Q Network
============================================================================

Description
-----------
This module serves as an concrete example on how to implement a D3QN baseline.
This baseline is of type Double Duelling Deep Q Network, as in Duelling Q Network and DoubleQ update.

It's main purpose is to provide an example of this network type running with Grid2Op. However, don't expect to obtain state of the art results.


Agent class
------------------------
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.DoubleDuelingDQN import DoubleDuelingDQN
    from l2rpn_baselines.DoubleDuelingDQN import train
    from l2rpn_baselines.DoubleDuelingDQN import evaluate

.. automodule:: l2rpn_baselines.DoubleDuelingDQN
    :members:
    :autosummary:

Configuration
------------------------
Training a model requires tweaking many hyperparameters, these can be found in a specific class attributes:

.. code-block:: python

    from l2rpn_baselines.DoubleDuelingDQN import DoubleDuelingDQNConfig

    # Set hyperparameters before training
    DoubleDuelingDQNConfig.LR = 1e-5
    DoubleDuelingDQNConfig.INITAL_EPSILON = 1.0
    DoubleDuelingDQNConfig.FINAL_EPSILON = 0.001
    DoubleDuelingDQNConfig.DECAY_EPSILON = 10000

.. automodule:: l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQNConfig
    :members:
    :undoc-members:

Internal classes
------------------------
The neural network model is defined in a separate class.
You may want to import it manually:

.. code-block:: python

    from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN_NN import DoubleDuelingDQN_NN


.. autoclass:: l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN_NN.DoubleDuelingDQN_NN
    :members:
    :autosummary:
