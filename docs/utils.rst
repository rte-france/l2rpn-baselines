.. currentmodule:: l2rpn_baselines.utils

Some utilitary functions and classes
====================================

Description
-----------
In this files are present a few utilitary scripts used in some other baselines, or that could be used
in different context.

They have been put together in this "module" as they can be reused for different baselines (avoid code duplicate)

The main tools are:

- :class:`BaseDeepQ` which is the root class for some baselines. This class holds only the code
  of the neural network. The architecture of the neural network can be customized thanks to the
  :class:`NNParam`
- :class:`DeepQAgent` this class will create an instance of :class:`BaseDeepQ` and
  will implement the agent interface (*eg* the `train`, `load` and `save` methods). The training procedure is
  unified (epsilon greedy for exploration, training  for a certain amount of steps etc.) but can be customized with
  :class:`TrainingParam`. The training procedure can be stopped at any given time and restarted from the last point
  almost flawlessly, it saves it neural network frequently as well as the other parameters etc.
- :class:`TrainingParam` allows to customized for some "common" procedure how to train the agent. More information
  can be gathered in the :ref:`training_param` section. This is fully serializable / de serializable in json format.
- :class:`NNParam` is used to specify the architecture of your neural network. Just like :class:`TrainingParam` this
  class also fully supports serialization / de serialization in json format. More about it is specified in the
  section :ref:`nn_param`

.. _training_param:

Focus on the training parameters
--------------------------------
TODO


.. _nn_param:

Focus on the architecture
--------------------------
TODO

Implementation Details
-----------------------
.. automodule:: l2rpn_baselines.utils
    :members:
    :autosummary:
