DeepQSimple: A simple implementation of the Deep Q Learning
===========================================================

Description
-----------
This file serves as an concrete example on how to implement a baseline, even more concretely than the "do nothing"
baseline. Don't expect to obtain state of the art method with this simple method however.

An example to train this model is available in the train function :ref:`Example-deepqsimple`

Exported class
--------------
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.DeepQSimple import train, evaluate, DeepQSimple

.. automodule:: l2rpn_baselines.DeepQSimple
    :members:
    :autosummary:

Other non exported class
------------------------
These classes need to be imported, if you want to import them with (non exhaustive list):
.. code-block:: python

    from l2rpn_baselines.DeepQSimple.DeepQ_NN import DeepQ_NN


.. autoclass:: l2rpn_baselines.DeepQSimple.DeepQ_NN.DeepQ_NN
    :members:
    :autosummary:
