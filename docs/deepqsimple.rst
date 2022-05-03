.. currentmodule:: l2rpn_baselines.DeepQSimple

DeepQSimple: A simple implementation of the Deep Q Learning
===========================================================

Description
-----------
This file serves as an concrete example on how to implement a baseline, even more concretely than the "do nothing"
baseline. Don't expect to obtain state of the art method with this simple method however.

An example to train this model is available in the train function :ref:`Example-deepqsimple`

.. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
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


.. autoclass:: l2rpn_baselines.DeepQSimple.deepQ_NN.DeepQ_NN
    :members:
    :autosummary:
