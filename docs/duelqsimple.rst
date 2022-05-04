.. currentmodule:: l2rpn_baselines.DuelQSimple

DuelQSimple: Double Duelling Deep Q Learning
=============================================

TODO reference the original paper

Description
-----------
This file serves as an concrete example on how to implement a baseline, even more concretely than the "do nothing"
baseline. Don't expect to obtain state of the art method with this simple method however.

An example to train this model is available in the train function :ref:`Example-duelqsimple`.

.. warning::
    This baseline recodes entire the RL training procedure. You can use it if you
    want to have a deeper look at Deep Q Learning algorithm and a possible (non 
    optimized, slow, etc. implementation ).
    
    For a much better implementation, you can reuse the code of :class:`l2rpn_baselines.PPO_RLLIB` 
    or the :class:`l2rpn_baselines.PPO_SB3` baseline.
        
Exported class
--------------
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.DuelQSimple import train, evaluate, DuelQSimple

.. automodule:: l2rpn_baselines.DuelQSimple
    :members:
    :autosummary:

Other non exported class
------------------------
These classes need to be imported, if you want to import them with (non exhaustive list):

.. code-block:: python

    from l2rpn_baselines.DuelQSimple.duelQ_NN import DuelQ_NN
    from l2rpn_baselines.DuelQSimple.duelQ_NN import DuelQ_NNParam


.. autoclass:: l2rpn_baselines.DuelQSimple.duelQ_NN.DuelQ_NN
    :members:
    :autosummary:


.. autoclass:: l2rpn_baselines.DuelQSimple.duelQ_NNParam.DuelQ_NNParam
    :members:
    :autosummary:
