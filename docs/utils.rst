.. currentmodule:: l2rpn_baselines.utils

.. _lr: ./utils.html#l2rpn_baselines.utils.TrainingParam.lr
.. _lr_decay_steps: ./utils.html#l2rpn_baselines.utils.TrainingParam.lr_decay_steps
.. _lr_decay_rate: ./utils.html#l2rpn_baselines.utils.TrainingParam.lr_decay_rate
.. _minibatch_size: ./utils.html#l2rpn_baselines.utils.TrainingParam.minibatch_size
.. _update_freq: ./utils.html#l2rpn_baselines.utils.TrainingParam.update_freq
.. _final_epsilon: ./utils.html#l2rpn_baselines.utils.TrainingParam.final_epsilon
.. _step_for_final_epsilon: ./utils.html#l2rpn_baselines.utils.TrainingParam.step_for_final_epsilon
.. _initial_epsilon: ./utils.html#l2rpn_baselines.utils.TrainingParam.initial_epsilon
.. _min_observation: ./utils.html#l2rpn_baselines.utils.TrainingParam.min_observation
.. _discount_factor: ./utils.html#l2rpn_baselines.utils.TrainingParam.discount_factor
.. _tau: ./utils.html#l2rpn_baselines.utils.TrainingParam.tau
.. _min_iter: ./utils.html#l2rpn_baselines.utils.TrainingParam.min_iter
.. _max_iter: ./utils.html#l2rpn_baselines.utils.TrainingParam.max_iter
.. _update_nb_iter: ./utils.html#l2rpn_baselines.utils.TrainingParam.update_nb_iter
.. _step_increase_nb_iter: ./utils.html#l2rpn_baselines.utils.TrainingParam.step_increase_nb_iter
.. _max_iter_fun: ./utils.html#l2rpn_baselines.utils.TrainingParam.max_iter_fun
.. _random_sample_datetime_start: ./utils.html#l2rpn_baselines.utils.TrainingParam.random_sample_datetime_start
.. _oversampling_rate: ./utils.html#l2rpn_baselines.utils.TrainingParam.oversampling_rate
.. _update_tensorboard_freq: ./utils.html#l2rpn_baselines.utils.TrainingParam.update_tensorboard_freq
.. _save_model_each: ./utils.html#l2rpn_baselines.utils.TrainingParam.save_model_each
.. _max_loss: ./utils.html#l2rpn_baselines.utils.TrainingParam.max_loss
.. _max_value_grad: ./utils.html#l2rpn_baselines.utils.TrainingParam.max_value_grad
.. _max_global_norm_grad: ./utils.html#l2rpn_baselines.utils.TrainingParam.max_global_norm_grad


utils: Some utility functions and classes
==========================================

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
The class :class:`TrainingParam` regroup a certain number of attributes with different roles. In the table below
we tried to list all the attributes and group them into attributes serving the same purpose.

=========================== ===========================================================================================
Utility                      Attribute names
=========================== ===========================================================================================
exploration                 `initial_epsilon`_, `step_for_final_epsilon`_, `final_epsilon`_
neural network learning     `minibatch_size`_, `update_freq`_, `min_observation`_
RL meta parameters          `discount_factor`_, `tau`_
limit duration of episode   `step_increase_nb_iter`_ \* , `min_iter`_, `max_iter`_, `update_nb_iter`_, `max_iter_fun`_
start an episode at random  `random_sample_datetime_start`_ \*
oversampling hard scenarios `oversampling_rate`_ \*
optimizer                   `lr`_, `lr_decay_steps`_, `lr_decay_rate`_, `max_global_norm_grad`_, `max_value_grad`_, `max_loss`_
saving / logging            `update_tensorboard_freq`_, `save_model_each`_
=========================== ===========================================================================================

\* when a "star" is present it means this parameters deactivate the whole utility. For example, setting
`step_increase_nb_iter`_ to ``None`` will deactivate the functionality "limit duration of episode"

.. _nn_param:

Focus on the architecture
--------------------------
TODO


Implementation Details
-----------------------
.. automodule:: l2rpn_baselines.utils
    :members:
    :autosummary:
