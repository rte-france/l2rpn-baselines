Change Log
===========
[TODO]
--------
- stack multiple states in `utils/DeepQAgent`

[0.5.0] - 2020-08-??
--------------------
- [FIXED] the counting of the action types frequency in tensorboard (for some baselines)
- [FIXED] a broken Replay buffer `utils.ReplayBuffer` (used in some baselines)
- [FIXED] a bug in using multiple environments for some baselines
- [FIXED] wrong q value update for some baselines
- [IMPROVED] descriptions and computation of the tensorboard information (for some baselines)
- [ADDED] better serializing as json of the `utils.NNParam` class
- [ADDED] the LeapNetEncoded baselines that uses a leap neural network (leap net) to create an
  embedding of the state of the powergrid.

[0.4.4] - 2020-07-07
--------------------
- [FIXED] now the baselines can fully support the grid2op MultiMix environment.

[0.4.3] - 2020-07-06
---------------------
- [FIXED] a bug the prevented to reload the baselines when the python version changed (for example
  if the baseline was trained with python 3.8 then you would not be able to load it to load it
  with python 3.6. This is a limitation of the "marshall" library used internally by keras. We
  found a way to fix it.

[0.4.2] - 2020-06-29
-----------------------
- [FIXED] a bug in the TrainingParam class (wrong realoading)
- [UPDATED] backward compatibility with 0.9.1.post1
- [ADDED] easier UI to load the baselines SAC, DeepQSimple, DuelQSimple and DuelQLeapNet

[0.4.1] - 2020-06-16
-----------------------
- [FIXED] `Issue 14 <https://github.com/rte-france/l2rpn-baselines/issues/14>`_ clearer interface and get rid
  of the "nb_env" in some baselines constructor. A helper function
  `make_multi_env` has also been created to help the creation of the appropariate multi environment.
- [FIXED] `Issue 13 <https://github.com/rte-france/l2rpn-baselines/issues/13>`_ the name have been properly updated
- [FIXED] `Issue 12 <https://github.com/rte-france/l2rpn-baselines/issues/12>`_ the appropriate documentation for the
  SAC baselines and all the kind
- [FIXED] `Issue 9 <https://github.com/rte-france/l2rpn-baselines/issues/9>`_ no more hard coded global variables for
  most of the baselines.

[0.4.0] - 2020-06-xx
--------------------
- [ADDED] convenience way to modify the architecture of the neural networks
- [ADDED] serialize / de serialize TrainingParams and NNArchi as json
- [ADDED] possibility get the proper observation from their attribute name
- [ADDED] action space is now serializable / de serializable
- [ADDED] documentation
- [ADDED] some tests
- [ADDED] the AsynchronousActorCritic baseline, that won the 2nd place to the first edition of l2rpn in 2019.

[0.3.0] - 2020-05-13
--------------------
- [ADDED] DeepQSimple, SAC and DuelQSimple baselines
- [ADDED] utilitary code to create more easily agents
- [ADDED] Multi processing training of agents using `grid2op.Environment.MultiEnvironment`
- [ADDED] leap net as a baseline
- [UDPATED] grid2op version `0.8.2`

[0.1.1] - 2020-04-23
--------------------
- [UPDATED] function `zip_for_codalab` now returns the path where the zip has been made **NB** this function
  might be moved in another repository soon.
- [UPDATED] The descriptions in the setup script.

[0.1.0] - 2020-04-23
--------------------
- [ADDED] initialization of the repository with some baselines, how to contribute etc.
