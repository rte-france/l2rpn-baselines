Change Log
===========

[TODO]
--------
- in the "examples" folder, make some examples for possible "submissions"
  usable in the competition for PPO_SB3 and PPO_RLLIB
- add a vectorized environment for PPO in stable baselines for example
  (ie map a MultiEnvironment into the proper stuff)
- code a baseline example using mazerl
- code a baseline using deepmind acme
- code a baseline with a GNN somewhere
- show an example on how to use some "heuristic" in training / evaluation of trained agents
- show an example of model based RL agent
- train sowhere a working baseline (that does better than do nothing)
- show an example of a baseline that uses a GNN

[0.7.0] - 2023-07-13
------------------------
- [ADDED] the "topo oracle agent" (contrib)
- [ADDED] the "curriculumagent" (contrib)

[0.6.0.post1] - 2022-07-05
---------------------------
- [FIXED] issue with the `PPO_SB3` agent when using a runner, particularly when no "heuristic" are
  used at inference time.

[0.6.0] - 2022-06-07
--------------------
- [BREAKING] name of the file inside the submodule are now lowercase (PEP 8 compliance)
  Use `from l2rpn_baselines.[BASELINENAME] import [BASELINENAME]` by replacing 
  `[BASELINENAME]` with ... the baseline name (*eg* `from l2rpn_baselines.DoNothing import DoNothing`)
- [FIXED] clean the documentation
- [FIXED] some bugs (especially in the type of actions) for some agents
- [ADDED] a code example to use stable baselines 3 (see `l2rpn_baselines.PPO_SB3`)
- [ADDED] a code example to use RLLIB (see `l2rpn_baselines.PPO_RLLIB`)
- [ADDED] an optimizer (see `l2rpn_baselines.OptimCVXPY`)
- [ADDED] some issue templates
- [ADDED] some examples in the "examples" folder

[0.5.1] - 2021-04-09
---------------------
- [FIXED] issue with grid2op version >= 1.2.3 for some baselines
- [FIXED] `Issue 26 <https://github.com/rte-france/l2rpn-baselines/issues/26>`_ : package can be installed even
  if the requirement for some baselines is not met.
- [UPDATED] `Kaist` baselines
- [ADDED] The expert agent

[0.5.0] - 2020-08-18
--------------------
- [BREAKING] remove the SAC baseline that was not correct. For backward compatibility, its code
  can still be accessed with SACOld
- [FIXED] the counting of the action types frequency in tensorboard (for some baselines)
- [FIXED] a broken Replay buffer `utils.ReplayBuffer` (used in some baselines)
- [FIXED] a bug in using multiple environments for some baselines
- [FIXED] wrong q value update for some baselines
- [IMPROVED] descriptions and computation of the tensorboard information (for some baselines)
- [IMPROVED] performance optimization for training and usage of some baselines
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
