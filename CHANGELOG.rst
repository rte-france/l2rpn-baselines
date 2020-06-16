Change Log
===========
[TODO]
--------
- stack multiple states in `utils/DeepQAgent`

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
