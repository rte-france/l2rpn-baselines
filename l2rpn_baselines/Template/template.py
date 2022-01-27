# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

from grid2op.Agent import DoNothingAgent


class Template(DoNothingAgent):
    """
    Note that a Baseline should always somehow inherit from :class:`grid2op.Agent.BaseAgent`.

    It serves as a template agent to explain how a baseline can be built.

    As opposed to bare grid2op Agent, baselines have 3 more methods:
    - :func:`Template.load`: to load the agent, if applicable
    - :func:`Template.save`: to save the agent, if applicable
    - :func:`Template.train`: to train the agent, if applicable

    The method :func:`Template.reset` is already present in grid2op but is emphasized here. It is called
    by a runner at the beginning of each episode with the first observation.

    The method :func:`Template.act` is also present in grid2op, of course. It the main method of the baseline,
    that receives an observation (and a reward and flag that says if an episode is over or not) an return a valid
    action.

    **NB** the "real" instance of environment on which the baseline will be evaluated will be built AFTER the creation
    of the baseline. The parameters of the real environment on which the baseline will be assessed will belong to the
    same class than the argument used by the baseline. This means that if a baseline is built with a grid2op
    environment "env", this environment will not be modified in any manner, all it's internal variable will not
    change etc. This is done to prevent cheating.

    """
    def __init__(self,
                 action_space,
                 observation_space,
                 name,
                 **kwargs):
        DoNothingAgent.__init__(self, action_space)
        self.do_nothing = self.action_space()
        self.name = name

    def act(self, observation, reward, done):
        """
        This is the main method of an Template. Given the current observation and the current reward
        (ie the reward that the environment send to the agent after the previous action has been implemented).

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.PlayableAction`
            The action chosen by the bot / controler / agent.

        """
        return self.do_nothing

    def reset(self, observation):
        """
        This method is called at the beginning of a new episode.
        It is implemented by baselines to reset their internal state if needed.

        Attributes
        -----------
        obs: :class:`grid2op.Observation.BaseObservation`
            The first observation corresponding to the initial state of the environment.
        """
        pass

    def load(self, path):
        """
        This function is used to build a baseline from a folder for example. It is recommended that this load
        function give different resulting depending on the :attr:`Template.name` of the baseline.
        For example, weights of a neural network can be saved under different names that ... depends on the
        name of the instance.

        If path is ``None`` is should be undertood as "don't load anything".

        Parameters
        ----------
        path: ``str``
            the path from which load the baseline.
        """
        pass

    def save(self, path):
        """
        This method is used to store the internal state of the baseline.

        Parameters
        ----------
        path: ``str``
            The location were to store the data of the baseline. If ``None`` it should be understood as "don't save".
            In any other cases it is more than recommended that, if "baseline" is a
            baseline, then:

            .. code-block:: python3

                path = "."  # or any other
                baseline.load(path)
                loaded_baseline = Template(...)  # built with the same parameters as "baseline"
                loaded_baseline.load(path)

            is a perfectly valid script (**eg** it will work perfectly) and that after loading, any call to
            "loaded_baseline.act" will give the results as the original "baseline.act". Or in other words, "baseline"
            and "loaded_baseline" represent the same Baseline, even though they are different instances of Baseline.
        """
        pass

    def train(self, env,
              iterations,
              save_path,
              **kwargs):
        """
        This function, if provided is used to train the baseline. Make sure to save it regularly with "baseline.save"
        for example.

        At the end of the training, it is r

        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The environment used to train your baseline.

        iterations: ``int``
            Number of training iterations used to train the baseline.

        save_path: ``str``
            Path were the final version of the baseline (**ie** after the "num_training_steps" training steps will
            be performed). It is more than recommended to save the results regurlarly during training, and to save
            the baseline at this location at the end.

        kwargs:
            Other key-words arguments used for training.

        Returns
        -------

        """
        # do the training as you want
        pass
        # don't forget to save your agent at the end!
