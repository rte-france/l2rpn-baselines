# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import pdb
from tabnanny import verbose
import warnings
import copy
import os
import grid2op
import json

from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv

from l2rpn_baselines.PPO_SB3.utils import SB3Agent

try:
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    _CAN_USE_STABLE_BASELINE = True
except ImportError:
    _CAN_USE_STABLE_BASELINE = False
    class MlpPolicy(object):
        """
        Do not use, this class is a template when stable baselines3 is not installed.
        
        It represents `from stable_baselines3.ppo import MlpPolicy`
        """
    
from l2rpn_baselines.PPO_SB3.utils import (default_obs_attr_to_keep, 
                                           default_act_attr_to_keep,
                                           remove_non_usable_attr,
                                           save_used_attribute)


def train(env,
          name="PPO_SB3",
          iterations=1,
          save_path=None,
          load_path=None,
          net_arch=None,
          logs_dir=None,
          learning_rate=3e-4,
          save_every_xxx_steps=None,
          model_policy=MlpPolicy,
          obs_attr_to_keep=copy.deepcopy(default_obs_attr_to_keep),
          act_attr_to_keep=copy.deepcopy(default_act_attr_to_keep),
          policy_kwargs=None,
          **kwargs):
    """
    This function will use stable baselines 3 to train a PPO agent on
    a grid2op environment "env".

    It will use the grid2op "gym_compat" module to convert the action space
    to a BoxActionSpace and the observation to a BoxObservationSpace.

    It is suited for the studying the impact of continuous actions:

    - on storage units
    - on dispatchable generators
    - on generators with renewable energy sources

    Parameters
    ----------
    env: :class:`grid2op.Environment`
        Then environment on which you need to train your agent.

    name: ``str```
        The name of your agent.

    iterations: ``int``
        For how many iterations (steps) do you want to train your agent. NB these are not episode, these are steps.

    save_path: ``str``
        Where do you want to save your baseline.

    load_path: ``str``
        If you want to reload your baseline, specify the path where it is located. **NB** if a baseline is reloaded
        some of the argument provided to this function will not be used.

    net_arch:
        The neural network architecture, used to create the neural network
        of the PPO (see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

    logs_dir: ``str``
        Where to store the tensorboard generated logs during the training. ``None`` if you don't want to log them.

    learning_rate: ``float``
        The learning rate, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

    save_every_xxx_steps: ``int``
        If set (by default it's None) the stable baselines3 model will be saved
        to the hard drive each `save_every_xxx_steps` steps performed in the
        environment.

    model_policy: 
        Type of neural network model trained in stable baseline. By default
        it's `MlpPolicy`

    obs_attr_to_keep: list of string
        Grid2op attribute to use to build the BoxObservationSpace. It is passed
        as the "attr_to_keep" value of the
        BoxObservation space (see
        https://grid2op.readthedocs.io/en/latest/gym.html#grid2op.gym_compat.BoxGymObsSpace)

    act_attr_to_keep: list of string
        Grid2op attribute to use to build the BoxGymActSpace. It is passed
        as the "attr_to_keep" value of the
        BoxAction space (see
        https://grid2op.readthedocs.io/en/latest/gym.html#grid2op.gym_compat.BoxGymActSpace)

    verbose: ``bool``
        If you want something to be printed on the terminal (a better logging strategy will be put at some point)

    policy_kwargs: ``dict``
        extra parameters passed to the PPO "policy_kwargs" key word arguments
        (defaults to ``None``)
        
    kwargs:
        extra parameters passed to the PPO from stable baselines 3

    Returns
    -------

    baseline: 
        The trained baseline as a stable baselines PPO element.


    .. _Example-ppo_stable_baseline:

    Examples
    ---------

    Here is an example on how to train a ppo_stablebaseline .

    First define a python script, for example

    .. code-block:: python

        import re
        import grid2op
        from grid2op.Reward import LinesCapacityReward  # or any other rewards
        from grid2op.Chronics import MultifolderWithCache  # highly recommended
        from lightsim2grid import LightSimBackend  # highly recommended for training !
        from l2rpn_baselines.PPO_SB3 import train

        env_name = "l2rpn_case14_sandbox"
        env = grid2op.make(env_name,
                           reward_class=LinesCapacityReward,
                           backend=LightSimBackend(),
                           chronics_class=MultifolderWithCache)

        env.chronics_handler.real_data.set_filter(lambda x: re.match(".*00$", x) is not None)
        env.chronics_handler.real_data.reset()
        # see https://grid2op.readthedocs.io/en/latest/environment.html#optimize-the-data-pipeline
        # for more information !

        try:
            trained_agent = train(
                  env,
                  iterations=10_000,  # any number of iterations you want
                  logs_dir="./logs",  # where the tensorboard logs will be put
                  save_path="./saved_model",  # where the NN weights will be saved
                  name="test",  # name of the baseline
                  net_arch=[100, 100, 100],  # architecture of the NN
                  save_every_xxx_steps=2000,  # save the NN every 2k steps
                  )
        finally:
            env.close()

    """
    if not _CAN_USE_STABLE_BASELINE:
        raise ImportError("Cannot use this function as stable baselines3 is not installed")
    
    # keep only usable attributes (if default is used)
    act_attr_to_keep = remove_non_usable_attr(env, act_attr_to_keep)
    
    # save the attributes kept
    if save_path is not None:
        my_path = os.path.join(save_path, name)
    save_used_attribute(save_path, name, obs_attr_to_keep, act_attr_to_keep)

    # define the gym environment from the grid2op env
    env_gym = GymEnv(env)
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(env.observation_space,
                                               attr_to_keep=obs_attr_to_keep)
    env_gym.action_space.close()
    env_gym.action_space = BoxGymActSpace(env.action_space,
                                          attr_to_keep=act_attr_to_keep)


    # Save a checkpoint every 1000 steps
    checkpoint_callback = None
    if save_every_xxx_steps is not None:
        if save_path is None:
            warnings.warn("save_every_xxx_steps is set, but no path are "
                          "set to save the model (save_path is None). No model "
                          "will be saved.")
        else:
            checkpoint_callback = CheckpointCallback(save_freq=save_every_xxx_steps,
                                                     save_path=my_path,
                                                     name_prefix=name)

    # define the policy
    if load_path is None:
        if policy_kwargs is None:
            policy_kwargs = {}
        if net_arch is not None:
            policy_kwargs["net_arch"] = net_arch
        if logs_dir is not None:
            if not os.path.exists(logs_dir):
                os.mkdir(logs_dir)
                
        nn_kwargs = {
            "policy": model_policy,
            "env": env_gym,
            "verbose": verbose,
            "learning_rate": learning_rate,
            "tensorboard_log": logs_dir,
            "policy_kwargs": policy_kwargs,
            **kwargs
        }
        agent = SB3Agent(env.action_space,
                         env_gym.action_space,
                         env_gym.observation_space,
                         nn_kwargs=nn_kwargs,
        )
    else:        
        agent = SB3Agent(env.action_space,
                         env_gym.action_space,
                         env_gym.observation_space,
                         nn_path=os.path.join(load_path, name)
        )

    # train it
    agent.nn_model.learn(total_timesteps=iterations,
                         callback=checkpoint_callback)

    # save it
    if save_path is not None:
        agent.nn_model.save(os.path.join(my_path, name))

    env_gym.close()
    return agent  # TODO

if __name__ == "__main__":

    import re
    import grid2op
    from grid2op.Reward import LinesCapacityReward  # or any other rewards
    from lightsim2grid import LightSimBackend  # highly recommended !
    from grid2op.Chronics import MultifolderWithCache  # highly recommended for training

    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name,
                       reward_class=LinesCapacityReward,
                       backend=LightSimBackend(),
                       chronics_class=MultifolderWithCache)

    env.chronics_handler.real_data.set_filter(lambda x: re.match(".*0$", x) is not None)
    env.chronics_handler.real_data.reset()
    # see https://grid2op.readthedocs.io/en/latest/environment.html#optimize-the-data-pipeline
    # for more information !
    train(env,
          iterations=1_000,
          logs_dir="./logs",
          save_path="./saved_model", 
          name="test4",
          net_arch=[200, 200, 200],
          save_every_xxx_steps=2000,
          )
