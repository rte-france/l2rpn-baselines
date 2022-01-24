# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import pdb
import warnings
import copy
import os
import grid2op
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv

import json

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

_default_obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                             "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                             "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status",
                             "storage_power", "storage_charge"]

_default_act_attr_to_keep = ["redispatch", "curtail", "set_storage"]


def train(env,
          name="ppo_stable_baselines",
          iterations=1,
          save_path=None,
          load_path=None,
          net_arch=None,
          logs_dir=None,
          learning_rate=3e-4,
          save_every_xxx_steps=None,
          model_policy=MlpPolicy,
          obs_attr_to_keep=copy.deepcopy(_default_obs_attr_to_keep),
          act_attr_to_keep=copy.deepcopy(_default_act_attr_to_keep),
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
        from grid2op.Reward import LinesCapacityReward  # or any other rewards
        from lightsim2grid import LightSimBackend  # highly recommended !
        from grid2op.Chronics import MultifolderWithCache  # highly recommended

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
            train(env,
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
    if act_attr_to_keep == _default_act_attr_to_keep:
        # by default, i remove all the attributes that are not supported by the action type
        # i do not do that if the user specified specific attributes to keep. This is his responsibility in
        # in this case
        modif_attr = []
        for el in act_attr_to_keep:
            if env.action_space.supports_type(el):
                modif_attr.append(el)
            else:
                warnings.warn(f"attribute {el} cannot be processed by the allowed "
                              "action type. It has been removed from the "
                              "gym space as well.")
        act_attr_to_keep = modif_attr

    if save_path is not None:
        # save the attributes kept
        my_path = os.path.join(save_path, name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(my_path):
            os.mkdir(my_path)

        with open(os.path.join(my_path, "obs_attr_to_keep.json"), encoding="utf-8", mode="w") as f:
            json.dump(fp=f, obj=obs_attr_to_keep)
        with open(os.path.join(my_path, "act_attr_to_keep.json"), encoding="utf-8", mode="w") as f:
            json.dump(fp=f, obj=act_attr_to_keep)

    # define the gym environment from the grid2op env
    env_gym = GymEnv(env)
    env_gym.observation_space.close()
    env_gym.observation_space =  BoxGymObsSpace(env.observation_space,
                                                attr_to_keep=obs_attr_to_keep)
    env_gym.action_space.close()
    env_gym.action_space = BoxGymActSpace(env.action_space, attr_to_keep=act_attr_to_keep)


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
        policy_kwargs = {}
        if net_arch is not None:
            policy_kwargs["net_arch"] = net_arch
        if logs_dir is not None:
            if not os.path.exists(logs_dir):
                os.mkdir(logs_dir)
        model = PPO(model_policy,
                    env_gym,
                    verbose=1,
                    learning_rate=learning_rate,
                    tensorboard_log=os.path.join(logs_dir, name),
                    policy_kwargs=policy_kwargs,
                    **kwargs)
    else:
        # TODO !
        model = PPO.load(os.path.join(load_path, name))

    # train it
    model.learn(total_timesteps=iterations,
                callback=checkpoint_callback)

    # save it
    if save_path is not None:
        model.save(os.path.join(my_path, name))

    env_gym.close()

if __name__ == "__main__":

    import re
    from grid2op.Reward import LinesCapacityReward  # or any other rewards
    from lightsim2grid import LightSimBackend  # highly recommended !
    from grid2op.Chronics import MultifolderWithCache  # highly recommended

    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name,
                       reward_class=LinesCapacityReward,
                       backend=LightSimBackend(),
                       chronics_class=MultifolderWithCache)

    env.chronics_handler.real_data.set_filter(lambda x: re.match(".*00$", x) is not None)
    env.chronics_handler.real_data.reset()
    # see https://grid2op.readthedocs.io/en/latest/environment.html#optimize-the-data-pipeline
    # for more information !

    train(env,
          iterations=10_000,
          logs_dir="./logs",
          save_path="./saved_model", 
          name="test",
          net_arch=[100, 100, 100],
          save_every_xxx_steps=2000,
          )


    # from grid2op.Action import CompleteAction
    # from grid2op.Reward import LinesCapacityReward
    # from lightsim2grid import LightSimBackend
    # from grid2op.Chronics import MultifolderWithCache

    # env = grid2op.make("educ_case14_storage",
    #                    test=True,
    #                    action_class=CompleteAction,
    #                    reward_class=LinesCapacityReward,
    #                    backend=LightSimBackend(),
    #                    chronics_class=MultifolderWithCache)

    # env.chronics_handler.real_data.set_filter(lambda x: True)
    # env.chronics_handler.real_data.reset()

    # train(env,
    #       iterations=10_000,
    #       logs_dir="./logs",
    #       save_path="./saved_model", 
    #       name="test4",
    #       net_arch=[100, 100, 100],
    #       save_every_xxx_steps=2000,
    #       )
