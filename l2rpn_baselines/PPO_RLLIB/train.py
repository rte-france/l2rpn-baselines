# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import grid2op
import copy
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace

from l2rpn_baselines.PPO_RLLIB.env_rllib import Env_RLLIB
from l2rpn_baselines.PPO_SB3 import (default_obs_attr_to_keep, 
                                     default_act_attr_to_keep,
                                     save_used_attribute,
                                     remove_non_usable_attr
                                    )
from l2rpn_baselines.PPO_RLLIB.rllibagent import RLLIBAgent

try:
    import ray
    from ray.rllib.agents import ppo
    from ray.tune.logger import pretty_print
    _CAN_USE_RLLIB = True
except ImportError as exc_:
    _CAN_USE_RLLIB = False


def train(env,
          name="ppo_rllib",
          iterations=1,
          save_path=None,
          load_path=None,
          net_arch=None,
          learning_rate=3e-4,
          verbose=False,
          save_every_xxx_steps=None,
          obs_attr_to_keep=copy.deepcopy(default_obs_attr_to_keep),
          act_attr_to_keep=copy.deepcopy(default_act_attr_to_keep),
          env_kwargs=None,
          **kwargs):
    """
    This function will use the rllib to train a PPO agent on
    a grid2op environment "env".

    It will use the grid2op "gym_compat" module to convert the action space
    to a BoxActionSpace and the observation to a BoxObservationSpace.

    It is suited for the studying the impact of continuous actions:

    - on storage units
    - on dispatchable generators
    - on generators with renewable energy sources

    .. warning::
        The environment used by RLLIB is copied and remade. This class does
        not work if you over specialize the environment !
        For example, opponent is not taken into account (yet), nor the chronics class
        etc.
        
        If you want such level of control, please use the `env_kwargs` parameters !
        
    Parameters
    ----------
    env: :class:`grid2op.Environment`
        Then environment on which you need to train your agent.
        
        Only the name of the environment, and its backend is used. The rest will
        be created by rllib.

    name: ``str```
        The name of your agent.

    iterations: ``int``
        For how many iterations do you want to train the model.
        These are **NOT** steps, but ray internal number of iterations.
        For some experiments we performed,  

    save_path: ``str``
        Where do you want to save your baseline.

    load_path: ``str``
        If you want to reload your baseline, specify the path where it is located. **NB** if a baseline is reloaded
        some of the argument provided to this function will not be used.

    net_arch:
        The neural network architecture, used to create the neural network
        of the PPO (see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

    learning_rate: ``float``
        The learning rate, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

    save_every_xxx_steps: ``int``
        If set (by default it's None) the stable baselines3 model will be saved
        to the hard drive each `save_every_xxx_steps` steps performed in the
        environment.

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

    env_kwargs: Optional[dict]
        Extra key word arguments passed to the building of the 
        grid2op environment.
        
    kwargs:
        extra parameters passed to the trainer from rllib

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
        import ray
        from grid2op.Reward import LinesCapacityReward  # or any other rewards
        from grid2op.Chronics import MultifolderWithCache  # highly recommended
        from lightsim2grid import LightSimBackend  # highly recommended for training !
            
        env_name = "l2rpn_case14_sandbox"
        env = grid2op.make(env_name,
                           backend=LightSimBackend())

        ray.init()  # if needed (you might have it already working somewhere)
        try:
            train(env,
                  iterations=10,  # any number of iterations you want
                  save_path="./saved_model",  # where the NN weights will be saved
                  name="test",  # name of the baseline
                  net_arch=[100, 100, 100],  # architecture of the NN
                  save_every_xxx_steps=2,  # save the NN every 2 training steps
                  env_kwargs={"reward_class": LinesCapacityReward,
                              "chronics_class": MultifolderWithCache,  # highly recommended
                              "data_feeding_kwargs": {
                                  'filter_func': lambda x: re.match(".*00$", x) is not None  #use one over 100 chronics to train (for speed)
                                  }
                  },
                  verbose=True
                  )
        finally:
            env.close()
            ray.shutdown()  # if needed (you might have it already working somewhere)
    
    """
    import jsonpickle
    if not _CAN_USE_RLLIB:
        raise ImportError("RLLIB is not installed on your machine")
    
    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        path_expe = os.path.join(save_path, name)
        if not os.path.exists(path_expe):
            os.mkdir(path_expe)
            
    # save the attributes kept
    act_attr_to_keep = remove_non_usable_attr(env, act_attr_to_keep)
    need_saving_final = save_used_attribute(save_path, name, obs_attr_to_keep, act_attr_to_keep)
    need_saving = need_saving_final and save_every_xxx_steps is not None
    
    if env_kwargs is None:
        env_kwargs = {}
    
    env_params = env.get_kwargs()
    env_config = {"env_name": env.env_name,
                  "backend_class": env_params["_raw_backend_class"],
                  "obs_attr_to_keep": obs_attr_to_keep,
                  "act_attr_to_keep": act_attr_to_keep, 
                  **env_kwargs}
    
    model_dict = {}
    if net_arch is not None:
        model_dict["fcnet_hiddens"] = net_arch
    env_config_ppo = {
        # config to pass to env class
        "env_config": env_config,
        #neural network config
        "lr": learning_rate,
        "model": model_dict,
        **kwargs
    }
        
    # store it
    encoded = jsonpickle.encode(env_config_ppo)
    with open(os.path.join(path_expe, "env_config.json"), "w", encoding="utf-8") as f:
        f.write(encoded)
    
    # define the gym environment from the grid2op env
    env_gym = GymEnv(env)
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(env.observation_space,
                                               attr_to_keep=obs_attr_to_keep)
    env_gym.action_space.close()
    env_gym.action_space = BoxGymActSpace(env.action_space,
                                          attr_to_keep=act_attr_to_keep)
    # then define a "trainer"
    agent = RLLIBAgent(g2op_action_space=env.action_space,
                       gym_act_space=env_gym.action_space,
                       gym_obs_space=env_gym.observation_space,
                       nn_config=env_config_ppo,
                       nn_path=load_path)
    
    for step in range(iterations):
        # Perform one iteration of training the policy with PPO
        result = agent.nn_model.train()
        if verbose:
            print(pretty_print(result))

        if need_saving and step % save_every_xxx_steps == 0:
            agent.nn_model.save(checkpoint_dir=path_expe)
            
    if need_saving_final:
        agent.nn_model.save(checkpoint_dir=path_expe)
        
    return agent
    
    
if __name__ == "__main__":
    import re
    import grid2op
    from grid2op.Reward import LinesCapacityReward  # or any other rewards
    from grid2op.Chronics import MultifolderWithCache  # highly recommended
    from lightsim2grid import LightSimBackend  # highly recommended for training !
    import ray
    
    
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name,
                       backend=LightSimBackend())
    
    ray.init()
    try:
        train(env,
              iterations=10,  # any number of iterations you want
              save_path="./saved_model",  # where the NN weights will be saved
              name="test3",  # name of the baseline
              net_arch=[100, 100, 100],  # architecture of the NN
              save_every_xxx_steps=2,  # save the NN every 2 training steps
              env_kwargs={"reward_class": LinesCapacityReward,
                          "chronics_class": MultifolderWithCache,  # highly recommended
                          "data_feeding_kwargs": {
                              'filter_func': lambda x: re.match(".*00$", x) is not None  #use one over 100 chronics to train (for speed)
                              }
              },
              verbose=True
              )
    finally:
        env.close()
        ray.shutdown()
