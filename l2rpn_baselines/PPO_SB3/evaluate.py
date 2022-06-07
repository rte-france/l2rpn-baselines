# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import json
from grid2op.Runner import Runner
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv
from l2rpn_baselines.utils.gymenv_custom import GymEnvWithHeuristics

from l2rpn_baselines.utils.save_log_gif import save_log_gif
from l2rpn_baselines.PPO_SB3.utils import SB3Agent


def evaluate(env,
             load_path=".",
             name="PPO_SB3",
             logs_path=None,
             nb_episode=1,
             nb_process=1,
             max_steps=-1,
             verbose=False,
             save_gif=False,
             gymenv_class=GymEnv,
             gymenv_kwargs=None,
             obs_space_kwargs=None,
             act_space_kwargs=None,
             iter_num=None,
             **kwargs):
    """
    This function will use stable baselines 3 to evaluate a previously trained
    PPO agent (with stable baselines 3) on
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

    load_path: ``str``
        If you want to reload your baseline, specify the path where it is located. **NB** if a baseline is reloaded
        some of the argument provided to this function will not be used.

    logs_dir: ``str``
        Where to store the tensorboard generated logs during the training. ``None`` if you don't want to log them.
    
    nb_episode: ``str``
        How many episodes to run during the assessment of the performances

    nb_process: ``int``
        On how many process the assessment will be made. (setting this > 1 can lead to some speed ups but can be
        unstable on some plaform)

    max_steps: ``int``
        How many steps at maximum your agent will be assessed

    verbose: ``bool``
        Currently un used

    save_gif: ``bool``
        Whether or not you want to save, as a gif, the performance of your agent. It might cause memory issues (might
        take a lot of ram) and drastically increase computation time.

    gymenv_class: 
        The class to use as a gym environment. By default `GymEnv` (from module grid2op.gym_compat)
    
    gymenv_kwargs: ``dict``
        Extra key words arguments to build the gym environment.
    
    iter_num:
        Which training iteration do you want to restore (by default: None means 
        "the last one")
        
    kwargs:
        extra parameters passed to the PPO from stable baselines 3

    Returns
    -------

    baseline: 
        The loaded baseline as a stable baselines PPO element.

    Examples
    ---------

    Here is an example on how to evaluate an  PPO agent (previously trained
    with stable baselines3):

    .. code-block:: python

        import grid2op
        from grid2op.Reward import LinesCapacityReward  # or any other rewards
        from lightsim2grid import LightSimBackend  # highly recommended !
        from l2rpn_baselines.PPO_SB3 import evaluate

        nb_episode = 7
        nb_process = 1
        verbose = True

        env_name = "l2rpn_case14_sandbox"
        env = grid2op.make(env_name,
                           reward_class=LinesCapacityReward,
                           backend=LightSimBackend()
                           )

        try:
            evaluate(env,
                    nb_episode=nb_episode,
                    load_path="./saved_model",  # should be the same as what has been called in the train function !
                    name="test",  # should be the same as what has been called in the train function !
                    nb_process=1,
                    verbose=verbose,
                    )

            # you can also compare your agent with the do nothing agent relatively
            # easily
            runner_params = env.get_params_for_runner()
            runner = Runner(**runner_params)

            res = runner.run(nb_episode=nb_episode,
                            nb_process=nb_process
                            )

            # Print summary
            if verbose:
                print("Evaluation summary for DN:")
                for _, chron_name, cum_reward, nb_time_step, max_ts in res:
                    msg_tmp = "chronics at: {}".format(chron_name)
                    msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
                    msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
                    print(msg_tmp)
        finally:
            env.close()

    """
    
    if obs_space_kwargs is None:
        obs_space_kwargs = {}
    if act_space_kwargs is None:
        act_space_kwargs = {}

    # load the attributes kept
    my_path = os.path.join(load_path, name)
    if not os.path.exists(load_path):
        os.mkdir(load_path)
    if not os.path.exists(my_path):
        os.mkdir(my_path)
        
    with open(os.path.join(my_path, "obs_attr_to_keep.json"), encoding="utf-8", mode="r") as f:
        obs_attr_to_keep = json.load(fp=f)
    with open(os.path.join(my_path, "act_attr_to_keep.json"), encoding="utf-8", mode="r") as f:
        act_attr_to_keep = json.load(fp=f)

    # create the action and observation space
    gym_observation_space =  BoxGymObsSpace(env.observation_space,
                                            attr_to_keep=obs_attr_to_keep,
                                            **obs_space_kwargs)
    gym_action_space = BoxGymActSpace(env.action_space,
                                      attr_to_keep=act_attr_to_keep,
                                      **act_space_kwargs)
    
    if os.path.exists(os.path.join(load_path, ".normalize_act")):
        for attr_nm in act_attr_to_keep:
            if (("multiply" in act_space_kwargs and attr_nm in act_space_kwargs["multiply"]) or 
                ("add" in act_space_kwargs and attr_nm in act_space_kwargs["add"]) 
               ):
                continue
            gym_action_space.normalize_attr(attr_nm)

    if os.path.exists(os.path.join(load_path, ".normalize_obs")):
        for attr_nm in obs_attr_to_keep:
            if (("divide" in obs_space_kwargs and attr_nm in obs_space_kwargs["divide"]) or 
                ("subtract" in obs_space_kwargs and attr_nm in obs_space_kwargs["subtract"]) 
               ):
                continue
            gym_observation_space.normalize_attr(attr_nm)
    
    gymenv = None
    if gymenv_class is not None and issubclass(gymenv_class, GymEnvWithHeuristics):
        if gymenv_kwargs is None:
            gymenv_kwargs = {}
        gymenv = gymenv_class(env, **gymenv_kwargs)
        
        gymenv.action_space.close()
        gymenv.action_space = gym_action_space
        
        gymenv.observation_space.close()
        gymenv.observation_space = gym_observation_space
        
    # create a grid2gop agent based on that (this will reload the save weights)
    full_path = os.path.join(load_path, name)
    grid2op_agent = SB3Agent(env.action_space,
                             gym_action_space,
                             gym_observation_space,
                             nn_path=os.path.join(full_path, name),
                             gymenv=gymenv,
                             iter_num=iter_num,
                             )

    if nb_episode == 0:
        return grid2op_agent, []
    
    # Build runner
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=grid2op_agent)
    
    # Run the agent on the scenarios
    if logs_path is not None:
        os.makedirs(logs_path, exist_ok=True)

    res = runner.run(path_save=logs_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     pbar=verbose,
                     **kwargs)

    # Print summary
    if verbose:
        print("Evaluation summary:")
        for _, chron_name, cum_reward, nb_time_step, max_ts in res:
            msg_tmp = "chronics at: {}".format(chron_name)
            msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
            msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)

    if save_gif:
        if verbose:
            print("Saving the gif of the episodes")
        save_log_gif(logs_path, res)
    return grid2op_agent, res


if __name__ == "__main__":

    import grid2op
    from grid2op.Reward import LinesCapacityReward  # or any other rewards
    from lightsim2grid import LightSimBackend  # highly recommended !

    nb_episode = 7
    nb_process = 1
    verbose = True

    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name,
                        reward_class=LinesCapacityReward,
                        backend=LightSimBackend()
                        )

    try:
        trained_agent, res_eval = evaluate(
                    env,
                    nb_episode=nb_episode,
                    load_path="./saved_model", 
                    name="test4",
                    nb_process=1,
                    verbose=verbose,
                    )

        # you can also compare your agent with the do nothing agent relatively
        # easily
        runner_params = env.get_params_for_runner()
        runner = Runner(**runner_params)

        res = runner.run(nb_episode=nb_episode,
                        nb_process=nb_process
                        )

        # Print summary
        if verbose:
            print("Evaluation summary for DN:")
            for _, chron_name, cum_reward, nb_time_step, max_ts in res:
                msg_tmp = "chronics at: {}".format(chron_name)
                msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
                msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
                print(msg_tmp)
    finally:
        env.close()
