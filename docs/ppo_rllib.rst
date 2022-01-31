.. currentmodule:: l2rpn_baselines.ppo_stablebaselines

PPO: with ray/rllib
===========================================================

Description
-----------
This "baseline" aims at providing a code example on how to use an agent
from the ray/rllib repository (see https://docs.ray.io/en/master/rllib/)
with grid2op.

It also serve a second goal, to show how to train a PPO agent to perform
continuous actions on the powergrid (*eg* adjusting the generator value, either
by applying `redispatching` kind of action for controlable generators or 
by with `curtailment` on generator using new renewable energy sources - solar and wind
or even to control the state of the storage units.)

It is pretty much the same as the :class:`l2rpn_baselines.PPO_SB3` but uses
rllib instead of stable Baselines3.

Exported class
--------------
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.PPO_RLLIB import train, evaluate, PPO_RLLIB

Used a trained agent
++++++++++++++++++++++

You first need to train it:

.. code-block:: python

    import re
    import grid2op
    from grid2op.Reward import LinesCapacityReward  # or any other rewards
    from grid2op.Chronics import MultifolderWithCache  # highly recommended
    from lightsim2grid import LightSimBackend  # highly recommended for training !
    import ray
    from l2rpn_baselines.PPO_RLLIB import train
    
    
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name,
                       backend=LightSimBackend())
    
    ray.init()
    try:
        trained_aget = train(
              env,
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
        ray.shutdown()

Then you can load it:

.. code-block:: python

    import grid2op
    from grid2op.Reward import LinesCapacityReward  # or any other rewards
    from lightsim2grid import LightSimBackend  # highly recommended !
    from l2rpn_baselines.PPO_RLLIB import evaluate

    nb_episode = 7
    nb_process = 1
    verbose = True

    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name,
                        reward_class=LinesCapacityReward,
                        backend=LightSimBackend()
                        )

    try:
        trained_agent = evaluate(
                 env,
                 nb_episode=nb_episode,
                 load_path="./saved_model",  # should be the same as what has been called in the train function !
                 name="test3",  # should be the same as what has been called in the train function !
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


Create an agent from scratch
++++++++++++++++++++++++++++++

For example, to create an agent **from scratch**, with some parameters:

.. code-block:: python

    import grid2op
    from grid2op.gym_compat import BoxGymObsSpace, BoxGymActSpace
    from lightsim2grid import LightSimBackend
    from l2rpn_baselines.PPO_RLLIB import PPO_RLLIB

    env_name = "l2rpn_case14_sandbox"  # or any other name
    obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                        "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                        "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status",
                        "storage_power", "storage_charge"]
    act_attr_to_keep = ["redispatch"]
    
    # create the grid2op environment
    env = grid2op.make(env_name, backend=LightSimBackend())
    
    # define the action space and observation space that your agent
    # will be able to use
    gym_observation_space = BoxGymObsSpace(env.observation_space, attr_to_keep=obs_attr_to_keep)
    gym_action_space = BoxGymActSpace(env.action_space, attr_to_keep=act_attr_to_keep)

    # define the configuration for the environment
    env_config = {"env_name": env.env_name,
                  "backend_class": LightSimBackend,
                  "obs_attr_to_keep": obs_attr_to_keep,
                  "act_attr_to_keep": act_attr_to_keep, 
                    # other type of parameters used in the "grid2op.make"
                    # function eg:
                    # "param": ...
                    # "reward_class": ...
                    # "other_reward": ...
                    # "difficulty": ...
                    }

    # now define the configuration for the PPOTrainer
    env_config_ppo = {
        # config to pass to env class
        "env_config": env_config,
        #neural network config
        "lr": 1e-4, # learning_rate
        "model": {
            "fcnet_hiddens": [100, 100, 100],  # neural net architecture
        },
        # other keyword arguments
    }
    
    # create a grid2gop agent based on that (this will reload the save weights)
    grid2op_agent = RLLIBAgent(env.action_space,
                                gym_action_space,
                                gym_observation_space,
                                nn_config=env_config_ppo,
                                nn_path=None  # don't load it from anywhere
                                )
    
    # use it
    obs = env.reset()
    reward = env.reward_range[0]
    done = False
    grid2op_act = grid2op_agent.act(obs, reward, done)
    obs, reward, done, info = env.step(grid2op_act)
    

.. note::
    The agent above is NOT trained. So it will basically output "random" actions.

    You should probably train it before hand (see the `train` function)


Detailed documentation
++++++++++++++++++++++++

.. automodule:: l2rpn_baselines.PPO_SB3
    :members:
    :autosummary:
