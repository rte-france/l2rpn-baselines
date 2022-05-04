.. currentmodule:: l2rpn_baselines.PPO_SB3

PPO: with stable-baselines3
===========================================================

Description
-----------
This "baseline" aims at providing a code example on how to use an agent
from the Sable Baselines3 repository (see https://stable-baselines3.readthedocs.io/en/master/)
with grid2op.

It also serve a second goal, to show how to train a PPO agent to perform
continuous actions on the powergrid (*eg* adjusting the generator value, either
by applying `redispatching` kind of action for controlable generators or 
by with `curtailment` on generator using new renewable energy sources - solar and wind
or even to control the state of the storage units.)


Exported class
--------------
You can use this class with:

.. code-block:: python

    from l2rpn_baselines.PPO_SB3 import train, evaluate, PPO_SB3


Used a trained agent
++++++++++++++++++++++

You first need to train it:

.. code-block:: python

    import re
    import grid2op
    from grid2op.Reward import LinesCapacityReward  # or any other rewards
    from lightsim2grid import LightSimBackend  # highly recommended !
    from grid2op.Chronics import MultifolderWithCache  # highly recommended for training
    from l2rpn_baselines.PPO_SB3 import train

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
          name="test",
          net_arch=[200, 200, 200],
          save_every_xxx_steps=2000,
          )

Then you can load it:

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


Create an agent from scratch
++++++++++++++++++++++++++++++

For example, to create an agent **from scratch**, with some parameters:

.. code-block:: python

    import grid2op
    from grid2op.gym_compat import BoxGymObsSpace, BoxGymActSpace
    from lightsim2grid import LightSimBackend
    from stable_baselines3.ppo import MlpPolicy
    from l2rpn_baselines.PPO_SB3 import PPO_SB3
        
    env_name = "l2rpn_case14_sandbox"  # or any other name
    
    # customize the observation / action you want to keep
    obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                        "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                        "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status",
                        "storage_power", "storage_charge"]
    act_attr_to_keep = ["redispatch"]
    
    # create the grid2op environment
    env = grid2op.make(env_name, backend=LightSimBackend())
    
    # define the action space and observation space that your agent
    # will be able to use
    env_gym = GymEnv(env)
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(env.observation_space,
                                               attr_to_keep=obs_attr_to_keep)
    env_gym.action_space.close()
    env_gym.action_space = BoxGymActSpace(env.action_space,
                                          attr_to_keep=act_attr_to_keep)
    
    # create the key word arguments used for the NN
    nn_kwargs = {
        "policy": MlpPolicy,
        "env": env_gym,
        "verbose": 0,
        "learning_rate": 1e-3,
        "tensorboard_log": ...,
        "policy_kwargs": {
            "net_arch": [100, 100, 100]
        }
    }
    
    # create a grid2gop agent based on that (this will reload the save weights)
    grid2op_agent = PPO_SB3(env.action_space,
                            env_gym.action_space,
                            env_gym.observation_space,
                            nn_kwargs=nn_kwargs
                           )
    

.. note::
    The agent above is NOT trained. So it will basically output "random" actions.

    You should probably train it before hand (see the `train` function)


Detailed documentation
++++++++++++++++++++++++

.. automodule:: l2rpn_baselines.PPO_SB3
    :members:
    :autosummary:
