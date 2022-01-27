.. currentmodule:: l2rpn_baselines.ppo_stablebaselines

PPO Stable Baselines
===========================================================

Description
-----------
This "baseline" aims at providing a code example on how to use an agent
from the Sable Baselines repository (see https://stable-baselines3.readthedocs.io/en/master/)
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

    from l2rpn_baselines.ppo_stablebaselines import train, evaluate, PPOSB_Agent


Create an agent from scratch
++++++++++++++++++++++++++++++

For example, to create an agent from scratch, with some parameters:

.. code-block:: python

    import grid2op
    from grid2op.gym_compat import GymEnv, BoxGymActSpace

    # create the grid2op environment
    env = grid2op.make(...)
    #############

    # convert it to a suitable gym environment
    env_gym = GymEnv(env)
    env_gym.action_space.close()
    env_gym.action_space = BoxGymActSpace(env.action_space)
    #############

    # create the PPO Stable Baselines agent (only some basic configs are given here)
    agent = PPOSB_Agent(env.action_space,
                        env_gym.action_space,
                        env_gym.observation_space,
                        nn_kwargs={
                            "policy": MlpPolicy,  # or any other stable baselines 3 policy
                            "env": env_gym,
                            "verbose": 1,  # or anything else
                            "learning_rate": 3e-4,  # you can change that
                            "policy_kwargs": {
                                "net_arch": [100, 100, 100]  # and that
                            }
                        },
                        nn_path=None
                        )

.. note::
    The agent above is NOT trained. So it will basically output "random" actions.

    You should probably train it before hand (see the `train` function)

Load a trained agent
+++++++++++++++++++++++
You can also load a trained agent, to use it with a grid2op environment, in a runner,
in grid2game or any other frameworks related to grid2op.


.. code-block:: python

    import grid2op
    from grid2op.gym_compat import GymEnv, BoxGymActSpace

    # create the grid2op environment
    env = grid2op.make(...)
    #############

    # convert it to a suitable gym environment
    env_gym = GymEnv(env)
    env_gym.action_space.close()
    env_gym.action_space = BoxGymActSpace(env.action_space)
    #############

    # create the PPO Stable Baselines agent (only some basic configs are given here)
    agent = PPOSB_Agent(env.action_space,
                        env_gym.action_space,
                        env_gym.observation_space,
                        nn_path=...  # path where you saved it !
                        )


Detailed documentation
++++++++++++++++++++++++

.. automodule:: l2rpn_baselines.ppo_stablebaselines
    :members:
    :autosummary:

Other non exported class
------------------------
These classes need to be imported, if you want to import them with (non exhaustive list):
.. code-block:: python

    from l2rpn_baselines.DeepQSimple.DeepQ_NN import DeepQ_NN


.. autoclass:: l2rpn_baselines.DeepQSimple.DeepQ_NN.DeepQ_NN
    :members:
    :autosummary:
