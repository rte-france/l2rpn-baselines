# Objective

This directory shows how to use grid2op, lightsim2grid and l2rpn-baselines to make a RL agent that is able to perform some actions on a grid2op environment using the PPO algorithm and the `stable-baselines3` rl library.

It focuses on the `PPO_SB3` baseline with a strong focus on **continuous** variables (curtailment and redispatching)

It will be usable on the `l2rpn_icaps_2021` grid2op environment

It is organized as follow:

1) you split the environment into training, validation and test
2) you train the agent (do not hesitate to change the parameters there) on the
   training set
3) you evaluate it on a dataset not used for training !
4) once your are "happy" with your results on step 3 (so you will probably need to
   run step 2 and 3 multiple times...) you can submit it to a l2rpn competition

## 1 Preparing the training environment

This is done by running the script `A_prep_env.py` 

In this phase, we do 3 things:

- we split the data set into a training, validation and test set. This is quite standard in ML (less in RL) and its main goal is to prevent overfitting. (we remind the scenarios on codalab will be different from the training set provided, though drawn from the same distribution)
- we initialize the computation of the scores. In the case of l2rpn competitions, the score is cannot be easily made into a reward function, it can only be computed when knowing the entire episode, at the end of the episode\*. 
- we compute the score of a few "standard" baselines to compared the trained agent with
- we use the previous runs to compute some "statistics" (average and standard deviation) used to normalize the actions / observations in the later scripts.

\* of course you can make a sparse reward from it. Your agent receive always 0.0 unless when "done = True" (so last step of the episode) where this score can be computed. This is not the approach we took here.

**This script might take a while to compute !**


## 2 Training the agent

In this phase where the training takes place and is implemented in the script `B_train_agent.py`

This script will show you how to modify the reward function (if needed), how to select some part of the observation and the action space to train a `PPO` using the "stable baselines 3" framework. This agent only uses **continuous** action types (`redispatching`, `curtailment` and action on `storage units`) and does not modify the topology at all.

This script leverage the most common pattern used by best performing submissions at previous l2rpn competitions and allows you to train agents using some "heuristics" (*eg* "do not act when the grid is safe" or "reconnect a powerline as soon as you can"). This is made possible by the implementation of such "heursitics" directly in the environment: the neural network (agent) only gets observations when it should do something. Said differently, when a heuristic can operate the grid, the NN is "skipped" and does not even sees the observation. At inference time, the same mechanism is used. This makes the training and the evaluation consistent with one another.

This also means that the number of steps performed by grid2op is higher than the number of observations seen by the agent. The training can take a long time.


What is of particular importance in this script, beside the usual "learning rate" and "neural network architecture" is the "`safe_max_rho`" meta parameters. This parameters controls when the agent is asked to perform an action (when any `obs.rho >= safe_max_rho`). If it's too high, then the agent will almost never act and might not learn anything. If it's too low then the "heuristic" part ("do nothing when the grid is safe") will not be used and the agent might take a lot of time to learn this.

**This script might take a while to compute !**

## 3 evaluate the agent

This is done with the script "`C_evaluate_trained_model.py`" and it reports the score as if you submitted your agent to codalab platform (be aware that the real score depends on the chronix in the validation / test set as well as the seed used but can also vary depending on the versions of the different packages you installed, especially grid2op and lightsim2grid).

Do not hesitate to use this script multiple times to make sure your agent is consistent (*eg* it would be rather optimistic to rely on an agent that performs really well on some runs and really poorly on others...).

You might also refine some of the "parameters" of your agents here. For example, by
default we use a `safe_max_rho` of 0.9, but you might want to change it to 0.8 or 0.95 to improve the performance of your agent.

## 4 preparing the submision

Before you submit your agent, you need to make sure that it is trained
on the same environment than the one it will be tested on (unless you took
particular care to have an agent able to operate different grids).

You also need to make sure that this agent run using the same packages version
than the one you used locally. For example, for wcci 2022 competition your agent
is expected to run on grid2op 1.7.1 and lightsim2grid version 0.7.0

Finally, you need to make sure that your agent does not use packages that are not availlable at test time. The list of available pacakge is usually found on the 
description of the competition (on this aspect: unless told otherwise, you are free to use any package that you want at training time. Only at test time you are required to provide an agent working with the installed packages).

Once done, all is required to submit your agent to the competition is that you
provide a "`make_agent(env, path)`" function with:

- `env` being a grid2op environment with the same properties as the one that
  that will be used to test your agent (but it's not the actual test environment)
- `path` is the location where the code is executed, it is usefull if you need
  extra data to use your agent (in this case the weights of the neural networks
  used in the policy or the normalizer for the observation, etc.)

A possible implementation is:

```python
from l2rpn_baselines.ppo_stable_baselines3 import evaluate
safe_max_rho = 0.9  # or the one you find most suited for your agent

def make_agent(env, path):
   agent, _ = evaluate(env,
                       load_path=os.path.join(path, TheDireYouUsedToSaveTheAgent),
                       name=TheNameOfYourAgent,
                       nb_episode=0,
                       gymenv_class=ThePossibleGymEnvClass,
                       gymenv_kwargs={"safe_max_rho": safe_max_rho}  # only if you used the `GymEnvWithRecoWithDN` environment, otherwise any
                       # other parmeters you might need
                       )
   return agent

# NB: by default:
#  - TheDireYouUsedToSaveTheAgent is "saved_model"
#  - TheNameOfYourAgent is "PPO_SB3"
#  - ThePossibleGymEnvClass is GymEnvWithRecoWithDN (that you need to 
#    import with `from l2rpn_baselines.utils import GymEnvWithRecoWithDN`)
```

(do not forget to include the `preprocess_act.json` and `preprocess_obs.json` files in the submission as well as the "saved_model" directory, if possible ony containing
the agent you want to test and not all your runs.)

All you need to do is to follow the instructions in the starting kit of the competition to zip properly all these data.
