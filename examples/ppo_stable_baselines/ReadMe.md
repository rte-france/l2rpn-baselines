# Objective

This repository demonstrates how to use grid2op, lightsim2grid and l2rpn-baselines to make a RL agent that is able to perform some actions on a grid2op environment using the PPO algorithm and the `stable-baselines3` rl library.

It focuses on the `PPO_SB3` baseline with a strong focus on **continuous** variables (curtailment and redispatching)

It will be usable on the `l2rpn_icaps_2021` grid2op environment

It is organized as follow:

1) you split the environment into training and validation
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

\* of course you can make a sparse reward from it. Your agent receive always 0.0 unless when "done = True" (so last step of the episode) where this score can be computed. This is not the approach we took here.

## 2 Training the agent

In this phase where the training takes place and is implemented in the script `B_train_agent.py`

This script will show you how to modify the reward function (if needed), how to select some part of the observation and the action space to train a `PPO` using the "stable baselines 3" framework. This agent only uses **continuous** action types (`redispatching`, `curtailment` and action on `storage units`) and does not modify the topology at all.

This script leverage the most common pattern used by best performing submissions at previous l2rpn competitions and allows you to train agents using some "heuristics" (*eg* "do not act when the grid is safe" or "reconnect a powerline as soon as you can"). This is made possible by the implementation of such "heursitics" directly in the environment: the neural network (agent) only gets observations when it should do something. Said differently, when a heuristic can operate the grid, the NN is "skipped" and does not even sees the observation. At inference time, the same mechanism is used. This makes the training and the evaluation consistent with one another.

This also means that the number of steps performed by grid2op is higher than the number of observations seen by the agent. The training can take a long time.


What is of particular importance in this script, beside the usual "learning rate" and "neural network architecture" is the "`safe_max_rho`" meta parameters. This parameters controls when the agent is asked to perform an action (when any `obs.rho >= safe_max_rho`). If it's too high, then the agent will almost never act and might not learn anything. If it's too low then the "heuristic" part ("do nothing when the grid is safe") will not be used and the agent might take a lot of time to learn this.

## 3 evaluate the agent

TODO

## 4 preparing the submision

TODO