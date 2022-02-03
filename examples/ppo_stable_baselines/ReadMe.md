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

In this phase TODO


## 3 evaluate the agent

TODO

## 4 preparing the submision

TODO