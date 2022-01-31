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

## 2 Training the agent

## 3 evaluate the agent

## 4 preparing the submision
