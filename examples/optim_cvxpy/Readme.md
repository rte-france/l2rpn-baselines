# Objective

This repo shows (and give some usefull parameters) how to use the optimization
`OptimCVXPY` method to tackle the problem in grid2op.

Parameters given here are not perfect but they allow to perform better than "do nothing"
on the particular scenario selected (selection has been made uniformly at random
between all scenario of the environment)

## On the educ_case14_storage

On this environment, the optimization procedure is pretty fast (~10-15 steps per second) and allow to 
get through almost all the scenarios.

It's probably possible to do better by fine tuning the other hyper parameters.

You can have a look at the [**optimcvxpy_educ_case14_storage.py**](./optimcvxpy_educ_case14_storage.py) file for more information.

## On the wcci 2022 environment

For this environment, the model is pretty slow (sometimes 10-15s per step which is relatively important).
This leads to around 30 mins for completing a full scenario of a week (2016 steps)

Because it took long time to compute, we only manage to find "good" parameters to do better than do nothing
for the selected scenarios.

You can have a look at the [**optimcvxpy_wcci_2022.py**](./optimcvxpy_wcci_2022.py) file for more information.
