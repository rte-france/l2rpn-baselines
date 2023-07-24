# Objective

This directory shows you the whole process of making a RL agent that uses stable-baselines3 and grid2op from its
training to the final submission in codalab.

All the files here are required for codalab.


## l2rpn baselines
In the sandbox competition (when there is no "alert" and no "sim2real") the l2rpn-baselines used in the evaluation docker is really old. It will be updated
for the real competition (tracks "alert" and "sim2real") but in the mean time I had to manually add the correct implementation of `GymEnvWithHeuristics`

Again, this is required for the sandbox, but will no longer be required (and I would advise against its inclusion) for the main tracks.

## Agent "training"

Training has been done for only a limited time, using only some continuous action (the training of an agent using discrete action will follow). This agent does not perform really well. Use as a "code template" (like an "hello world") and probably not as the gold standard of anything.

In particular, we did not:
- take too much time on how to normalize the data
- chose carefully the part of the observation the agent will use
- use some observation "stacking" (stack multiple observation)
- use the most relevant part of the action (we show only how to use continuous actions, which is not the target of the competition)
- tune the neural network shape (size and number of layers)
- chose the correct RL algorithm (we used PPO as a default here, but maybe SAC or any other RL algorithm can work better)
- tune the learning of the neural network (learning rate, number of updates, loss, etc.)
- tune the meta parameters of the RL algorithm (discount factor, number of episode to "roll out" before making updates, number of updates)
- tune the meta parameters of the Gymnasium environment (`safe_max_rho` and `curtail_margin`)

We just adapted the example code of "how to train a PPO agent with stable baselines" and added a few useful tricks  on how to make it 
work "efficiently" in grid2op, this include:
- using `lightsim2grid` solver
- caching the data in memory (using `MultifolderWithCache`)
- splitting the original environment between train, evaluation and test set
- selecting which part of the observations to keep
- normalizing the observations
- using some "heuristics" to handle "trivial" actions, like reconnecting powerline instead of
  spending time to try to learn it

This explains why it's not a "minimal code example". We do not pretend that everything in this agent is mandatory at all. Some
part might be totally useless.