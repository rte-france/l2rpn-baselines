# L2RPN Baselines

This package holds implementation baselines for the [L2RPN challenge](https://l2rpn.chalearn.org/)

We thank kindly all baselines [contributors](../AUTHORS.txt).

*Disclaimer* All baselines shown in this code are used to serve as example, good practices or demonstrate some concepts. They are in no way optimal and none of them
(to our knowledge) have been calibrated (learning rate is not tuned, neither is the number of layers, the size
of each layers, the activation functions etc.)


## 1. Current available baselines

A list of top performers to some of the past L2RPN competitions can
be found in the documentation at https://l2rpn-baselines.readthedocs.io/en/latest/external_contributions.html

In this package will find some other implementation (not tuned, to serve as examples):

 - [Template](/l2rpn_baselines/Template):

   This a template baseline, provided as an example for contributors.

 - [DoNothing](/l2rpn_baselines/DoNothing):

   The most simple baseline, that takes no actions until it fails.
   
 - [ExpertAgent](/l2rpn_baselines/ExpertAgent)

   An "expert" agent. It uses some expert knowledge about powergrid and graph theory to
   take action facing when there are some overloads.  

 - [PPO_RLLIB](/l2rpn_baselines/PPO_RLLIB)

   Demonstrates how to use a PPO model (reinforcement learning model that achieved good
   performances in some L2RPN competitions) with "ray / rllib" RL framework.

 - [PPO_SB3](/l2rpn_baselines/PPO_SB3)

   Demonstrates how to use a PPO model (reinforcement learning model that achieved good
   performances in some L2RPN competitions) with "stable baselines 3" RL framework.

 - [OptimCVXPY](/l2rpn_baselines/OptimCVXPY)

   Shows how to use a optimization package (in this case cvxpy) to build an
   agent proposing actions computed from this optimizer. Similar to the
   "RL" baseline, for this one the "optimization modeling" is far from 
   state of the art and can be greatly improved.

## 2. How to?

### 2.a Use a baseline
There are multiple way to use a baseline. 

#### Evaluate the performance of a given baseline
Say you want to evaluate the performance on some baselines on a provided environment. For that, you can 
directly use the provided script given by the baseline author.
 
```python
import grid2op
from l2rpn_baselines.Template import evaluate
env = grid2op.make()
res = evaluate(env)
```
You can have more information about extra argument of the "evaluate" function in the [evaluate](Template/evaluate.py) 
file.


#### Train a baseline
In some cases, we baseline author proposed a dedicated code to train their baseline. If that is the case, 
it is easy to use it:
```python
import grid2op
from l2rpn_baselines.Template import train
env = grid2op.make()
res = train(env)
```
You can have more information about extra argument of the "train" function in the [train](Template/train.py) 
file.

#### Load it
/!\ If you want to have access to the baseline named "Template", and know how this baseline works in detail, you
can do the following from a python shell or in a python script:
```python3
import l2rpn_baseline
from l2rpn_baseline.Template import Template
```
And you can use it with a gri2op environment, and perform anything you want with it. **NB** using a baseline
this way requires that you know how to use it, how to build the class you imported, how to train etc.


## 2.b Propose a new baseline
The best way to submit a new baseline is to post an issue on the official github repository of this package 
[l2rpn-baselines](https://github.com/rte-france/l2rpn-baselines) and follow the appropriate template.

Note that before acceptance baselines will be checked by RTE teams. In order to ease the review process, it is
recommended that you post your baseline under one of the following license:
- Apache
- MIT
- BSD clause 2
- BSD clause 3 
- MPL v2.0
