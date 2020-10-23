# L2RPN Baselines

This package holds reference baselines for the [L2RPN challenge](https://l2rpn.chalearn.org/)

We thank kindly all baselines [contributors](../AUTHORS.txt).

*Disclaimer* All baselines shown in this code are used to serve as example. They are in no way optimal and none of them
(to our knowledge) have been calibrated (learning rate is not tuned, neither is the number of layers, the size
of each layers, the activation functions etc.)


## 1. Current available baselines

 - [Template](/l2rpn_baselines/Template):

   This a template baseline, provided as an example for contributors.

 - [DoNothing](/l2rpn_baselines/DoNothing):

   The most simple baseline, that takes no actions until it fails.

 - [DoubleDuelingDQN](/l2rpn_baselines/DoubleDuelingDQN):

   An example of a Double-DQN implementation.

 - [DoubleDuelingRDQN](/l2rpn_baselines/DoubleDuelingRDQN):

   An example of a Recurrent Deep-Q Network implementation.

 - [SliceRDQN](/l2rpn_baselines/SliceRDQN):

   A multi Recurrent Q-streams implementation.
   Where each action class has it's own Q network embedded in the global net. 

 - [DeepQSimple](/l2rpn_baselines/DeepQSimple):

   A simple implementation of the Deep Q Learning algorithm
   
 - [DuelQSimple](/l2rpn_baselines/DuelQSimple):

   An alternative implementation to the Double DQN implementation. 
   
 - [DuelQLeapNet](/l2rpn_baselines/DuelQLeapNet):

   Another alternative implementation to the Double DQN implementation that uses the LeapNet see 
   [LeapNet](https://github.com/BDonnot/leap_net) as a way to model the Q-value.
 
 - [PandapowerOPFAgent](/l2rpn_baselines/PandapowerOPFAgent) 
   
   A baseline thats uses an "Optimal Power Flow", a specific method develop by the power system community to 
   control the flows.
   
 - [Kaist](/l2rpn_baselines/Kaist)
 
   The winning agent of the WCCI 2020 competition based on graph neural networks and transformers.
   
 - [ExpertAgent](/l2rpn_baselines/ExpertAgent)

   An "expert" agent. It uses some expert knowledge about powergrid and graph theory to
   take action facing when there are some overloads.  
   
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


