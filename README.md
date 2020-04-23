# L2RPN_Baselines
Repository hosting reference baselines for the [L2RPN challenge](https://l2rpn.chalearn.org/)

# Install 

## Requirements
`python3 >= 3.6`

## Instal from PyPI
```sh
pip3 install l2rpn_baselines
```
## Install from source
```sh
git clone https://github.com/rte-france/l2rpn-baselines.git
cd l2rpn-baselines
pip3 install -U .
cd ..
rm -rf l2rpn-baselines
```

# Contribute

We welcome contributions: see the [contribute guide](/CONTRIBUTE.md) for details.

# Get started with a baseline

Say you want to know how you compared with the "DoubleDuelingDQN" baseline implementation in this repository (for the
sake of this example).

## Train it (optional)
As no weights are provided for this baselines by default (yet), you will first need to train such a baseline:

```python
import grid2op
from l2rpn_baselines.DoubleDuelingDQN import train
env = grid2op.make()
res = train(env, save_path="THE/PATH/TO/SAVE/IT", iterations=100)
```

You can have more information about extra argument of the "train" function in the 
[CONTRIBUTE](/CONTRIBUTE.md) file.

## Evaluate it
Once trained, you can reload it and evaluate its performance with the provided "evaluate" function:

```python
import grid2op
from l2rpn_baselines.DoubleDuelingDQN import evaluate
env = grid2op.make()
res = evaluate(env, load_path="THE/PATH/TO/LOAD/IT.h5", nb_episode=10)
```

You can have more information about extra argument of the "evaluate" function in the 
[CONTRIBUTE](/CONTRIBUTE.md) file.
