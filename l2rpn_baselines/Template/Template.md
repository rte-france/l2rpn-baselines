# Authors
 - Benjamin Donnot \<benjamin.donnot@rte-france.com\>
 - Jean Grizet \<jean.grizet@gmail.com\>

# Publications
 - If applicable

# Performances
```bash
./evaluate.py --path_data ~/data_grid2op/l2rpn_2019 --path_model na

Evaluation summary:
	For chronics located at 000
		 - cumulative reward: 888427.612122
		 - number of time steps completed: 799 / 8000
```

# Training duration
 - If applicable

# Training hyperparameters
 - If applicable
```json
{
  "lr": 1e-05,
  "batch_size": 32,
  "iter": 131328,
  "e_start": 0.9,
  "e_end": 0.0,
  "e_decay": 32768,
  "discount": 0.99,
  "buffer_size": 65536,
  "update_freq": 32,
  "update_hard": 5,
  "update_soft": 0.01,
  "reward": {
    "game": {
      "name": "GameplayReward",
      "reward_min": -1.0,
      "reward_max": 1.0,
      "weight": 10.0
    }
  }
```

# Training environment

## System
```bash
uname -s -r -v

Linux 5.3.0-46-generic #38~18.04.1-Ubuntu SMP Tue Mar 31 04:17:56 UTC 2020
```

## CPU

```bash
lscpu | head -n 15

Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              12
On-line CPU(s) list: 0-11
Thread(s) per core:  2
Core(s) per socket:  6
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               158
Model name:          Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
Stepping:            10
CPU MHz:             800.047
```

## GPU

```bash
lspci | grep VGA

01:00.0 VGA compatible controller: NVIDIA Corporation GP106 [GeForce GTX 1060 6GB] (rev a1)
```

