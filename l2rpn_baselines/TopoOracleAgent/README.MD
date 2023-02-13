# Authors
 - Antoine Marot 

# How to use
You need to install Oracle4Grid in order to use this agent :
Detailed documentation here : 

https://oracle4grid.readthedocs.io/en/latest/index.html

- install in your environment

`pip install oracle4grid (compatible with lastest Grid2op 1.8 and lightsim2grid)

# Getting started

- Choose the environement and the scenario you want to study
- Define Unitary actions of interest as in ressources/actions/Ieee14_Sandbox_test/unitary_actions_l2rpn_2019.json (several format possible as described there https://oracle4grid.readthedocs.io/en/latest/DESCRIPTION.html#action-parsing )
```json
{"sub":
	{"4": 
		[{"lines_id_bus": [[4, 2]], "loads_id_bus": [[3, 2]]},
			{"lines_id_bus": [[4, 2], [6, 2]]},
			{"lines_id_bus": [[1, 2], [17, 2]]}],
	"3": [{"lines_id_bus": [[6, 2], [16, 2]]}, 
		{"lines_id_bus": [[6, 2], [15, 2]]}, 
		{"lines_id_bus": [[16, 2], [5, 2]]}, 
		{"lines_id_bus": [[16, 2], [5, 2], [6, 2]]}], 
	"1": [{"lines_id_bus": [[0, 2], [4, 2]], "loads_id_bus": [[0, 2]]}, 
		{"lines_id_bus": [[0, 2], [4, 2], [2, 2]]}, 
		{"lines_id_bus": [[0, 2], [3, 2], [4, 2]]}], 
	"8": [{"lines_id_bus": [[19, 2], [10, 2], [11, 2]]},
		{"lines_id_bus": [[16, 2], [11, 2]]}], 
	"5": [{"lines_id_bus": [[17, 2], [9, 2], [7, 2]]}]
	}
}
```
- Choose your reward or score and set it in ressources/constants.py
```python
# Grid2Op Env constants
class EnvConstants:
    def __init__(self):
        # Reward class used for computing best path for oracle
        self.reward_class = YourReward#OracleL2RPNReward
```
- Launch the training phase to find the best path of actions. Try to run it on your largest CPU server to get massive speedup. Define the combinatorial depth of actions you want to explore. Make explicit if your reward should be minimize (resp. maximize) by setting best_action_path_type to "shortest" (resp. "longest")
```python
    train(env,
          name = args.name,chronic=args.scenario,
          env_seed=args.env_seed,agent_seed=args.env_seed,
          iterations = args.max_timesteps,
          max_combinatorial_depth=args.max_depth,
          save_path = args.save_dir,
          logs_path = args.logs_dir,
          action_file_path = args.action_file,
          nb_process=args.n_cores,
          best_action_path_type=args.best_action_path_type,
          reward_significant_digit=args.n_significant_digits)
```
- You will get the best action path stored in save_path root folder in oracle_actions.csv
```
timestep | action_path
t0       | line-4-3
t1       | line-4-3
[...]
t116     | line-4-3_line-16-4
t117     | line-4-3_line-16-4
[...]
t288     | line-4-3
```
- You can then evaluate it with your agent using. Only one CPU is needed at this stage
```python
    evaluate(env,
          name = args.name,chronic=args.scenario,
          env_seed=args.env_seed,agent_seed=args.env_seed,
          iterations = args.max_timesteps,
          max_combinatorial_depth=args.max_depth,
          save_path = args.save_dir,
          logs_path = args.logs_dir,
          action_file_path = args.action_file,
          reward_significant_digit=args.n_significant_digits,
          verbose=args.verbose,
          save_gif=args.gif
          )
```


