import argparse
from l2rpn_baselines.utils.str2bool import str2bool


# TODO add possibilitiy to add new CLI when calling the function
def cli_eval():
    parser = argparse.ArgumentParser(description="Train baseline DDQN")
    parser.add_argument("--path_data", required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--path_model", required=True,
                        help="The path to the model [.h5]")
    parser.add_argument("--path_logs", required=False,
                        default="./logs_eval", type=str,
                        help="Path to output logs directory")
    parser.add_argument("--nb_episode", required=False,
                        default=1, type=int,
                        help="Number of episodes to evaluate")
    parser.add_argument("--nb_process", required=False,
                        default=1, type=int,
                        help="Number of cores to use")
    parser.add_argument("--max_steps", required=False,
                        default=8000, type=int,
                        help="Maximum number of steps per scenario")
    parser.add_argument("--verbose", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Enable verbose runner mode..")
    parser.add_argument("--save_gif", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Save the gif as \"epidose.gif\" in the episode path module.")
    return parser
