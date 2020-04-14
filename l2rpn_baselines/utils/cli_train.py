import argparse


def cli_train():
    parser = argparse.ArgumentParser(description="Train baseline DDQN")
    parser.add_argument("--num_train_steps", required=False,
                        default=1024, type=int,
                        help="Number of training iterations")
    parser.add_argument("--path_save", required=False,
                        help="Path where the model should be saved.")
    parser.add_argument("--name", required=False,
                        help="Name given to your model.")
    parser.add_argument("--path_loading", required=False,
                        help="Path from which to reload your model from (by default ``None`` to NOT reload anything)")
    return parser