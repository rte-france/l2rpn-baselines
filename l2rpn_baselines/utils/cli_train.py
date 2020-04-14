import argparse


def cli_train():
    parser = argparse.ArgumentParser(description="Train baseline DDQN")
    parser.add_argument("--path_data", required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--name", required=True,
                        help="The name of the model")
    parser.add_argument("--batch_size", required=False,
                        default=32, type=int,
                        help="Mini batch size (defaults to 32)")
    parser.add_argument("--num_pre_steps", required=False,
                        default=256, type=int,
                        help="Number of random steps before training")
    parser.add_argument("--num_train_steps", required=False,
                        default=1024, type=int,
                        help="Number of training iterations")
    parser.add_argument("--trace_len", required=False,
                        default=8, type=int,
                        help="Size of the trace to use during training")
    parser.add_argument("--learning_rate", required=False,
                        default=1e-5, type=float,
                        help="Learning rate for the Adam optimizer")
    parser.add_argument("--resume", required=False,
                        help="Path to model.h5 to resume training with")
    return parser