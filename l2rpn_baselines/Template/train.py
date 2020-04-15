#!/usr/bin/env python3

from l2rpn_baselines.Template.TemplateBaseline import TemplateBaseline

def train(env,
          name="Template",
          iterations=1,
          save_path=None,
          load_path=None,
          **kwargs):
    """
    This an example function to train a baseline.

    In order to be valid, if you chose (which is recommended) to provide a training script to help other retrain your
    baseline in different environments, or for longer period of time etc. This script should be contain the "train"
    function with at least the following arguments.

    Parameters
    ----------
    env: :class:`grid2op.Environment.Environment`
        The environmnent on which the baseline will be trained

    name: ``str``
        Fancy name you give to this baseline.

    iterations: ``int``
        Number of training iterations to perform

    save_path: ``str``
        The path where the baseline will be saved at the end of the training procedure.

    load_path: ``str``
        Path where to look for reloading the model. Use ``None`` if no model should be loaded.

    kwargs:
        Other key-word arguments that you might use for training.

    """

    baseline = TemplateBaseline(env.action_space,
                                env.observation_space,
                                name=name)

    if load_path is not None:
        baseline.load(load_path)

    baseline.train(env, iterations, save_path)
    # as in our example (and in our explanation) we recommend to save the mode regurlarly in the "train" function
    # it is not necessary to save it again here. But if you chose not to follow these advice, it is more than
    # recommended to save the "baseline" at the end of this function with:
    # baseline.save(path_save)


if __name__ == "__main__":
    """
    This is a possible implementation of the train script.
    """
    import grid2op
    from l2rpn_baselines.utils import cli_train
    args_cli = cli_train().parse_args()
    env = grid2op.make()
    train(env=env,
          name=args_cli.name,
          iterations=args_cli.num_train_steps,
          save_path=args_cli.path_save,
          load_path=args_cli.path_loading)
