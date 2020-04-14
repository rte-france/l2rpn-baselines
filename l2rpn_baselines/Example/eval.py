"""
In this file, explain how to evaluate your agent.
"""
import os
from grid2op.Runner import Runner
from grid2op.Plot import EpisodeReplay

from l2rpn_baselines.Example.ExampleAgent import ExampleAgent


def main(env, args_cli):
    """

    Parameters
    ----------
    env
    args_cli

    Returns
    -------

    """
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = args_cli.verbose

    # Create the agent (this piece of code can change)
    agent = ExampleAgent(env.action_space, env.observation_space)

    # Load weights from file (for example)
    agent.load(args_cli.path_model)

    # Build runner
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)

    # you can do stuff with your model here

    # start the runner
    res = runner.run(path_save=args_cli.path_logs,
                     nb_episode=args_cli.nb_episode,
                     nb_process=args_cli.nb_process,
                     max_iter=args_cli.max_steps,
                     pbar=False)

    # Print summary
    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

    if args_cli.save_gif:
        ep_replay = EpisodeReplay(agent_path=args_cli.path_logs)
        for _, chron_name, cum_reward, nb_time_step, max_ts in res:
            ep_replay.replay_episode(chron_name,
                                     video_name=os.path.join(args_cli.path_logs, chron_name, "epidose.gif"),
                                     display=False)


if __name__ == "__main__":
    import grid2op
    from l2rpn_baselines.utils import cli_eval
    args_cli = cli_eval().parse_args()
    env = grid2op.make()