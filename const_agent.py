import numpy as np
import gym
from itertools import count

from dac_utils import *
from sat4j_env import SAT4JEnvSelHeur


class ConstAgent:
    """
    Simple double DQN Agent
    """

    def __init__(self, eval_env: gym.Env, action: list,
                 bumpers: list=None, bumperStrategies: list=None):
        self._eval_env = eval_env
        self._action = action
        self.bump = bumpers
        self.bumps = bumperStrategies

    def __repr__(self):
        return 'Const Agent'

    def eval(self, episodes: int, max_env_time_steps: int):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play

        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        this_env = self._eval_env
        for e in range(episodes):
            ed, es, er = 0, 0, 0

            s = this_env.reset()
            step_index = 0
            for _ in count():
                a = self._action
                ed += 1
                if self.bump is not None and self.bumps is not None:
                    a = [self.bump[step_index], self.bumps[step_index]]
                    step_index += 1
                print(a)

                ns, r, d, _ = this_env.step(a)
                er += r
                es += 1
                if es >= max_env_time_steps or d:
                    break
                s = ns
            steps.append(es)
            rewards.append(er)
            decisions.append(ed)

        return steps, rewards, decisions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Const Agent to evaluate all individual actions')
    parser.add_argument('instances', type=str, help="Instances to train on", nargs='*')
    parser.add_argument('--actions', nargs='*', type=int, help='Actions to set it to (Int not string)')
    parser.add_argument('--out-dir',
                        default=None,
                        type=str,
                        help='Directory to save results. Defaults to tmp dir.')
    parser.add_argument('--out-dir-suffix',
                        default='seed',
                        type=str,
                        choices=['seed', 'time'],
                        help='Created suffix of directory to save results.')
    parser.add_argument('--seed', '-s',
                        default=12345,
                        type=int,
                        help='Seed')
    parser.add_argument('--port', '-p',
                        default=33377,
                        type=int)
    parser.add_argument('--env-max-steps',
                        default=200,
                        type=int,
                        help='Maximal steps in environment before termination.')
    parser.add_argument('--sat4j-jar-path', type=str, help='Path to sat4j jar',
                        default=os.environ.get('SAT4J_PATH'))
    parser.add_argument('--only-control-bumper', action='store_true', help='Flag to indicate that only the bumper '\
                        'parameter is being controlled.')
    parser.add_argument('--use-additional-features', action='store_true', help='Flag to indicate that additional '
                        'features describing the problem instance(s) should be used.')
    parser.add_argument('--bumper', nargs='+', type=int, default=None)
    parser.add_argument('--bumpes', nargs='+', type=int, default=None)

    # setup output dir
    args = parser.parse_args()
    if args.sat4j_jar_path is None:
        parser.print_usage()
        print('train_and_eval_chainer_agent.py: error: Environment Variable \'SAT4J_PATH\' unset ->'
              ' the following argument is required: --sat4j-jar-path')
        exit(0)

    if args.bumper is not None and args.bumpes is None:
        args.bumpes = []
        for b in args.bumper:
            args.bumpes.append(0)

    out_dir = prepare_output_dir(args, user_specified_dir=args.out_dir,
                                 subfolder_naming_scheme=args.out_dir_suffix)
    eval_env = SAT4JEnvSelHeur(host='', port=args.port + 1, time_step_limit=args.env_max_steps, work_dir=out_dir,
                               jar_path=args.sat4j_jar_path, instances=args.instances,
                               use_expert_feats=args.use_additional_features)

    agent = ConstAgent(eval_env, args.actions, args.bumper, args.bumpes)
    max_env_time_steps = args.env_max_steps

    print('#'*80)
    print(f'Using agent type "{agent}" to evaluate')
    print('#'*80)
    num_eval_episodes = len(args.instances)

    steps, rewards, decisions = agent.eval(num_eval_episodes, max_env_time_steps)
    eval_stats = dict(
        elapsed_time=0,
        training_steps=0,
        training_eps=0,
        avg_num_steps_per_eval_ep=float(np.mean(steps)),
        avg_num_decs_per_eval_ep=float(np.mean(decisions)),
        avg_rew_per_eval_ep=float(np.mean(rewards)),
        std_rew_per_eval_ep=float(np.std(rewards)),
        eval_eps=len(steps)
    )
    with open(os.path.join(out_dir, 'scores.json'), 'a+') as out_fh:
        json.dump(eval_stats, out_fh)
        out_fh.write('\n')

    per_inst_dict = {eval_env.instances[i]: dict(
        num_steps=steps[i],
        num_decisions=decisions[i],
        reward=rewards[i]
    ) for i in range(len(steps))}
    with open(os.path.join(out_dir, 'scores_per_inst.json'), 'a+') as out_fh:
        json.dump(per_inst_dict, out_fh)
        out_fh.write('\n')
