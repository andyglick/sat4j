import os
import json
import pickle

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import count
from collections import namedtuple
import time

from dac_utils import *
from sat4j_env import SAT4JEnvSelHeur

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tt(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    if device == "cuda":
        return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
    else:
        return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


def soft_update(target, source, tau):
    """
    Simple Helper for updating target-network parameters
    :param target: target network
    :param source: policy network
    :param tau: weight to regulate how strongly to update (1 -> copy over weights)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    See soft_update
    """
    soft_update(target, source, 1.0)


class Q(nn.Module):
    """
    Simple fully connected Q function. Also used for skip-Q when concatenating behaviour action and state together.
    Used for simpler environments such as mountain-car or lunar-lander.
    """

    def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """
    Simple Replay Buffer. Used for standard DQN learning.
    """

    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, next_state, reward, done):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array([self._data.terminal_flags[i] for i in batch_indices])
        return tt(batch_states), tt(batch_actions), tt(batch_next_states), tt(batch_rewards), tt(batch_terminal_flags)

    def save(self, path):
        with open(os.path.join(path, 'rpb.pkl'), 'wb') as fh:
            pickle.dump(list(self._data), fh)

    def load(self, path):
        with open(os.path.join(path, 'rpb.pkl'), 'rb') as fh:
            data = pickle.load(fh)
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data.states = data[0]
        self._data.actions = data[1]
        self._data.next_states = data[2]
        self._data.rewards = data[3]
        self._data.terminal_flags = data[4]
        self._size = len(data[0])


class DQN:
    """
    Simple double DQN Agent
    """

    def __init__(self, state_dim: int, action_dim: int, gamma: float,
                 env: gym.Env, eval_env: gym.Env, train_eval_env: gym.Env = None, vision: bool = False,
                 factored_action_space: list = None, do_checkpoint: bool = True):
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param eval_env: environment to evaluate on with training data
        :param vision: boolean flag to indicate if the input state is an image or not
        """
        if not vision:  # For featurized states
            self._q = Q(state_dim, action_dim).to(device)
            self._q_target = Q(state_dim, action_dim).to(device)
        else:  # For image states, i.e. Atari
            raise NotImplementedError

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._action_dim = action_dim

        self._replay_buffer = ReplayBuffer(1e6)
        self._env = env
        self._eval_env = eval_env
        self._train_eval_env = train_eval_env
        self._facts = [[i, j] for i in range(factored_action_space[0]) for j in range(factored_action_space[1])]
        self.do_checkpoint = do_checkpoint

    def save_rpb(self, path):
        self._replay_buffer.save(path)

    def load_rpb(self, path):
        self._replay_buffer.load(path)

    def get_action(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def run_and_log_evaluation(self, eval_eps, start_time, total_steps, e, only_checkpoint: bool = True):
        """
        Method that takes care of runing evaluation and storing results in easy to read format.
        :param eval_eps: number of evaluation_episodes
        :param start_time: time.time() of training start
        :param total_steps: the total performed steps so far
        :param e: the current episode at which we evaluate
        :param only_checkpoint: flag indicating if runs are performed or only the checkpoint path is stored.
        """
        print('Begin Evaluation')
        if not only_checkpoint:
            eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps)
        else:
            eval_s, eval_r, eval_d = [-1], [-1], [-1]
            checkpoint_path = os.path.join(out_dir, 'checkpoints', f'{total_steps:05d}')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            self.save_model(checkpoint_path)
        eval_stats = dict(
            elapsed_time=time.time() - start_time,
            training_steps=total_steps,
            training_eps=e,
            avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
            avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
            avg_rew_per_eval_ep=float(np.mean(eval_r)),
            std_rew_per_eval_ep=float(np.std(eval_r)),
            eval_eps=eval_eps,
            eval_insts=self._eval_env.instances,
            reward_per_insts=eval_r,
            steps_per_insts=eval_s
        )
        if only_checkpoint:
            eval_stats['checkpoint_path'] = checkpoint_path

        filename = 'eval_scores.json' if not only_checkpoint else 'eval_checkpoints.json'
        with open(os.path.join(out_dir, filename), 'a+') as out_fh:
            json.dump(eval_stats, out_fh)
            out_fh.write('\n')

        # Do the same thing on the training data if required
        if self._train_eval_env is not None:
            if not only_checkpoint:
                eval_s, eval_r, eval_d = self.eval(eval_eps, max_env_time_steps, train_set=True)
            else:
                eval_s, eval_r, eval_d = [-1], [-1], [-1]
                checkpoint_path = os.path.join(out_dir, 'checkpoints', total_steps)
            eval_stats = dict(
                elapsed_time=time.time() - start_time,
                training_steps=total_steps,
                training_eps=e,
                avg_num_steps_per_eval_ep=float(np.mean(eval_s)),
                avg_num_decs_per_eval_ep=float(np.mean(eval_d)),
                avg_rew_per_eval_ep=float(np.mean(eval_r)),
                std_rew_per_eval_ep=float(np.std(eval_r)),
                eval_eps=eval_eps,
                eval_insts=self._train_eval_env.instances,
                reward_per_insts=eval_r,
                steps_per_insts=eval_s
            )
            if only_checkpoint:
                eval_stats['checkpoint_path'] = checkpoint_path

            filename = 'train_scores.json' if not only_checkpoint else 'train_checkpoints.json'
            with open(os.path.join(out_dir, filename), 'a+') as out_fh:
                json.dump(eval_stats, out_fh)
                out_fh.write('\n')
        print('End Evaluation')

    def train(self, episodes: int, max_env_time_steps: int, epsilon: float, eval_eps: int = 1,
              eval_every_n_steps: int = 1, max_train_time_steps: int = 1_000_000):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_eps: numper of episodes to run for evaluation
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        :return:
        """
        total_steps = 0
        start_time = time.time()
        print(f'Start training at {start_time}')
        for e in range(episodes):
            # print('\033c')
            # print('\x1bc')
            if e % 100 == 0:
                print("%s/%s" % (e + 1, episodes))
            s = self._env.reset()
            for t in range(max_env_time_steps):
                a = self.get_action(s, epsilon)
                if self._facts is not None:
                    env_a = self._facts[a]
                else:
                    env_a = a
                ns, r, d, _ = self._env.step(env_a)
                total_steps += 1

                ########### Begin Evaluation
                if (total_steps % eval_every_n_steps) == 0:
                    self.run_and_log_evaluation(eval_eps, start_time, total_steps, e,
                                                only_checkpoint=self.do_checkpoint)
                ########### End Evaluation

                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)

                ########### Begin double Q-learning update
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = \
                    self._replay_buffer.random_next_batch(64)
                target = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                         self._q_target(batch_next_states)[torch.arange(64).long(), torch.argmax(
                             self._q(batch_next_states), dim=1)]
                current_prediction = self._q(batch_states)[torch.arange(64).long(), batch_actions.long()]

                loss = self._loss_function(current_prediction, target.detach())

                self._q_optimizer.zero_grad()
                loss.backward()
                self._q_optimizer.step()

                soft_update(self._q_target, self._q, 0.01)
                ########### End double Q-learning update

                if d:
                    break
                s = ns
                if total_steps >= max_train_time_steps:
                    break
            if total_steps >= max_train_time_steps:
                break

        # Final evaluation
        self.run_and_log_evaluation(eval_eps, start_time, total_steps, e,
                                    only_checkpoint=self.do_checkpoint)

    def __repr__(self):
        return 'DDQN'

    def eval(self, episodes: int, max_env_time_steps: int, train_set: bool = False):
        """
        Simple method that evaluates the agent with fixed epsilon = 0
        :param episodes: max number of episodes to play
        :param max_env_time_steps: max number of max_env_time_steps to play

        :returns (steps per episode), (reward per episode), (decisions per episode)
        """
        steps, rewards, decisions = [], [], []
        this_env = self._eval_env if not train_set else self._train_eval_env
        with torch.no_grad():
            for e in range(episodes):
                ed, es, er = 0, 0, 0

                s = this_env.reset()
                for _ in count():
                    a = self.get_action(s, 0)
                    if self._facts is not None:
                        env_a = self._facts[a]
                    else:
                        env_a = a
                    ed += 1

                    ns, r, d, _ = this_env.step(env_a)
                    er += r
                    es += 1
                    if es >= max_env_time_steps or d:
                        break
                    s = ns
                steps.append(es)
                rewards.append(er)
                decisions.append(ed)

        return steps, rewards, decisions

    def save_model(self, path):
        torch.save(self._q.state_dict(), os.path.join(path, 'Q'))

    def load(self, path):
        self._q.load_state_dict(torch.load(os.path.join(path, 'Q')))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Online DQN training')
    parser.add_argument('instances', type=str, help="Instances to train on", nargs='*')
    parser.add_argument('--val-instances', type=str, help="Instances to validate on", nargs='*')
    parser.add_argument('--episodes', '-e',
                        default=100,
                        type=int,
                        help='Number of training episodes.')
    parser.add_argument('--training-steps', '-t',
                        default=1_000_000,
                        type=int,
                        help='Number of training episodes.')

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
    parser.add_argument('--eval-after-n-steps',
                        default=10 ** 3,
                        type=int,
                        help='After how many steps to evaluate')
    parser.add_argument('--env-max-steps',
                        default=200,
                        type=int,
                        help='Maximal steps in environment before termination.')
    parser.add_argument('--load-model', default=None)
    parser.add_argument('--agent-epsilon', default=0.2, type=float, help='Fixed epsilon to use during training',
                        dest='epsilon')
    parser.add_argument('--sat4j-jar-path', type=str, help='Path to sat4j jar',
                        default=os.environ.get('SAT4J_PATH'))
    parser.add_argument('--only-control-bumper', action='store_true', help='Flag to indicate that only the bumper '
                                                                           'parameter is being controlled.')
    parser.add_argument('--use-additional-features', action='store_true', help='Flag to indicate that additional '
                                                                               'features describing the problem instance(s) should be used.')
    parser.add_argument('--reward-type', choices=['time', 'time_proxy', 'control_steps'], default='control_steps')
    parser.add_argument('--direct-evaluation', action='store_true', help='Do evaluate directly during training.')
    parser.add_argument('--validate', default=None, help='Path to trajectory file to validate', type=str)
    parser.add_argument('--validate-type', choices=['train', 'eval'], default='eval',
                        help='Which instances to evaluate', type=str)

    # setup output dir
    args = parser.parse_args()
    if args.sat4j_jar_path is None:
        parser.print_usage()
        print('train_and_eval_chainer_agent.py: error: Environment Variable \'SAT4J_PATH\' unset ->'
              ' the following argument is required: --sat4j-jar-path')
        exit(0)

    if not args.load_model and not args.validate:
        out_dir = prepare_output_dir(args, user_specified_dir=args.out_dir,
                                     subfolder_naming_scheme=args.out_dir_suffix)
        eval_dir = os.path.join(out_dir, 'eval_envdir')
        os.makedirs(eval_dir, exist_ok=False)
    elif args.load_model:
        out_dir = eval_dir = os.path.join(args.load_model, '..', 'val_envdir')
        os.makedirs(eval_dir, exist_ok=False)
    else:
        out_dir = os.path.join(os.path.dirname(args.validate))
        eval_dir = os.path.join(os.path.dirname(args.validate), 'eval_envdir')

    env = SAT4JEnvSelHeur(host='', port=args.port, time_step_limit=args.env_max_steps, work_dir=out_dir,
                          jar_path=args.sat4j_jar_path, instances=args.instances,
                          use_expert_feats=args.use_additional_features, reward_type=args.reward_type)
    eval_env = SAT4JEnvSelHeur(host='', port=args.port + 1, time_step_limit=args.env_max_steps, work_dir=eval_dir,
                               jar_path=args.sat4j_jar_path,
                               instances=args.instances if args.val_instances is None else args.val_instances,
                               use_expert_feats=args.use_additional_features, reward_type=args.reward_type)
    # Setup agent
    state_dim = env.observation_space.shape[0]
    if not args.only_control_bumper:
        action_dim = np.prod(env.action_space.nvec)
        facts = env.action_space.nvec
    else:
        action_dim = env.action_space.nvec[0]
        facts = [env.action_space.nvec[0], 1]

    if args.validate_type == 'train':
        agent = DQN(state_dim, action_dim, gamma=0.99, env=env, eval_env=eval_env, train_eval_env=env,
                    factored_action_space=facts, do_checkpoint=not args.direct_evaluation)
    else:
        agent = DQN(state_dim, action_dim, gamma=0.99, env=env, eval_env=eval_env,
                    factored_action_space=facts, do_checkpoint=not args.direct_evaluation)

    episodes = args.episodes
    max_env_time_steps = args.env_max_steps
    epsilon = args.epsilon

    if args.val_instances is not None:
        num_eval_episodes = len(args.val_instances)
    else:
        num_eval_episodes = len(args.instances)
    if num_eval_episodes == 1: num_eval_episodes = 3

    if args.validate is not None:

        outfile_dir_name = os.path.join(os.path.dirname(args.validate),
                                        'eval_scores.json' if args.validate_type != 'train' else 'train_scores.json')
        skip_checkpoints = []
        if os.path.exists(outfile_dir_name):
            with open(outfile_dir_name, 'r') as infh:
                for line in infh:
                    data = json.loads(line)
                    skip_checkpoints.append(data['checkpoint_path'])
        with open(args.validate, 'r') as infh, open(outfile_dir_name, 'a+', buffering=1) as outfh:
            for line in infh:
                data = json.loads(line)
                if data['checkpoint_path'] in skip_checkpoints:
                    print(f'Already validated {data["checkpoint_path"]} -> skipping')
                else:
                    print(f'Validating {data["checkpoint_path"]}')
                    agent.load(data['checkpoint_path'])
                    eval_s, eval_r, eval_d = agent.eval(num_eval_episodes, max_env_time_steps,
                                                        train_set=args.validate_type == 'train')
                    data['avg_num_steps_per_eval_ep'] = float(np.mean(eval_s))
                    data['avg_num_decs_per_eval_ep'] = float(np.mean(eval_d))
                    data['avg_rew_per_eval_ep'] = float(np.mean(eval_r))
                    data['std_rew_per_eval_ep'] = float(np.mean(eval_r))
                    data['reward_per_insts'] = eval_r
                    data['steps_per_insts'] = eval_s
                    if args.validate_type != 'train':
                        insts_looked_at = []
                        counter = 0
                        for index in range(len(agent._eval_env.instances)):
                            while (index + counter) % len(agent._eval_env.instances) in agent._eval_env.too_simple_inst:
                                counter += 1
                            actual_index = (index + counter) % len(agent._eval_env.instances)
                            insts_looked_at.append(agent._eval_env.instances[actual_index])
                        data['eval_insts'] = insts_looked_at,
                    else:
                        insts_looked_at = []
                        counter = 0
                        for index in range(len(agent._train_eval_env.instances)):
                            while (index + counter) % len(agent._train_eval_env.instances) in agent._train_eval_env.too_simple_inst:
                                counter += 1
                            actual_index = (index + counter) % len(agent._train_eval_env.instances)
                            insts_looked_at.append(agent._train_eval_env.instances[actual_index])
                        data['eval_insts'] = insts_looked_at,
                    json.dump(data, outfh)
                    outfh.write('\n')
    elif args.load_model is None:
        print('#' * 80)
        print(f'Using agent type "{agent}" to learn')
        print('#' * 80)
        agent.train(episodes, max_env_time_steps, epsilon, num_eval_episodes, args.eval_after_n_steps,
                    max_train_time_steps=args.training_steps)
        os.mkdir(os.path.join(out_dir, 'final'))
        agent.save_model(os.path.join(out_dir, 'final'))
        agent.save_rpb(os.path.join(out_dir, 'final'))
    else:
        print('#' * 80)
        print(f'Loading {agent} from {args.load_model}')
        print('#' * 80)
        agent.load(args.load_model)
        steps, rewards, decisions = agent.eval(1, 100000)
        np.save(os.path.join(out_dir, 'eval_results.npy'), [steps, rewards, decisions])
