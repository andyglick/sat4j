import socket
import time
import typing
from copy import deepcopy
from enum import Enum
import os
from typing import Union
import json
import subprocess

import numpy as np
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
from gym.utils import seeding


class StateType(Enum):
    """Class to define numbers for state types"""
    RAW = 1
    DIFF = 2
    ABSDIFF = 3
    NORMAL = 4
    NORMDIFF = 5
    NORMABSDIFF = 6


class SAT4JEnvSelHeur(Env):

    def __init__(self, host: str = '', port: int = 12345,
                 num_steps=None, state_type: Union[int, StateType] = StateType.DIFF,
                 seed: int = 12345, max_rand_steps: int = 0, config_dir: str = '.',
                 port_file_id=None, external: bool = False,
                 time_step_limit: int = -1, work_dir: str = '.',
                 instance: str = None, jar_path: str = 'dist/CUSTOM/sat4j-kth.jar',
                 instances: list = None,
                 use_expert_feats: bool = False,
                 reward_type: str = 'control_steps'):
        """
        Initialize environment
        """

        self._heuristic_state_features = ["bumper", "bumpStrategy", "time", "decisions", "depth", "decisionLevel"]
        self._expert_features = ["numberOfVariables", "numberOfOriginalConstraints",
                                 "org.sat4j.pb.constraints.pb.OriginalBinaryClausePBOriginal",
                                 "org.sat4j.pb.constraints.pb.MinWatchCardPBOriginal",
                                 "org.sat4j.pb.constraints.pb.LearntHTClausePBLearned",
                                 "org.sat4j.pb.constraints.pb.MaxWatchPbLearned",
                                 "org.sat4j.pb.constraints.pb.MinWatchCardPBLearned",
                                 "propagation", "restarts", "reduceddb"]
        if use_expert_feats:
            self._heuristic_state_features += self._expert_features

        bumpers = ["ANY", "ASSIGNED", "FALSIFIED",
                   "FALSIFIED_AND_PROPAGATED", "EFFECTIVE", "EFFECTIVE_AND_PROPAGATED"]
        bumpStrategies = ["ALWAYS_ONE", "DEGREE",
                          "COEFFICIENT", "RATIO_DC", "RATIO_CD"]
        self.action_space = MultiDiscrete([len(bumpers), len(bumpStrategies)])
        self.index_action_map = {0: {i: v for i, v in enumerate(bumpers)},
                                 1: {j: w for j, w in enumerate(bumpStrategies)}}
        self.action_index_map = {0: {v: i for i, v in enumerate(bumpers)},
                                 1: {w: j for j, w in enumerate(bumpStrategies)}}

        total_state_features = (1 * len(self._heuristic_state_features))
        # TODO should be fixed for any other method than DDQN
        self.observation_space = Box(
            low=np.array([-np.inf for _ in range(total_state_features)]),
            high=np.array([np.inf for _ in range(total_state_features)]),
            dtype=np.float32
        )

        self.__skip_transform = [True, True, False, False, False, False]
        self.__skip_transform_expert_features = [True, True, True, True, False, False, False, False, False, True]
        if use_expert_feats:
            self.__skip_transform += self.__skip_transform_expert_features

        self.host = host
        self.port = port

        self.socket = None
        self.conn = None

        self._prev_state = None
        self.num_steps = num_steps
        self.time_step_limit = time_step_limit

        self.__state_type = StateType(state_type)
        self.__norm_vals = []
        self._config_dir = config_dir
        self._port_file_id = port_file_id

        self._transformation_func = None
        # create state transformation function with inputs (current state, previous state, normalization values)
        if self.__state_type == StateType.DIFF:
            self._transformation_func = lambda x, y, z, skip: x - y if not skip else x
        elif self.__state_type == StateType.ABSDIFF:
            self._transformation_func = lambda x, y, z, skip: abs(x - y) if not skip else x
        elif self.__state_type == StateType.NORMAL:
            self._transformation_func = lambda x, y, z, skip: SAT4JEnvSelHeur._save_div(x, z) if not skip else x
        elif self.__state_type == StateType.NORMDIFF:
            self._transformation_func = lambda x, y, z, skip: \
                SAT4JEnvSelHeur._save_div(x, z) - SAT4JEnvSelHeur._save_div(y, z) if not skip else x
        elif self.__state_type == StateType.NORMABSDIFF:
            self._transformation_func = lambda x, y, z, skip: \
                abs(SAT4JEnvSelHeur._save_div(x, z) - SAT4JEnvSelHeur._save_div(y, z)) if not skip else x

        self.rng = np.random.RandomState(seed=seed)
        self.max_rand_steps = max_rand_steps
        self.__step = 0
        self.__start_time = None
        self.done = True  # Starts as true as the expected behavior is that before normal resets an episode was done.
        self.sat4j = None
        self.__external = external
        self.wd = work_dir
        self.instance = instance
        self.jarpath = jar_path
        self.instances = instances
        self._inst_pointer = 0
        self.__reward_type = reward_type

    @staticmethod
    def _save_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def send_msg(self, msg: bytes):
        """
        Send message and prepend the message size

        Based on comment from SO see [1]
        [1] https://stackoverflow.com/a/17668009

        :param msg: The message as byte
        """
        # Prefix each message with a 4-byte length (network byte order)
        msg = str.encode("{:>04d}".format(len(msg))) + msg
        self.conn.sendall(msg)

    def recv_msg(self):
        """
        Recieve a whole message. The message has to be prepended with its total size
        Based on comment from SO see [1]
        """
        # TODO maybe we want to send message lengths again
        # # Read message length and unpack it into an integer
        raw_msglen = self.recvall(4)
        if not raw_msglen:
            return None
        msglen = int(raw_msglen.decode())
        # Read the message data
        return self.recvall(msglen)

    def recvall(self, n: int):
        """
        Given we know the size we want to recieve, we can recieve that amount of bytes.
        Based on comment from SO see [1]

        :param n: Number of bytes to expect in the data
        """
        # Helper function to recv n bytes or return None if EOF is hit
        data = b''
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _process_data(self):
        """
        Split received json into state reward and done
        :return:
        """
        msg = self.recv_msg()
        if len(msg) == 0:
            raise ConnectionError('Empty message received')
        else:
            msg = msg.decode('utf-8')
        done = msg.strip() == 'done'
        if done:  # Done flag is sent seperately from final state so we have to query again.
            msg = self.recv_msg().decode('utf-8')
        data = eval(msg)
        if self.__reward_type == 'time':
            r = self._last_time - time.time()
        elif self.__reward_type == 'time_proxy':
            r = data['timeProxy'] / 10 ** 9
        else:
            r = -1 / self.time_step_limit
        self._last_time = time.time()

        state = []

        for feature in self._heuristic_state_features:
            state.append(data[f"{feature}"])

        state[0] = self.action_index_map[0][state[0]]
        state[1] = self.action_index_map[1][state[1]]
        if self._prev_state is None:
            self.__norm_vals = deepcopy(state)
            self._prev_state = deepcopy(state)
        if self.__state_type != StateType.RAW:  # Transform state to DIFF state or normalize
            state = list(map(self._transformation_func, state, self._prev_state, self.__norm_vals,
                             self.__skip_transform))
        self._prev_state = state
        return np.array(state, dtype=np.float32), r, done

    def step(self, action: typing.Union[int, typing.List[int]]):
        """
        Play RL-Action
        :param action:
        :return:
        """
        self.__step += 1
        assert len(action) == 2, f"Expected a pair of actions got {len(action)}"
        msg = {"bumper": self.index_action_map[0][action[0]],
               "bumpStrategy": self.index_action_map[1][action[1]]}
        if not self.conn:
            self.close()
            raise Exception('Connection unexpected closed')
        self.conn.sendall(json.dumps(msg).encode('utf-8') + "\n".encode('utf-8'))
        s, r, d = self._process_data()
        info = {}
        if d:
            self.done = True
            self.kill_connection()
        if 0 < self.time_step_limit < self.__step:
            d = True
            info['needs_reset'] = True
            self.kill_connection()
            self.done = True
        return s, r, d, info

    def reset(self):
        """
        Initialize SAT4J
        :return:
        """
        self.done = False
        self._prev_state = None
        self.__step = 0
        self.__start_time = time.time()
        self.kill_connection()
        if not self.socket:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(10)
            self.socket.bind((self.host, self.port))

        self.socket.listen()

        if self.instances is not None:
            self.instance = self.instances[self._inst_pointer]
            self._inst_pointer = (self._inst_pointer + 1) % len(self.instances)
            print(f'Using instance {self.instance}')

        if not self.__external:
            command = [
                'java',
                '-jar',
                f'{self.jarpath}',
                '-port',
                f'{self.port}',
                '-br',
                'externaldac',
                '-sync',
                f'{self.instance}'
            ]
            with open(os.path.join(self.wd, 'sat4j.out'), 'a+') as fout, \
                    open(os.path.join(self.wd, 'sat4j.err'), 'a+') as ferr:
                self.sat4j = subprocess.Popen(command, stdout=fout, stderr=ferr)

        self.conn, address = self.socket.accept()

        self.conn.sendall("START\n".encode('utf-8'))
        tmp_msg = self.recv_msg().decode('utf-8')
        if tmp_msg.strip() != 'CONFIRM':
            print(tmp_msg, tmp_msg.strip())
            raise ConnectionAbortedError('Could not establish start procedure')

        print('connection established')
        print('Waiting on initial state')
        self._last_time = time.time()
        s, _, _ = self._process_data()
        self.conn.sendall("CONFIRM\n".encode('utf-8'))
        print('Received initial state')
        # do a default first step (if we have instance features we don't need this but without we do
        s, _, _, _ = self.step([0, 0])

        return s

    def kill_connection(self):
        """Kill the connection"""
        count = 0
        if self.conn:
            tmp_msg = 'NONE'
            try:
                while tmp_msg.strip() != 'CONFIRM':
                    self.conn.sendall("END\n".encode('utf-8'))
                    self.conn.send("\n".encode('utf-8'))
                    tmp_msg = self.recv_msg().decode('utf-8')
                    count += 1
                    if count >= 10: break
            # the errors can happen if sat4j already closed the connection after finding a solution
            except (BrokenPipeError, ConnectionResetError, AttributeError):
                self.conn.close()
                self.conn = None
                pass
            else:
                self.conn.shutdown(2)
                self.conn.close()
                self.conn = None
        if self.socket:
            self.socket.shutdown(2)
            self.socket.close()
            self.socket = None
        if self.sat4j:
            self.sat4j.terminate()
        if count >= 10:
            raise OSError('Could not confirm shutdown of SAT4J')

    def close(self):
        """
        Needs to "kill" the environment
        :return:
        """
        self.kill_connection()

    def render(self, mode: str = 'human') -> None:
        """
        Required by gym.Env but not implemented
        :param mode:
        :return: None
        """
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':
    """
    Only for debugging purposes
    """
    import sys

    HOST = ''  # The server's hostname or IP address
    PORT = int(sys.argv[1])  # The port used by the server
    if len(sys.argv) == 4:
        a0 = int(sys.argv[2])
        a1 = int(sys.argv[3])
    else:
        a0 = a1 = None

    dir_path = os.path.dirname(os.path.realpath(__file__))
    env = SAT4JEnvSelHeur(host=HOST, port=PORT, time_step_limit=100,
                          instance=os.path.join(dir_path, 'normalized-ECrand4regsplit-v030-n1.opb.bz2'),
                          jar_path=os.path.join(dir_path, 'dist/CUSTOM/sat4j-kth.jar'))
    rs = []
    try:
        for i in range(10):
            print('-' * 100)
            s = env.reset()
            # print(s)
            done = False
            r_ = 0
            while not done:
                if a0 is not None:
                    a = [a0, a1]
                else:
                    a = [np.random.randint(6), np.random.randint(5)]
                # print(a)
                s, r, done, _ = env.step(a)
                r_ += r
                # print('#'*10)
                # print(s, r, done)
                # print('#'*10)
            print(i, r_)
            rs.append(r_)
    except Exception as e:
        env.close()
        raise e
    finally:
        print('Closing Env')
        env.close()
        print(f'Actions: [{a0}, {a1}] | Average reward: {np.mean(rs)}')
