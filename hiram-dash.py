#!/usr/bin/env python
# Human Preferences for
# DQN Lib: https://github.com/keon/deep-q-learning/blob/master/dqn.py
# Play Lib: https://raw.githubusercontent.com/openai/gym/master/gym/utils/play.py
# DL from Human Pref: https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/
# DL from Human Pref: https://arxiv.org/abs/1706.03741
# Implicit Imitation: https://www.aaai.org/Papers/JAIR/Vol19/JAIR-1916.pdf
# Note try Replaying top 3 attempts
# ENV_LOCAL
local_env = True
render = False
import random
import sqlite3
import time
# FIFO
from collections import deque

# ENV_PLUS
import gym
# ENV_REMOTE
import gym_remote.client as grc
import gym_remote.exceptions as gre
import numpy as np
# ML/RL
import pandas as pd
import tensorflow as tf
from anyrl.algos import DQN
from anyrl.models import MLPQNetwork, EpsGreedyQNetwork, rainbow_models
from anyrl.rollouts import BasicPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
from graphdb import GraphDB

# Set Variables
seed = 123
done_penalty = -10
np.random.seed(seed)
EXPLOIT_BIAS = 0.2
RL_PLAY_PCT = 2  # 2
TOTAL_TIMESTEPS = int(1e6)
COMPLETION = 9000  # Estimated End
batches = 4
TRAINING_STEPS = 4500  # 20000
BUFFER_SIZE = 1024
MIN_BUFFER_SIZE = 256
STEPS_PER_UPDATE = 3
ITERS_PER_LOG = 200
BATCH_SIZE = 64
EPSILON = 0.1
LEARNING_RATE = 0.0000625  # 0000625
# References
# https://github.com/keon/deep-q-learning/blob/master/dqn.py

print(seed, RL_PLAY_PCT, TRAINING_STEPS, LEARNING_RATE)

# Create Storage
conn = sqlite3.connect('retro.db')
db_method = 'replace'
db = conn.cursor()
gb = GraphDB('graph1.db')

# Setup Storage
stats_col = ["level", "episode", "steps", 'curr_pos', "curr_action", "prev_action", "acts1", "acts3", "acts5", "acts7",
             "acts9",
             "acts11"
    , "acts33", "safety", "esteem", "belonging", "potential", "human", "total_reward"]
df = pd.DataFrame(columns=stats_col)
df.to_sql('game_stats', conn, if_exists=db_method)

sarsa_col = ["level", "action_cluster", "rewards", "last_reward", "esteem"]
sarsa = pd.DataFrame(columns=sarsa_col)
sarsa.to_sql('sarsa', conn, if_exists=db_method)


def main():
    if local_env:  # Select Random Level if local
        from retro_contest.local import make
        levels = ['SpringYardZone.Act3',
                  'SpringYardZone.Act2',
                  'GreenHillZone.Act3',
                  'GreenHillZone.Act1',
                  'StarLightZone.Act2',
                  'StarLightZone.Act1',
                  'MarbleZone.Act2',
                  'MarbleZone.Act1',
                  'MarbleZone.Act3',
                  'ScrapBrainZone.Act2',
                  'LabyrinthZone.Act2',
                  'LabyrinthZone.Act1',
                  'LabyrinthZone.Act3']
        level_choice = levels[random.randrange(0, 13, 1)]
        env = make(game='SonicTheHedgehog-Genesis', state=level_choice)
    else:
        print('connecting to remote environment')
        env = grc.RemoteEnv('tmp/sock')
        print('starting episode')
    # env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    env = TrackedEnv(env)

    new_ep = True  # New Episode Flag
    solutions = env.solutions  # Track Solutions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(state_size, action_size)
    env.assist = False
    env.trainer = False  # Begin with mentor led exploration
    env.reset()  # Initialize Gaming Environment
    while env.total_steps_ever <= TOTAL_TIMESTEPS:  # Interact with Retro environment until Total TimeSteps expire.
        while env.trainer:
            print('Entering Self Play')
            env.is_done()
            keys = getch()
            if keys == 'A':
                env.control(-1)
            if keys == 'B':
                env.control(4)
            if keys == 'C':
                env.control(3)
            if keys == 'D':
                env.control(2)
            if keys == 'rr':
                env.trainer = False
                continue
            if keys == ' ':
                env.close()
                env = make(game='SonicTheHedgehog-Genesis', state=levels[random.randrange(0, 13, 1)])
                env = TrackedEnv(env)
                env.reset()  # Initialize Gaming Environment

        if env.agent == 'DQN':
            tf.reset_default_graph()  # Initialize TF
            print('Entering DQN')
            with tf.Session() as sess:
                def make_net(name, reuse=tf.AUTO_REUSE):
                    return MLPQNetwork(sess,
                                       env.action_space.n,
                                       gym_space_vectorizer(env.observation_space),
                                       name,
                                       layer_sizes=[32])

                env.agent = 'DQN'
                dqn = DQN(make_net('online'), make_net('target'))
                bplayer = BasicPlayer(env, EpsGreedyQNetwork(dqn.online_net, EPSILON),
                                      batch_size=STEPS_PER_UPDATE)
                optimize = dqn.optimize(learning_rate=LEARNING_RATE)

                sess.run(tf.global_variables_initializer())

                dqn.train(num_steps=TRAINING_STEPS,
                          player=bplayer,
                          replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                          optimize_op=optimize,
                          target_interval=200,
                          batch_size=64,
                          min_buffer_size=200,
                          handle_ep=lambda _, rew: print('Exited DQN with : ' + str(rew) + str(env.steps)))
                sess.close()

        if env.agent == 'Rainbow':
            tf.reset_default_graph()  # Initialize TF
            print('Entering Rainbow')
            with tf.Session() as sess:
                env.agent = 'Rainbow'
                dqn2 = DQN(*rainbow_models(sess,
                                           env.action_space.n,
                                           gym_space_vectorizer(env.observation_space),
                                           min_val=-200,
                                           max_val=200))
                player = NStepPlayer(BasicPlayer(env, dqn2.online_net), STEPS_PER_UPDATE)
                optimize = dqn2.optimize(learning_rate=1e-4)

                sess.run(tf.global_variables_initializer())

                dqn2.train(num_steps=TRAINING_STEPS,  # 2000000,  # Make sure an exception arrives before we stop.
                           player=player,
                           replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                           optimize_op=optimize,
                           train_interval=1,
                           target_interval=8192,
                           batch_size=32,
                           min_buffer_size=20000)
                sess.close()

        if env.agent == 'Replay' and env.episode >= 7:
            env.agent = 'Replay'
            solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
            best_pair = solutions[-1]
            new_rew = exploit(env, best_pair[1]) #Remove Reset
            best_pair[0].append(new_rew)
            print('replayed best with reward %f' % new_rew)
            print(best_pair[0])
            env.is_done()


        if env.agent == 'JERK':
            done = False
            env.agent = 'JERK'
            while not done:
                rew, done = move(env, 100)
                if not done and rew <= 0:
                    _, done = move(env, 70, left=True)
                if done:
                    env.reset()


def getch():  # Enable Keyboard
    import sys, termios
    old_settings = termios.tcgetattr(0)
    new_settings = old_settings[:]
    new_settings[3] &= ~termios.ICANON
    try:
        termios.tcsetattr(0, termios.TCSANOW, new_settings)
        ch = sys.stdin.read(1)
        print(ch)
    finally:
        termios.tcsetattr(0, termios.TCSANOW, old_settings)

    return ch


def move(env, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=4):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    total_rew = 0.0
    done = False
    steps_taken = 0
    jumping_steps_left = 0
    while not done and steps_taken < num_steps:
        if left:
            action = env.control(0)
        else:
            action = env.control(1)
        if jumping_steps_left > 0:
            action = env.control(6)
            jumping_steps_left -= 1
        else:
            if random.random() < jump_prob:
                jumping_steps_left = jump_repeat - 1
                action = env.control(6)
        _, rew, done, _ = env.step(action)
        total_rew += rew
        steps_taken += 1
        if done:
            break
    return total_rew, done


def exploit(env, sequence):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    env.agent = 'Replay'
    done = False
    sequence2 = env.best_game #bypass normal replay
    idx = 0

    # Logic for replaying game
    # rewards = gb('rewards').game_rewards(list)
    # sequence = gb(9779.833488464355).game_sequences(list)
    # df = pd.DataFrame(sequence).T
    # print(df.iloc[0][0])

    while not done:
        if idx >= len(sequence):
            move(env, 15)
            env.control()
        else:
            new_state, rew, done, _ = env.step(sequence[idx])
        idx += 1
    env.total_replays += 1
    env.replay_reward_history.append(env.total_reward)
    return env.total_reward


#DEFINE GYM
class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_choices = self._actions
        self.action_space = gym.spaces.Discrete(len(self._actions))

        self.level_choice = ''
        # Toggles
        self.agent = ''
        self.assist = False  # Allow human trainer
        self.trainer = False  # Enables and tracks training
        self.rl = False  # Enables RL exploration
        self.resume = False  # Resumes RL exploration
        # Env Counters
        self.episode = 0
        self.total_steps_ever = 0
        self.steps = 0
        self.total_reward = 0
        self.rl_total_reward = 0
        self.total_replays = 0
        # Position Counters
        self.curr_loc = 0
        self._max_x = 0
        self.done = False
        # Scene Trackers
        self.last_obs = []
        self.solutions = []
        self.action_history = []
        self.reward_history = []
        self.seq_replay = []
        self.step_rew_history = []
        self.replay_reward_history = []
        self.best_game = []
        self.jerk = deque(maxlen=2000)
        self.dqn = deque(maxlen=2000)
        self.rainbow = deque(maxlen=2000)
        # Maslow Trackers
        self.survival = False  # Tracks done, seeks to prevent done unless potential reached
        self.safety = False  # Tracks rings, seeks to prevent ring lose
        self.belonging = False  # Tracks collision, seeks to avoid bad friends
        self.esteem = False  # Tracks reward, seeks
        self.potential = False  # Tracks completion || reward > 9000
        #Queues
        self.memory = deque(maxlen=2000)
        self.is_stuck_pos = deque(maxlen=2000)
        self.is_stuck_pos.append(0)
        self.in_danger_pos = deque(maxlen=2000)
        self.in_danger_pos.append(0)
        self.track_evolution = deque(maxlen=2000)
        # Storage
        self.table = pd.read_sql_query("SELECT * from game_stats", conn)

    def is_done(self):
        if self.done:
            self.reset()

    def best_sequence(self):
        """
        Get the prefix of the trajectory with the best
        cumulative reward.
        """
        max_cumulative = max(self.reward_history)
        for i, rew in enumerate(self.reward_history):
            if rew == max_cumulative:
                return self.action_history[:i+1]
        raise RuntimeError('unreachable')

    def select_agent(self):
        foo = ['DQN', 'Rainbow', 'JERK', 'Replay']
        agent = random.choice(foo) if self.episode > 1 else 'JERK'
        return agent

    # pylint: disable=E0202
    def reset(self, **kwargs):
        print('Episode', self.episode, self.agent, self.steps, self.total_reward)
        self.action_history = []
        self.reward_history = []
        self.curr_loc = 0
        self._max_x = 0
        self.total_reward = 0
        self.steps = 0
        self.episode += 1
        rewards = gb('rewards').game_rewards(list)
        if rewards:
            self.best_game = gb(max(rewards)).game_sequences(list)
        self.agent = self.select_agent()
        return self.env.reset(**kwargs)

    def maslow(self, done, info):
        if (self.total_reward >= COMPLETION) and done:
            self.potential = True
        else:
            self.potential = False
        if not done:
            self.survival = True
        if (int(self.reward_history[-2]) <= int(self.reward_history[-1])):
            self.esteem = True
        else:  # If esteem is low (i.e, stuck, in time loop, new scenario ask for help)
            if self.assist:
                if not done:
                    self.trainer = True
            else:
                self.esteem = False
                self.is_stuck_pos.append(self.total_reward)
        # if info['rings'] > 0:
        #     self.safety = True
        # else:
        #     self.safety = False

    def process_frames(self, obs):
        x_t1 = skimage.color.rgb2gray(obs)
        # x_t1 = skimage.transform.resize(x_t1, (84, 84))
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        return x_t1

    # PROCESSING
    def get_rew_hist(self, a):
        acts = np.array(self.step_rew_history)
        acts = np.sum(acts[-a:])
        acts = np.round(acts, 4)
        return acts

    def insert_stats(self, action, obs, rew, done, info, start_time, stop_time):
        # tm = trained_model.predict(self.table[-1])
        self.action_history.append(action.copy())
        self.step_rew_history.append(rew)  # Step Reward
        self.total_reward += rew
        self.curr_loc += rew
        rew = max(0, self.curr_loc - self._max_x)  # Net 0 reward
        self._max_x = max(self._max_x, self.curr_loc)  #Max level reached
        self.reward_history.append(self.total_reward)
        if self.agent == 'Rainbow':
            self.rainbow.append(self.total_reward)
        if self.agent == 'DQN':
            self.dqn.append(self.total_reward)
        if self.agent == 'JERK':
            self.jerk.append(self.total_reward)

        if self.steps > 1:
            self.maslow(done, info)
        self.steps += 1
        # Setup
        acts1 = self.get_rew_hist(1)
        acts3 = self.get_rew_hist(3)
        acts5 = self.get_rew_hist(5)
        acts7 = self.get_rew_hist(7)
        acts9 = self.get_rew_hist(9)
        acts11 = self.get_rew_hist(11)
        acts33 = self.get_rew_hist(33)
        ac = 5 if len(self.action_history) > 5 else len(self.action_history)
        if self.steps > 1:
            prev_loc = self.reward_history[-2]
            prev_action = str(self.action_history[-2])
            curr_action = str(self.action_history[-1])
            action_cluster = str(self.action_history[-ac:])
            gb.store_relation(self.total_reward, 'reached_from',
                              {'curr_action': curr_action, "prev_loc": prev_loc})
            db.execute("INSERT INTO sarsa VALUES (NULL,?,?, ?,?,?)",
                       (self.level_choice, action_cluster, self.get_rew_hist(5), self.get_rew_hist(1), self.esteem))
            conn.commit()
            db.execute("INSERT INTO game_stats VALUES (NULL,?,?, ?,?, ?,?,?, ?,?,?,?,?,?,?,?,?,?,?,?)",
                       (self.level_choice, self.episode, self.steps, prev_loc, curr_action, prev_action
                        , acts1, acts3, acts5, acts7, acts9, acts11, acts33
                        , int(self.safety), int(self.esteem), int(self.belonging), int(self.potential)
                        , int(self.trainer), int(self.total_reward)))
            conn.commit()

            if prev_loc < self.curr_loc:  # pos_rew
                gb.store_relation(int(prev_loc), 'has_action',
                                  {'curr_action': curr_action, 'curr_reward': acts1, 'start_time': start_time
                                      , 'stop_time': stop_time, 'agent': self.agent})
                gb.store_relation(int(prev_loc), 'is_before_chron',
                                  {'curr_loc': self.curr_loc, 'curr_action': curr_action, 'curr_reward': acts1,
                                   'start_time': start_time
                                      , 'stop_time': stop_time, 'agent': self.agent})
            if prev_loc == self.curr_loc:  # net_rew
                gb.store_relation('stuck', 'at_place_spatial',
                                  {'prev_loc': prev_loc, 'curr_action': curr_action, 'curr_reward': acts1,
                                   'curr_loc': self.curr_loc
                                      , 'start_time': start_time, 'stop_time': stop_time, 'agent': self.agent})
            if prev_loc <= 0 and self.curr_loc > 0:
                gb.store_relation('unstuck', 'at_place_spatial',
                                  {'prev_loc': prev_loc, 'curr_action': curr_action, 'curr_reward': acts1,
                                   'curr_loc': self.curr_loc
                                      , 'start_time': start_time, 'stop_time': stop_time, 'agent': self.agent})
            if done:
                gb.store_relation('act_of_god', 'at_place_spatial',
                                  {'prev_loc': prev_loc, 'curr_action': curr_action, 'curr_reward': acts1,
                                   'start_time': start_time
                                      , 'stop_time': stop_time, 'agent': self.agent})
                gb.store_relation(self.agent, 'is_done_at',
                                  {'prev_loc': prev_loc, 'total_reward': self.total_reward})
                gb.store_relation('rewards', 'game_rewards', max(self.reward_history))
                gb.store_relation(max(self.reward_history), 'game_sequences', self.best_sequence())

            # gb.store_relation(self.steps, 'is_during_chron', {'curr_action': curr_action, 'curr_reward': acts1})
            # gb.store_relation(self.steps, 'is_after_chron', {'curr_action': curr_action, 'curr_reward': acts1})
            # gb.store_relation(self.steps, 'best_micro_seq', {'curr_action': curr_action, 'curr_reward': acts1})
            # gb.store_relation(self.steps, 'has_advantage', {'curr_action': curr_action, 'curr_reward': acts1})
            # gb.store_relation(self.steps, 'has_disadvantage', {'curr_action': curr_action, 'curr_reward': acts1})
            # gb.store_relation(self.steps, 'resembles', {'curr_action': curr_action, 'curr_reward': acts1})
            # gb.store_relation(self.steps, 'contrasts', {'curr_action': curr_action, 'curr_reward': acts1})
            # gb.store_relation(self.steps, 'caused', {'curr_action': curr_action, 'curr_reward': acts1})
            # gb.store_relation(self.steps, 'solved', {'curr_action': curr_action, 'curr_reward': acts1})

        # return tm

    def control(self, a=None):  # Enable Disrete Actions pylint: disable=W0221
        if not a:
            a = self.action_space.sample()
        obs, rew, done, info = self.step(self._actions[a].copy())
        if done:
            self.reset()
        return self._actions[a].copy()

    def step(self, action, *args):
        test = np.array(action).ndim
        if test < 1:
            action = self._actions[action].copy()
        self.total_steps_ever += 1
        start_time = time.time()
        obs, rew, done, info = self.env.step(action)
        stop_time = time.time()
        self.last_obs = obs
        self.insert_stats(action, obs, rew, done, info, start_time, stop_time)
        rew = max(0, self.curr_loc - self._max_x) if self.agent != 'JERK' else rew
        if done:
            self.done = done
            self.solutions.append(([max(self.reward_history)], self.best_sequence()))
            if done_penalty:
                rew = done_penalty
        if render:
            self.render()
        return obs, rew, done, info

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)