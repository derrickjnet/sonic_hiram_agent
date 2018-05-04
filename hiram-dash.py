#!/usr/bin/env python
# Human Preferences for
# DQN Lib: https://github.com/keon/deep-q-learning/blob/master/dqn.py
# Play Lib: https://raw.githubusercontent.com/openai/gym/master/gym/utils/play.py
# DL from Human Pref: https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/
# DL from Human Pref: https://arxiv.org/abs/1706.03741
# Implicit Imitation: https://www.aaai.org/Papers/JAIR/Vol19/JAIR-1916.pdf

# ENV_LOCAL
from retro_contest.local import make
# ENV_REMOTE
import gym_remote.client as grc
import gym_remote.exceptions as gre
# ENV_PLUS
import gym
import sqlite3
import random
import time
# FIFO
from collections import deque
# ML/RL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage as ndi
from skimage import feature
from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
#Load ML
# trained_model = load_ml_model('reward.ml')

# GLOBALS
local_env = True
EMA_RATE = 0.2
EXPLOIT_BIAS = 0.15
TOTAL_TIMESTEPS = int(1e6)
completion = 9000  # Estimated End
batches = 4
# References
# https://github.com/keon/deep-q-learning/blob/master/dqn.py


# Create Storage
conn = sqlite3.connect('retro.db')
db = conn.cursor()

# Setup Storage
stats_col = ["episode", "steps", "cur_action","prev_action","acts1", "acts3", "acts5", "acts7", "acts9", "acts11"
    , "acts33", "safety", "esteem", "belonging", "potential", "human","total_reward"]
df = pd.DataFrame(columns=stats_col)
df.to_sql('game_stats', conn, if_exists='replace')

sarsa_col = ["cluster","points","esteem"]
sarsa = pd.DataFrame(columns=sarsa_col)
sarsa.to_sql('sarsa', conn, if_exists='replace')

def main():

    if local_env: #Select Random Level if local
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
        env = make(game='SonicTheHedgehog-Genesis', state=levels[random.randrange(0, 13, 1)])
    else:
        env = grc.RemoteEnv('tmp/sock')
    env = TrackedEnv(env)
    env.reset()  # Initialize Gaming Environment
    new_ep = True  # New Episode Flag
    solutions = []  # Track Solutions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(state_size, action_size)
    agent = DQNAgent(state_size, action_size)  # Create DQN Agent
    env.play = False
    env.trainer = False  # Begin with mentor led exploration
    env.resume_rl(False)  # Begin with RL exploration

    while env.total_steps_ever <= TOTAL_TIMESTEPS:  # Interact with Retro environment until Total TimeSteps expire.

        while env.trainer:
            keys = getch()
            if keys == 'A':
                env.step(env.control(-1))
            if keys == 'B':
                env.step(env.control(4))
            if keys == 'C':
                env.step(env.control(3))
            if keys == 'D':
                env.step(env.control(2))
            if keys == ' ':
                env.step(env.control(5))
            print('Entering Self Play')
            time.sleep(25.0 / 1000.0)

        if new_ep:  # If new episode....
            if env.rl:  # Try RL exploration
                if not env.resume:  # Resume RL exploration
                    state = env.reset()
                else:
                    state = env.step(env.control())
                done = False
                while not done:
                    action = agent.act(state, env)
                    next_state, reward, done, _ = env.step(action)
                    reward = reward if not done else -10
                    # next_state = np.reshape(next_state, [1, state_size])
                    agent.remember(state, action, reward, next_state, done, env)
                    state = next_state
                env.resume_rl(False)
                continue
            #Else JERK replay
            if (solutions and
                    random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                best_pair = solutions[-1]
                new_rew = exploit(env, best_pair[1], agent)
                best_pair[0].append(new_rew)
                print('replayed best with reward %f' % new_rew)
                print(best_pair[0])
                continue
            else:
                env.reset()
                new_ep = False
        #Else JERK play
        rew, new_ep = move(env, 100)
        if not new_ep and rew <= 0:
            env.resume_rl()
            # print('backtracking due to negative reward: %f' % rew)
            # _, new_ep = move(env, 70, left=True)
        data = pd.read_sql_query("SELECT * FROM game_stats", conn)
        #print(data[-5:])
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))
            env.episode += 1
    agent.save('retro.model')


# Enable Keyboard
def getch():
    import sys, tty, termios
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

def capture_moment(env,frames,done):
    if done:
        env.in_danger_pos.append(env.total_reward)
        img = env.observation_history[-1]
        show_moment(img)
    elif env.total_reward > max(env.is_stuck_pos):
        env.is_stuck_pos.append(env.total_reward)
        img = env.observation_history[-1]
        show_moment(img)
        # img = env.observation_history[frames+5]
        # show_moment(img)

def show_moment(img):
    plt.imshow(img)
    plt.show()


def exploit(env, sequence, agent):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    state = env.reset()
    done = False
    idx = 0
    env.rl = False
    while not done:
        if idx >= len(sequence):
            env.resume_rl()
            done = True
        else:
            new_state, rew, done, _ = env.step(sequence[idx])
            agent.remember(state, sequence[idx], rew, new_state, done, env)  # Train agent from best sequence
            state = new_state
        idx += 1
    # agent.replay(batches)
    env.total_replays += 1
    env.replay_reward_history.append(env.total_reward)
    return env.total_reward


# DEFINE AGENT
class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_shape=(1, 224, 320), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, env):
        self.memory.append((state, action, reward, next_state, done,
                            env.safety, env.esteem, env.belonging, env.potential))

    def act(self, state, env):
        if np.random.rand() <= self.epsilon:
            return env.control()  # Fix
        print('RL Move')
        act_values = self.model.predict(state)
        print(act_values)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        print('Replaying')
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done, _, _, _, _ in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            print(target_f)
            target_f[0][action] = target_f
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


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
        self.action_space = gym.spaces.Discrete(len(self._actions))


        #Toggles
        self.play = True #Allow human trainer
        self.trainer = False #Enables and tracks training
        self.rl = False #Enables RL exploration
        self.resume = False #Resumes RL exploration
        #Env Counters
        self.episode = 1
        self.total_steps_ever = 0
        self.total_reward = 0
        self.rl_total_reward = 0
        self.total_replays = 0
        #Position Counters
        self._cur_x = 0
        self._max_x = 0
        #Scene Trackers
        self.action_history = []
        self.reward_history = []
        self.seq_replay = []
        self.rl_reward_history = []
        self.replay_reward_history = []
        #Maslow Trackers
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
        #Storage
        self.table = pd.read_sql_query("SELECT * from game_stats", conn)





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

    # pylint: disable=E0202
    def reset(self, **kwargs):
        self.action_history = []
        self.reward_history = []
        self._cur_x = 0
        self._max_x = 0
        self.total_reward = 0
        self.steps = 0
        return self.env.reset(**kwargs)

    def resume_rl(self, a=True):
        self.rl = a
        self.resume = a

    def maslow(self, done, info):
        if (self.total_reward >= completion) and done:
            self.potential = True
        else:
            self.potential = False
        if not done:
            self.survival = True
        if int(self.reward_history[-2]) < int(self.reward_history[-1]):
            self.esteem = True
            if self._cur_x == self._max_x: #If rewards are progressing have the computer resume.
                self.trainer = False
        else:  # If esteem is low (i.e, stuck, in time loop, new scenario ask for help)
            if self.play:
                if not done:
                    self.trainer = True
            else:
                self.esteem = False
                if self._cur_x == self._max_x:
                    self.trainer = False
                self.resume_rl()
                self.is_stuck_pos.append(self.total_reward)
        if info['rings'] > 0:
            self.safety = True
        else:
            self.safety = False

    def process_frames(self, obs):
        x_t1 = skimage.color.rgb2gray(obs)
        # x_t1 = skimage.transform.resize(x_t1, (84, 84))
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        return x_t1

    # PROCESSING
    def analytics(self, a):
        acts = np.array(self.rl_reward_history)
        acts = np.sum(acts[-a:])
        acts = np.round(acts, 4)
        return acts

    def stats(self):
        # tm = trained_model.predict(self.table[-1])
        cur_action = str(self.action_history[-1])
        if self.steps > 5:
            prev_action = str(self.action_history[-2])
            cluster_action = str(self.action_history[-5:])
            db.execute("INSERT INTO sarsa VALUES (NULL,?, ?,?)",
                       (cluster_action,self.analytics(5),self.esteem))
            conn.commit()
        else: prev_action = ''
        acts1 = self.analytics(1)
        acts3 = self.analytics(3)
        acts5 = self.analytics(5)
        acts7 = self.analytics(7)
        acts9 = self.analytics(9)
        acts11 = self.analytics(11)
        acts33 = self.analytics(33)
        db.execute("INSERT INTO game_stats VALUES (NULL,?, ?, ?,?,?, ?,?,?,?,?,?,?,?,?,?,?,?)",
                   (self.episode, self.steps, cur_action, prev_action,acts1, acts3, acts5, acts7, acts9, acts11, acts33
                    , int(self.safety), int(self.esteem)
                    , int(self.belonging), int(self.potential), int(self.trainer),int(self.total_reward)))
        conn.commit()
        # return tm

    def control(self, a=None):  # Enable Disrete Actions pylint: disable=W0221
        if not a:
            a = self.action_space.sample()
        return self._actions[a].copy()

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.rl_reward_history.append(rew) #True Reward
        obs = self.process_frames(obs)
        self.total_reward += rew
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x) #Net 0 reward
        self._max_x = max(self._max_x, self._cur_x) #Max level reached
        self.reward_history.append(self.total_reward)
        if self.steps > 1:
            self.maslow(done, info)
        self.steps += 1
        self.stats()
        self.render()
        return obs, rew, done, info

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)