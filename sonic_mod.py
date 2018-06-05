"""
Environments and wrappers for Sonic training.
#D. Johnson - Mod to Sonic_Utils
"""

import gym
import random
import time
import math
import numpy as np
import pandas as pd
import sqlite3
import gym_remote.client as grc
import cv2
import pytesseract
cv2.ocl.setUseOpenCL(False)
from gym import spaces
from graphdb import GraphDB
from collections import deque
from baselines.common.atari_wrappers import FrameStack
from sklearn.cluster import KMeans
# from sklearn.feature_extraction import image
# from skimage import data, io
# from matplotlib import pyplot as plt

from abc import ABC, abstractmethod
from collections import OrderedDict
import time

# Create Storage
conn = sqlite3.connect('retro.db')
db_method = 'append'
db = conn.cursor()
gb = GraphDB('graph.db')

#Load Storage
from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model
# trained_step = load_ml_model("next_step.dill")


# Setup Storage
stats_col = ["level", "episode", "steps",'curr_pos', "curr_action", "prev_action", "acts1", "acts3", "acts5", "acts7", "acts9",
             "acts11"
    , "penalty", "safety", "esteem", "belonging", "potential", "human", "total_reward"]
df = pd.DataFrame(columns=stats_col)
df.to_sql('game_stats', conn, if_exists=db_method)

sarsa_col = ["level", "action_cluster", "rewards", "last_reward", "esteem"]
sarsa = pd.DataFrame(columns=sarsa_col)
sarsa.to_sql('sarsa', conn, if_exists=db_method)


def make_env(stack=True, scale_rew=True, local=False, level_choice=None):
    """
    Create an environment with some standard wrappers.
    """
    print(stack,scale_rew,local)
    if local:  # Select Random Level if local
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
        if not level_choice:
            level_choice = levels[random.randrange(0, 13, 1)]
        else:
            level_choice = levels[level_choice]
        env = make(game='SonicTheHedgehog-Genesis', state=level_choice)
    else:
        print('connecting to remote environment')
        env = grc.RemoteEnv('tmp/sock')
        print('starting episode')
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 100
        self.height = 100
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        print(pytesseract.image_to_string(frame))
        v = np.median(frame)
        sigma=0.33
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # gray = cv2.Canny(gray, threshold1=lower, threshold2=upper)
        frame = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01


class Player(ABC):
    """
    An object that runs a model on an environment and
    produces transitions that can be stored in a replay
    buffer.

    A Player serves a similar role in DQN as a Roller plays in
    policy-gradient algorithms.
    """
    @abstractmethod
    def play(self):
        """
        Gather a sequence of transition dicts.

        Each transition dict should have these keys:
          'obs': the starting observation.
          'model_outs': the batch-size 1 model output
            after seeing the observation.
          'rewards': the rewards after taking the action.
            For n-step Q-learning, there are n rewards.
          'new_obs': the new observation, or None if the
            transition is terminal.
          'info': the info dictionary from the step when
            the action was taken.
          'start_state': the model's state when it chose
            to take the given action. This may be None.
            This is represented as a batch-size 1 value.
            For example, it might be [1 x n].
          'episode_id': a comparable value that identifies
            the particular episode for this timestep.
          'episode_step': the timestep of the initial
            observation, starting from 0.
          'end_time': the UNIX timestamp when the action
            was finished being applied taken.
          'is_last': a boolean indicating if this is the
            last transition for the episode. This may not
            be True even if new_obs is None, since n-step
            Q-learning can result in multiple terminal
            transitions.
          'total_reward': the total reward for the episode
            up to and including this transition.
            If 'new_obs' is None, then this is the total
            reward for the episode.

        All transitions must be ordered such that, if you
        lay out the transitions first according to play()
        call index and then according to the order within
        a batch, episode_step should always be increasing
        for the corresponding episode_id.
        """
        pass


class BasicPlayer(Player):
    """
    A Player that uses a single Gym environment to gather
    sequential batches of 1-step transitions.
    """

    def __init__(self, env, model, batch_size=1):
        self.env = env
        self.model = model
        self.batch_size = batch_size
        self._needs_reset = True
        self._cur_state = None
        self._last_obs = None
        self._episode_id = -1
        self._episode_step = 0
        self._total_reward = 0.0

    def play(self):
        return [self._gather_transition() for _ in range(self.batch_size)]

    def _gather_transition(self):
        if self._needs_reset:
            self._needs_reset = False
            self._cur_state = self.model.start_state(1)
            self._last_obs = self.env.reset()
            self._episode_id += 1
            self._episode_step = 0
            self._total_reward = 0.0
        output = self.model.step([self._last_obs], self._cur_state)
        new_obs, rew, self._needs_reset, info = self.env.step(output['actions'][0])
        self._total_reward += rew
        res = {
            'obs': self._last_obs,
            'model_outs': output,
            'rewards': [rew],
            'new_obs': (new_obs if not self._needs_reset else None),
            'info': info,
            'start_state': self._cur_state,
            'episode_id': self._episode_id,
            'episode_step': self._episode_step,
            'end_time': time.time(),
            'is_last': self._needs_reset,
            'total_reward': self._total_reward
        }
        self._cur_state = output['states']
        self._last_obs = new_obs
        self._episode_step += 1
        return res


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

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)

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

        def action(self, a):  # pylint: disable=W0221
            return self.s_actions[a].copy()

        self.level_choice = ''
        # Toggles
        self.agent = ''
        self.assist = False  # Allow human trainer
        self.trainer = False  # Enables and tracks training
        self.resume = False  # Resumes RL exploration
        # Env Counters
        self.episode = 0
        self.total_steps_ever = 0
        self.steps = 0
        self.total_reward = 0
        # Position Counters
        self.curr_loc = 0
        self._max_x = 0
        self.enemy_cnt = 0
        self.stuck_cnt = 0
        self.unknown_cnt = 0
        self.done = False
        # Scene Trackers
        self.global_rewards = 0
        self.last_obs = []
        self.step_history = []
        self.action_history = []
        self.reward_history = []
        self.solutions = []
        self.step_rew_history = []
        self.step_penalty_history = []
        self.start_time = time.time()
        # Maslow Trackers
        self.survival = False  # Tracks done, seeks to prevent done unless potential reached
        self.safety = False  # Tracks rings, seeks to prevent ring lose
        self.belonging = False  # Tracks collision, seeks to avoid bad friends
        self.esteem = False  # Tracks reward, seeks
        self.potential = False  # Tracks completion || reward > 9000
        # Queues
        self.is_stuck_pos = deque(maxlen=2000)
        self.is_stuck_pos.append(0)
        # Storage
        self.graph = gb
        self.moves = pd.DataFrame(self.graph('agent').take_action(list))

    def best_sequence(self):
        """
        Get the prefix of the trajectory with the best
        cumulative reward.
        """
        max_cumulative = max(self.reward_history)
        for i, rew in enumerate(self.reward_history):
            if rew == max_cumulative:
                return self.action_history[:i + 1]
        raise RuntimeError('unreachable')

    # pylint: disable=E0202
    def reset(self, spawn=True, **kwargs):
        print('Episode', self.episode, self.agent, self.steps, self.total_reward)
        self.action_history = []
        self.reward_history = []
        self._max_x = 0
        self.curr_loc = 0
        self.episode += 1
        self.steps = 0
        self.total_reward = 0
        self.enemy_cnt = 0
        self.stuck_cnt =0
        self.unknown_cnt = 0
        _, self.global_rewards = self.calc_rewards()
        self.start_time = time.time()
        if (spawn  and self.episode > 1 and ((random.random() < .1 + self.total_steps_ever / 1000000) or self.global_rewards/10000 >= .45)):
            self.env.reset(**kwargs)
            new_state, rew, done = self.spawn()
            return new_state
        return self.env.reset(**kwargs)

    def calc_rewards(self):
        try:
            rewards = gb('rewards').game_rewards(list)
            min_spawn = float(np.min(rewards))
            max_spawn = float(np.max(rewards))
        except:
            min_spawn = 0
            max_spawn=0
        return  min_spawn, max_spawn

    def spawn(self):
        _, max_spawn = self.calc_rewards()
        #get_start_time,update_table to confirm action, location, reward
        play_seq = gb(max_spawn).game_sequences(list)
        play_df = pd.DataFrame(play_seq).T
        idx = 0
        idx_end = (len(play_df)-7)
        x_loc = 0
        while idx < idx_end:
            new_state, rew, done, _ = self.step(play_df.iloc[idx][0])
            if done:
                self.reset()
            x_loc += rew
            idx += 1
        return new_state, rew, done

    def stuck(self):
       if np.median(self.reward_history[-60:]) == self.reward_history[-1]:
           return True
       else:
           return False

    def maslow(self, done, info):

        self.potential = False
        if not done:
            self.survival = True
        if (int(self.reward_history[-2]) <= int(self.reward_history[-1])):
            self.esteem = True
        # if info['rings'] > 0:
        #     self.safety = True
        # else:
        #     self.safety = False

    # PROCESSING
    def get_rew_hist(self, a):
        acts = np.array(self.step_rew_history)
        acts = np.sum(acts[-a:])
        acts = np.round(acts, 4)
        return acts


    def get_stats(self):
        table = pd.read_sql_query("SELECT * from game_stats", conn)
        return table

    def penalty(self):
        if self.steps > 1:
            rew_pct = pd.Series(self.step_rew_history)
            rew_pct = rew_pct.pct_change()
            rew_pct = rew_pct[-1:]
        else:
            rew_pct = 0
        return  rew_pct


    def insert_stats(self, action, obs, rew, done, info, start_time, stop_time):
        # tm = trained_model.predict(self.table[-1])
        self.action_history.append(action) #Last Step
        self.step_rew_history.append(rew)# Last Reward
        self.step_penalty_history.append(float(self.penalty())) #Last Penalty
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        # patches = np.array(image.extract_patches_2d(obs, (42,42),max_patches=4)) #NORMAL,STUCK,DISRUPTED,JUMP
        # for patch in patches:
        #     io.imshow(patch)
        #     plt.show()
        # Setup
        acts1 = self.get_rew_hist(1)
        acts3 = self.get_rew_hist(3)
        acts3 = self.get_rew_hist(3)
        acts5 = self.get_rew_hist(5)
        acts7 = self.get_rew_hist(7)
        acts9 = self.get_rew_hist(9)
        acts11 = self.get_rew_hist(11)
        penalty = float(self.penalty()) if (not math.isinf(self.penalty()) and math.isnan(self.penalty())) else None
        ac = 5 if len(self.action_history) > 5 else len(self.action_history)
        if self.steps > 1:
            self.maslow(done, info)
            prev_loc = round(self.reward_history[-2],1)
            prev_action = str(self.action_history[-2])
            curr_action = str(self.action_history[-1])
            action_cluster = str(self.action_history[-5:])
            gb.store_relation(self.total_reward, 'reached_from',
                              {'curr_action': curr_action, "prev_loc": prev_loc})
            db.execute("INSERT INTO sarsa VALUES (NULL,?,?, ?,?,?)",
                       (self.level_choice, action_cluster, self.get_rew_hist(5), self.get_rew_hist(1), self.esteem))
            conn.commit()
            db.execute("INSERT INTO game_stats VALUES (NULL,?,?, ?,?, ?,?,?, ?,?,?,?,?,?,?,?,?,?,?,?)",
                       (self.level_choice, self.start_time, self.steps, prev_loc, curr_action, prev_action
                        , acts1, acts3, acts5, acts7, acts9, acts11, penalty
                        , int(self.safety), int(self.esteem), int(self.belonging), int(self.potential)
                        , int(self.trainer), int(self.total_reward)))
            conn.commit()

            if not done:
                gb.store_relation(prev_loc, 'has_action',
                                  {'curr_action': curr_action, 'curr_reward': acts1, 'penalty': penalty})
                gb.store_relation(prev_loc, 'is_before_chron',
                                  {'curr_loc': self.curr_loc, 'curr_action': curr_action, 'curr_reward': acts1,
                                   'start_time': start_time
                                      , 'stop_time': stop_time})
                if prev_loc == self.curr_loc:  # net_rew
                    gb.store_relation('stuck', 'at_place_spatial',
                                      {'prev_loc': prev_loc, 'curr_action': curr_action, 'curr_reward': acts1,
                                       'curr_loc': self.curr_loc, 'penalty': penalty
                                          , 'start_time': start_time, 'stop_time': stop_time})
                if prev_loc <= 0 and self.curr_loc > 0: #update to be prev reward is zero or negative and curr_loc > global
                    gb.store_relation('unstuck', 'at_place_spatial',
                                      {'prev_loc': prev_loc, 'curr_action': curr_action, 'curr_reward': acts1,
                                       'curr_loc': self.curr_loc, 'penalty': penalty
                                          , 'start_time': start_time, 'stop_time': stop_time})
            else:
                gb.store_relation('act_of_god', 'at_place_spatial',
                                  {'prev_loc': prev_loc, 'curr_action': curr_action,'curr_loc': self.curr_loc,
                                   'curr_reward': acts1,'penalty': penalty,'start_time': start_time, 'stop_time': stop_time})
            gb.store_relation('rewards', 'game_rewards', max(self.reward_history))
            gb.store_relation(max(self.reward_history), 'game_sequences', self.best_sequence())
        self.steps += 1

    def predict_reward(self):
        prediction = trained_step.predict(self.moves[-1:])
        return prediction

    def predict_action(self,lookup=False):
        try:
            x = self.graph(round(self.curr_loc,1)).has_action(list)
            full = pd.DataFrame(x).dropna().drop_duplicates()
            topx = full.nlargest(5, 'curr_reward')
            botx = full.nsmallest(5, 'curr_reward')
        except:
            print("You can't predict the future.")
            topx = None
            botx = None
            full = None
        return topx, full

    def cluster_action(self,step_num=-1):
        moves = self.moves
        moves['steps_sum'] = moves['prev_reward_2'] + moves['prev_reward_1'] + moves['reward']
        moves_kmean = KMeans(n_clusters=11, random_state=0).fit(moves)
        #Split Prep and Run????
        return moves_kmean.predict(moves[step_num:])

    def control(self, a=None):  # Enable Disrete Actions pylint: disable=W0221
        if not a:
            a = self.action_space.sample()
        obs, rew, done, info = self.step(a)
        # print(a,rew)
        return obs, rew, done, info

    def move(self, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=4):
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
                action = self.control(2)
            else:
                action = self.control(3)
            if jumping_steps_left > 0:
                action = self.control(6)
                jumping_steps_left -= 1
            else:
                if random.random() < jump_prob:
                    jumping_steps_left = jump_repeat - 1
                    action = self.control(6)
            _, rew, done, _ = self.step(action)
            total_rew += rew
            steps_taken += 1
        return total_rew, done

    def step(self, action, *args):
        action_num = action
        test = np.array(action).ndim
        if test < 1:
            action = self._actions[action].copy()
        self.total_steps_ever += 1
        obs, rew, done, info = self.env.step(action) # Took the step
        stop_time = time.time()
        if self.steps >= 5: #Warm up before game starts
            #Q-Value
            gb.store_relation('agent', 'take_action',
                              {'prev_action_2': self.step_history[-3],
                               'prev_reward_2': self.step_rew_history[-3],
                               'prev_action_1': self.step_history[-2],
                               'prev_reward_1': self.step_rew_history[-2]
                                  , 'action': action_num, 'reward': rew})
        self.done = done
        self.step_history.append(action_num)
        self.last_obs = obs
        self.curr_loc += rew
        #Acquire-Bond-Comprehend-Defend
        self.insert_stats(action_num, obs, rew, done, info, self.start_time, stop_time)
        rew = max(0, self.curr_loc - self._max_x)
        # rew = rew + float(self.penalty()*.1)
        # if rew == 0 and self.steps > 3:
        #     if self.step_rew_history[-2] > 2 and self.step_rew_history[-1] < 0 and float(self.penalty()) < -1:
        #         rew = min(-10,float(self.penalty()))
        #         self.enemy_cnt += 1
        #     elif (math.isnan(self.penalty())): #or math.isinf(self.penalty()
        #         rew = min(-5, float(self.penalty()))
        #         self.stuck_cnt += 1
        #     elif float(self.penalty() < 0):
        #         rew = min(-1.5, float(self.penalty()))
        #     else:
        #         self.unknown_cnt += 1
        self._max_x = max(self._max_x, self.curr_loc)
        # self.render()
        return obs, rew, done, info


