"""
Environments and wrappers for Sonic training.
"""

import gym
import math
import statistics
import numpy as np
import cv2
import pandas as pd
import time
import random
# import the necessary packages
import argparse
import imutils
import cv2

from baselines.common.atari_wrappers import FrameStack
from gym import spaces
from graphdb import GraphDB
from skimage.measure import compare_ssim

import gym_remote.client as grc

gb = GraphDB('graph.db')

def make_env(stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = grc.RemoteEnv('tmp/sock')
    env = SonicDiscretizer(env)
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
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
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

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0
        self.global_rewards = 0
        self.replay_retry = .1
        self.max_score = 10000
        self.start_time = time.time()
        self.action_history = []
        self.reward_history = []
        self.obs_history = []
        self.mod_reward_history = []

    def reset(self, spawn=bool(random.getrandbits(1)),**kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        self.start_time = time.time()
        self.action_history = []
        self.reward_history = []
        self.step_rew_history = []
        if (spawn and self.global_rewards/self.max_score > self.replay_retry):
            reset_obs = self.env.reset(**kwargs)
            new_state, rew, done = self.spawn(reset_obs)
            self.replay_retry = self.global_rewards/self.max_score
            return new_state
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        stop_time = time.time()
        self._cur_x += rew
        self.insert_stats(action, obs, rew, done, info, self.start_time, stop_time)
        rew = self.calc_experience(obs,rew)
        self.mod_reward_history.append(rew)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

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

    def calc_rewards(self):
        try:
            rewards = gb('rewards').game_rewards(list)
            min_spawn = float(np.min(rewards))
            max_spawn = float(np.max(rewards))
        except:
            min_spawn = 0
            max_spawn=0
        return  min_spawn, max_spawn

    def calc_experience(self,obs,rew):
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        loc_score = max(0, self._cur_x - self._max_x)
        penalty_score = float(self.penalty()) if not math.isnan(self.penalty()) else 0
        exp_score = statistics.mean([loc_score, penalty_score,rew]) #location progress, step (pct_rew), reward
        print(loc_score,penalty_score,rew, exp_score)
        if exp_score == 0:
            rtn_score = -10
        else:
            rtn_score = loc_score
        return rtn_score

    def insert_stats(self, action, obs, rew, done, info, start_time, stop_time):
        self.action_history.append(action)  # Last Step
        self.reward_history.append(self._cur_x)
        self.step_rew_history.append(rew)  # Last Reward
        if done:
            gb.store_relation('rewards', 'game_rewards', max(self.reward_history))
            gb.store_relation(max(self.reward_history), 'game_sequences', self.best_sequence())
            self.global_rewards = max(self._cur_x,self.global_rewards)


    def penalty(self):
        rew_pct = pd.Series(self.step_rew_history)
        rew_pct = rew_pct.pct_change()
        rew_pct = rew_pct[-1:]
        return  rew_pct

    def spawn(self,reset_obs):
        _, max_spawn = self.calc_rewards()
        #get_start_time,update_table to confirm action, location, reward
        play_seq = gb(max_spawn).game_sequences(list)
        play_df = pd.DataFrame(play_seq).T
        idx = 0
        idx_end = len(play_df)-50 if len(play_df) <= 3000 else 500
        x_loc = 0
        while idx < idx_end:
            new_state, rew, done, _ = self.step(play_df.iloc[idx][0])
            if done:
                self.reset()
            x_loc += rew
            idx += 1
        return new_state, rew, done