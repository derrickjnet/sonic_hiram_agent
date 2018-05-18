"""
Environments and wrappers for Sonic training.
"""

import random
import time
import gym
import numpy as np
import pandas as pd
from graphdb import GraphDB

from baselines.common.atari_wrappers import WarpFrame, FrameStack
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
        self.agent ='Rainbow'
        self._cur_x = 0
        self._max_x = 0
        self.curr_loc = 0
        self.episode = 0
        self.steps = 0
        self.total_reward = 0
        self.action_history = []
        self.reward_history = []
        self.step_rew_history = []

    def reset(self, spawn=False, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        self.curr_loc = 0
        self.steps = 0
        self.total_reward = 0
        self.action_history = []
        self.reward_history = []
        self.step_rew_history = []
        self.episode += 1
        if spawn and self.episode % 4 == 0:
            self.env.reset(**kwargs)
            new_state, rew, done = self.spawn()
            return new_state
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        start_time = time.time()
        obs, rew, done, info = self.env.step(action)
        stop_time = time.time()
        self.insert_stats(action,obs,rew,done,info,start_time,stop_time)
        self._cur_x += rew
        #Acquire-Bond-Comprehend-Defend
        rew = max(0, self._cur_x - self._max_x) if np.median(self.step_rew_history[-3:]) != rew else -5 if not done else -10
        self._max_x = max(self._max_x, self._cur_x)
        self.steps += 1
        return obs, rew, done, info

    def spawn(self):
        rewards = gb('rewards').game_rewards(list)
        min_spawn = float(np.min(rewards))
        mode_spawn = float(np.median(rewards))
        max_spawn = float(np.max(rewards))

        play_seq = gb(max_spawn).game_sequences(list)
        play_df = pd.DataFrame(play_seq).T
        idx = 0
        idx_end = (len(play_df)-5)
        x_loc = 0
        while idx < idx_end:
            new_state, rew, done, _ = self.step(play_df.iloc[idx][0])
            if done:
                self.reset()
            x_loc += rew
            idx += 1
        return new_state, rew, done

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

    def insert_stats(self, action, obs, rew, done, info, start_time, stop_time):
        self.action_history.append(action.copy())
        self.step_rew_history.append(rew)  # Step Reward
        self.reward_history.append(self.total_reward)
        self.total_reward += rew
        self.curr_loc += rew
        if self.steps > 1:
            prev_loc = self.reward_history[-2]
            prev_action = str(self.action_history[-2])
            curr_action = str(self.action_history[-1])
            gb.store_relation(self.total_reward, 'reached_from',
                                  {'curr_action': curr_action, "prev_loc": prev_loc})
            if prev_loc < self.curr_loc:  # pos_rew
                gb.store_relation(int(prev_loc), 'has_action',
                                  {'curr_action': curr_action, 'curr_reward': rew, 'start_time': start_time
                                      , 'stop_time': stop_time, 'agent': self.agent})
                gb.store_relation(int(prev_loc), 'is_before_chron',
                                  {'curr_loc': self.curr_loc, 'curr_action': curr_action, 'curr_reward': rew,
                                   'start_time': start_time
                                      , 'stop_time': stop_time, 'agent': self.agent})
            if prev_loc == self.curr_loc:  # net_rew
                gb.store_relation('stuck', 'at_place_spatial',
                                  {'prev_loc': prev_loc, 'curr_action': curr_action, 'curr_reward': rew,
                                   'curr_loc': self.curr_loc
                                      , 'start_time': start_time, 'stop_time': stop_time, 'agent': self.agent})
            if prev_loc <= 0 and self.curr_loc > 0:
                gb.store_relation('unstuck', 'at_place_spatial',
                                  {'prev_loc': prev_loc, 'curr_action': curr_action, 'curr_reward': rew,
                                   'curr_loc': self.curr_loc
                                      , 'start_time': start_time, 'stop_time': stop_time, 'agent': self.agent})
        if done:
            gb.store_relation('act_of_god', 'at_place_spatial',
                              {'prev_loc': prev_loc, 'curr_action': curr_action, 'curr_reward': rew,
                               'start_time': start_time
                                  , 'stop_time': stop_time, 'agent': self.agent})
            gb.store_relation(self.agent, 'is_done_at',
                              {'prev_loc': prev_loc, 'total_reward': self.total_reward})
            gb.store_relation('rewards', 'game_rewards', max(self.reward_history))
            gb.store_relation(max(self.reward_history), 'game_sequences', self.best_sequence())
