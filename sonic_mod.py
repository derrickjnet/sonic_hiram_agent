"""
Environments and wrappers for Sonic training.
#D. Johnson - Mod to Sonic_Utils
"""

import gym
import random
import time
import numpy as np
import pandas as pd
import sqlite3
from graphdb import GraphDB
from collections import deque
from baselines.common.atari_wrappers import WarpFrame, FrameStack
import gym_remote.client as grc

seed = 33
np.random.seed(seed)

# Create Storage
conn = sqlite3.connect('retro.db')
db_method = 'replace'
db = conn.cursor()
gb = GraphDB('graph.db')

#Load Storage
from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model
# trained_step = load_ml_model("next_step.dill")


# Setup Storage
stats_col = ["level", "episode", "steps",'curr_pos', "curr_action", "prev_action", "acts1", "acts3", "acts5", "acts7", "acts9",
             "acts11"
    , "acts33", "safety", "esteem", "belonging", "potential", "human", "total_reward"]
df = pd.DataFrame(columns=stats_col)
df.to_sql('game_stats', conn, if_exists=db_method)

sarsa_col = ["level", "action_cluster", "rewards", "last_reward", "esteem"]
sarsa = pd.DataFrame(columns=sarsa_col)
sarsa.to_sql('sarsa', conn, if_exists=db_method)

def make_env(stack=True, scale_rew=True, local=False):
    """
    Create an environment with some standard wrappers.
    """
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
        level_choice = levels[random.randrange(0, 13, 1)]
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
        self.step_history = []
        self.action_history = []
        self.reward_history = []
        self.seq_replay = []
        self.step_rew_history = []
        self.replay_reward_history = []
        self.jerk = 0
        self.dqn = 0
        self.rainbow = 0
        # Maslow Trackers
        self.survival = False  # Tracks done, seeks to prevent done unless potential reached
        self.safety = False  # Tracks rings, seeks to prevent ring lose
        self.belonging = False  # Tracks collision, seeks to avoid bad friends
        self.esteem = False  # Tracks reward, seeks
        self.potential = False  # Tracks completion || reward > 9000
        # Queues
        self.memory = deque(maxlen=2000)
        self.is_stuck_pos = deque(maxlen=2000)
        self.is_stuck_pos.append(0)
        self.in_danger_pos = deque(maxlen=2000)
        self.in_danger_pos.append(0)
        self.track_evolution = deque(maxlen=2000)
        # Storage
        self.table = pd.read_sql_query("SELECT * from game_stats", conn)
        self.graph = gb

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
        self.curr_loc = 0
        self._max_x = 0
        self.total_reward = 0
        self.steps = 0
        self.episode += 1
        if spawn and self.episode > 50 and self.episode % 5 == 0:
            self.env.reset(**kwargs)
            new_state, rew, done = self.spawn()
            return new_state
        return self.env.reset(**kwargs)

    def spawn(self):
        rewards = gb('rewards').game_rewards(list)
        min_spawn = float(np.min(rewards))
        mode_spawn = float(np.median(rewards))
        max_spawn = float(np.max(rewards))

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

    def maslow(self, done, info):

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
                # self.resume_rl()
                self.is_stuck_pos.append(self.total_reward)
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

    def insert_stats(self, action, obs, rew, done, info, start_time, stop_time):
        # tm = trained_model.predict(self.table[-1])
        self.action_history.append(action.copy())
        self.step_rew_history.append(rew)  # Step Reward
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
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
            self.maslow(done, info)
            prev_loc = self.reward_history[-2]
            prev_action = str(self.action_history[-2])
            curr_action = str(self.action_history[-1])
            action_cluster = str(self.action_history[-5:])
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
        self.steps += 1



    def control(self, a=None):  # Enable Disrete Actions pylint: disable=W0221
        if not a:
            a = self.action_space.sample()
        obs, rew, done, info = self.step(self._actions[a].copy())
        self.step_history.append(a)
        if self.steps >= 5:
            gb.store_relation('agent', 'take_action',
                              {'start':time.time(),'prev_action_1':self.step_history[-1],'prev_reward_1':self.step_rew_history[-1]
                                  ,'action':a,'reward':rew})
        # print(a,rew)
        if done:
            self.reset()
        return self._actions[a].copy()

    def predict_reward(self):
        moves = pd.DataFrame(self.graph('agent').take_action(list))
        prediction = trained_step.predict(moves[-1:])
        return prediction

    def predict_action(self,lookup=False):
        x = self.graph(self.curr_loc).has_action(list)
        da = pd.DataFrame(x).drop_duplicates()
        topx = da.nlargest(3, 'curr_reward')
        move = (topx['curr_action'][0])
        return move

    def step(self, action, *args):
        test = np.array(action).ndim
        if test < 1:
            action = self._actions[action].copy()
        self.total_steps_ever += 1
        start_time = time.time()
        obs, rew, done, info = self.env.step(action)
        stop_time = time.time()
        self.last_obs = obs
        self.curr_loc += rew
        #Acquire-Bond-Comprehend-Defend
        self.insert_stats(action, obs, rew, done, info, start_time, stop_time)
        if self.steps > 20:
            1==1
            # print('reward:',rew,'pred_reward:',self.predict_reward())
        rew = max(0, self.curr_loc - self._max_x) #AllowBacktracking
        self._max_x = max(self._max_x, self.curr_loc)
        # self.render()
        return obs, rew, done, info


