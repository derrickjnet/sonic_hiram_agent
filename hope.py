#!/usr/bin/env python
#D. Johnson - Rainbow Agent for Training Purposes

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""
import os
# os.remove('graph.db') #Scrub Env
# os.remove('retro.db') #Scrub Env

import random
import tensorflow as tf
import numpy as np
import pandas as pd
from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from sonic_mod import AllowBacktracking, make_env

local_env = True
scrub_env = False
trainer = True



def main():
    env = make_env(local=local_env, level_choice=-3)
    env = AllowBacktracking(env)

    env.reset()
    done = False
    while True:
        qtable_top,full = env.predict_action()
        while env.episode == 1:
            fwd_moves = [3,-1]
            random.choice(fwd_moves)
            obs, rew, done, _ = env.control(random.choice(fwd_moves))
            if done:
                env.episode += 1
                env.reset()
        if qtable_top is None:
            obs, rew, done, _ = env.control()
        else:

            qtable_top.round(1)
            qtable = int(qtable_top.iloc[0]['curr_action'])
            obs, rew, done, _ = env.control(qtable)
        if done:
            env.reset()
        if rew <= 0:
            obs, rew, done, _ = env.control()
            if done:
                env.reset()




if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
