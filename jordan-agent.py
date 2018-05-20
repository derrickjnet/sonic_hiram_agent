#!/usr/bin/env python
#D. Johnson - ML Agent for Training Purposes
"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""
import random
import tensorflow as tf
from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre
import pandas as pd
from auto_ml import Predictor

from sonic_mod import AllowBacktracking, make_env

local_env = True

def main():
    env = make_env(stack=True,scale_rew=False,local=local_env)
    env = AllowBacktracking(env)
    state_size = env.observation_space
    action_size = int(env.action_space.n)
    env.reset()
    i = 0
    move_table = 50 #Build move table (Q) by testing each action x amount of times
    moves = pd.DataFrame(env.graph('agent').take_action(list))
    moves['steps_sum'] = moves['prev_reward_1'] + moves['prev_reward_2'] + moves['prev_reward_3']

    # moves.ix[:, 'action']   #select column
    # print(moves)
    #idea 5 columns,

    # Setup Auto_ML
    df_train = moves.sample(frac=.6)
    df_test = moves.sample(frac=.4)

    rew_descriptions = {
        'reward': 'output',
        'start': 'ignore'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=rew_descriptions)
    ml_predictor.train(df_train, ml_for_analytics=True)
    # ml_predictor.score(df_test, df_test.acts1)
    print('predictions',ml_predictor.predict(moves[-1:]))
    ml_predictor.save(file_name='next_step.dill', verbose=True)



if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
