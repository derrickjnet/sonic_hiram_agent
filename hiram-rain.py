#!/usr/bin/env python
#D. Johnson - Rainbow Agent for Training Purposes

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""
import random
import tensorflow as tf
from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models, MLPQNetwork, EpsGreedyQNetwork
from anyrl.rollouts import BasicPlayer, BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from sonic_mod import AllowBacktracking, make_env

local_env = True
BATCH_SIZE = 64
STEPS_PER_UPDATE = 3
EPSILON = 0.1
LEARNING_RATE = 1e-4 #0000625
TRAINING_STEPS = 2000000 #20000


def main():
    """Run DQN until the environment throws an exception."""
    env = AllowBacktracking(make_env(stack=False, scale_rew=False,local=local_env))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:

        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4) #1e-4
        sess.run(tf.global_variables_initializer())
        env.agent = 'Rainbow'
        dqn.train(num_steps=TRAINING_STEPS, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
