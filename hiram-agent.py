#!/usr/bin/env python
#D. Johnson - Playable Agent for Training Purposes
# Human Preferences for
# DQN Lib: https://github.com/keon/deep-q-learning/blob/master/dqn.py
# Play Lib: https://raw.githubusercontent.com/openai/gym/master/gym/utils/play.py
# DL from Human Pref: https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/
# DL from Human Pref: https://arxiv.org/abs/1706.03741
# Implicit Imitation: https://www.aaai.org/Papers/JAIR/Vol19/JAIR-1916.pdf
#Note try Prioritized Replay
import random
# ENV_REMOTE
import gym_remote.client as grc
import gym_remote.exceptions as gre
import numpy as np
# ML/RL
import tensorflow as tf
from anyrl.algos import DQN
from anyrl.models import MLPQNetwork, EpsGreedyQNetwork
from anyrl.rollouts import BasicPlayer, PrioritizedReplayBuffer
from anyrl.spaces import gym_space_vectorizer
from sklearn.decomposition import PCA

from sonic_mod import AllowBacktracking, make_env

# ENV_LOCAL
local_env = False
render = False
train = False

# Load ML
# trained_model = load_ml_model('reward.ml')

# GLOBAL
seed = 33
pca = PCA(.85)
done_penalty = -10
np.random.seed(seed)
EXPLOIT_BIAS = 0.001
RL_PLAY_PCT = 1 #2
TOTAL_TIMESTEPS = int(1e6)
COMPLETION = 9000  # Estimated End
batches = 4
TRAINING_STEPS = 2000000 #20000
BUFFER_SIZE = 1024
MIN_BUFFER_SIZE = 256
STEPS_PER_UPDATE = 3
ITERS_PER_LOG = 200
BATCH_SIZE = 64
EPSILON = 0.1
LEARNING_RATE = 0.0001 #0000625
# References
# https://github.com/keon/deep-q-learning/blob/master/dqn.py

print(seed,RL_PLAY_PCT,TRAINING_STEPS,LEARNING_RATE)

def main():
    if local_env:  # Select Random Level if local
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
        env = make_env(stack=False, scale_rew=False, local=local_env)
    else:
        print('connecting to remote environment')
        env = grc.RemoteEnv('tmp/sock')
        print('starting episode')

    env = AllowBacktracking(env)

    solutions = env.solutions  # Track Solutions
    state_size = env.observation_space
    action_size = env.action_space.n
    print(state_size, action_size)
    env.assist = False
    env.trainer = train  # Begin with mentor led exploration
    env.reset()

    while env.total_steps_ever <= TOTAL_TIMESTEPS:  # Interact with Retro environment until Total TimeSteps expire.
        while env.trainer:
            print('Entering Self Play')
            keys = getch()
            if keys == 'A':
                env.control(-1)
            if keys == 'B':
                env.control(4)
            if keys == 'C':
                env.control(3)
            if keys == 'D':
                env.control(2)
                buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
                actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                           ['DOWN', 'B'], ['B']]
            if keys == 'rr':
                env.trainer = False
                continue
            if keys == ' ':
                env.close()
                env = make_env(stack=False, scale_rew=False, local=local_env)
                env = AllowBacktracking(env)
                env.reset()  # Initialize Gaming Environment
                env.trainer = True

        if env.episode % RL_PLAY_PCT == 0:

            tf.reset_default_graph()
            with tf.Session() as sess:
                def make_net(name):
                    return MLPQNetwork(sess,
                                       env.action_space.n,
                                       gym_space_vectorizer(env.observation_space),
                                       name,
                                       layer_sizes=[32])

                dqn = DQN(make_net('online'), make_net('target'))
                bplayer = BasicPlayer(env, EpsGreedyQNetwork(dqn.online_net, EPSILON),
                                     batch_size=STEPS_PER_UPDATE)
                optimize = dqn.optimize(learning_rate=LEARNING_RATE)

                sess.run(tf.global_variables_initializer())

                env.agent = 'DQN'
                dqn.train(num_steps=TRAINING_STEPS,
                          player=bplayer,
                          replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                          optimize_op=optimize,
                          target_interval=200,
                          batch_size=64,
                          min_buffer_size=200,
                          handle_ep=lambda _, rew: print('Exited DQN with : ' + str(rew) + str(env.steps)))

        new_ep = True  # New Episode Flag
        while new_ep:
            if new_ep:
                if (solutions and
                        random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                    new_state, new_rew, done = env.spawn()
                    continue
                else:
                    env.reset()
                    new_ep = False
            env.agent = 'JERK'
            rew, new_ep = move(env, 100)
            if not new_ep and rew <= 0:
                #print('backtracking due to negative reward: %f' % rew)
                _, new_ep = move(env, 70, left=True)
            if new_ep:
                solutions.append(([max(env.reward_history)], env.best_sequence()))


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
            action = env.control(2)
        else:
            action = env.control(3)
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
    return total_rew, done

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)