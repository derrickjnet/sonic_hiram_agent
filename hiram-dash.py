#env
from retro_contest.local import make
import gym
import gym.spaces as spaces
import gym_remote.client as grc
import gym_remote.exceptions as gre
import time
from collections import deque
#db
#visual
import cv2
import matplotlib.pyplot as plt
#ml
import numpy as np
import random
from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer


# Author: Derrick L. Johnson (derrickj@derrickj.net)
#Hiram's Dash - Blends JERK, Replay Experience,  with DQN



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

#global

EMA_RATE = 0.20
EXPLOIT_BIAS = 0.20
TOTAL_TIMESTEPS = int(1e6)
frames = -10
def main():
    """Run JERK on the attached environment."""
    LEVEL = levels[random.randrange(0, 13, 1)]
    print(LEVEL)
    env = make(game='SonicTheHedgehog-Genesis', state=LEVEL)
    env = TrackedEnv(env)
    new_ep = True
    solutions = []
    while True:
        if new_ep:
            if (solutions and
                    random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                best_pair = solutions[-1]
                new_rew = exploit(env, best_pair[1])
                best_pair[0].append(new_rew)
                print('replayed best with reward %f' % new_rew)
                continue
            else:
                env.reset()
                new_ep = False
        rew, new_ep = move(env, 100)
        if not new_ep and rew <= 0:
            _, new_ep = move(env, 70, left=True)
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))

def move(env, num_steps, left=False, jump_prob= 1.0 / 10.0, jump_repeat=4):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    done = False
    total_rew = 0.0
    steps_taken = 0
    jumping_steps_left = 0
    prev_state = []
    if left:
        capture_moment(env, frames,done)
        # env.maslow()
    while not done and steps_taken < num_steps:
        action = np.zeros((12,), dtype=np.bool)
        action[6] = left
        action[7] = not left
        if jumping_steps_left > 0:
            action[0] = True
            jumping_steps_left -= 1
        else:
            if random.random() < jump_prob:
                jumping_steps_left = jump_repeat - 1
                action[0] = True
        new_state, rew, done, _ = env.step(action)
        env.render()
        env.remember(prev_state, action, rew, new_state, done, left)
        prev_state = new_state
        total_rew += rew
        steps_taken += 1

        if done:
            capture_moment(env,frames,done)
            #env.maslow()
            break
    return total_rew, done


def capture_moment(env,frames,done):
    if done:
        env.in_danger_pos.append(env.total_reward)
        img = env.observation_history[frames]
        show_moment(img)
    elif env.total_reward > max(env.is_stuck_pos):
        env.is_stuck_pos.append(env.total_reward)
        img = env.observation_history[frames]
        show_moment(img)
        img = env.observation_history[frames+5]
        show_moment(img)

def show_moment(img):
    plt.imshow(img)
    plt.show()


def exploit(env, sequence):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    env.reset()
    done = False
    idx = 0
    while not done:
        if idx >= len(sequence):
            env.action_space.sample()
        else:
            _, _, done, _ = env.step(sequence[idx])
        idx += 1
    return env.total_reward


class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        self.total_steps_ever = 0
        self.ep_max_reward = 0
        self.seq_replay = []
        self.memory = deque(maxlen=2000)
        self.is_stuck_pos = deque(maxlen=2000)
        self.is_stuck_pos.append(0)
        self.in_danger_pos = deque(maxlen=2000)
        self.in_danger_pos.append(0)

    def remember(self, state, action, reward, new_state, done, left):
        self.memory.append([state, action, reward, new_state, done, left, time.time()])

    def replay(self):
        batch_size = 8
        if len(self.memory) < batch_size:
            return
        print('replaying')
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state[0])
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)


    def maslow(self, done,info):
        self.survival = True #Tracks done, seeks to prevent done unless potential reached
        self.safety = True #Tracks rings, seeks to prevent ring lose
        self.belonging = False #Tracks collision, seeks to avoid bad friends
        self.esteem = False #Tracks reward, seeks
        self.potential = False #Tracks completion || reward > 9000
        if self.total_reward > 9000 & done:
            self.potential = True
        if done:
            self.survival = False
        if self.reward_history[-2] >= self.reward_history[-1]:
            self.esteem = False
            print(self.reward_history[-2], self.reward_history[-1])
        if info['rings'] > 0:
            self.safety = True
        print(self.survival,self.safety,self.belonging,self.esteem,self.potential)



        intel = self.is_stuck_pos + self.in_danger_pos


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
        self.observation_history = []
        self.max_reward = 0
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action) #Takes Action
        self.total_steps_ever += 1
        self.action_history.append(action.copy()) #Tracks Actions for
        self.observation_history.append(obs)  # Append state to oberservation history
        self.total_reward += rew # Tracks total reward / current position
        self.reward_history.append(self.total_reward)
        if self.total_steps_ever > 1:
            self.maslow(done,info)
        return obs, rew, done, info

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
