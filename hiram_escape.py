#env
from retro_contest.local import make
import gym
import gym.spaces as spaces
import gym_remote.client as grc
import gym_remote.exceptions as gre
import argparse
#db
#visual
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
import imutils
#ml
import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Dropout
from keras.optimizers import Adam
from collections import deque
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import tensorflow as tf

# Author: Derrick L. Johnson (derrickj@derrickj.net)
#Hiram's Agent - Blends exploration with DQN
#  Initialize Q(s, a) arbitrarily
# Repeat (for each episode):
# 	Initialize s
# 	Choose a from s using policy derived from Q
# 	While (s is not a terminal state):
# 		Take action a, observe r, s'
# 		Choose a' from s' using policy derive from Q
# 		Q(s,a) += alpha * (r + gamma * max,Q(s', a') - Q(s,a))
# 		s = s', a = a'
# Initialize Q(s,a) arbitrarily
# 		Choose a from s using policy derived from Q
# 		Take action a, observe r, s'
# 		Q(s,a) += alpha * (r + gamma * max,Q(s') - Q(s,a))
# 		s = s'


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
is_stuck_img = []
is_stuck_pos = [0]
max_rewards = []
experiences = []

def main():
    LEVEL = levels[random.randrange(0, 13, 1)]
    print('Begining: ',LEVEL)
    #Environment Classes
    stack = True
    scale_rew = True
    warp = True
    env = make(game='SonicTheHedgehog-Genesis', state=LEVEL)
    # env = grc.RemoteEnv('tmp/sock')
    print('initial')
    debug(env)
    if scale_rew:
        print('Scale')
        env = RewardScaler(env)
        debug(env)
    if warp:
        print('Warp')
        env = WarpFrame(env)
        debug(env)
    if stack:
        print('Stack')
        env = FrameStack(env, 4)
        debug(env)
    env = TrackedEnv(env)
    debug(env)
    # ML Varables:
    gamma = 0.9
    epsilon = .95
    jerk_attempts = 15
    trial = 0
    trials = 1000
    trial_len = 500
    updateTargetNetwork = 1000
    steps = []
    new_ep = True
    solutions = []
    total_rew = 0.0
    done = False
    game_reward = True
    steps_taken = 0
    while True and steps_taken < TOTAL_TIMESTEPS:

        if new_ep:
            if (solutions and
                    random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                best_pair = solutions[-1]
                new_rew = exploit(env, best_pair[1])
                best_pair[0].append(new_rew)
                max_rewards.append((new_rew))
                print('replayed best with reward %f and max rewards: %f' % (new_rew, max(max_rewards)))
                continue
            else:
                env.reset()
                new_ep = False
        rew, new_ep = move(env,100)
        if not new_ep and rew <= 0:
            #\print('backtracking due to negative reward: %f' % rew)
            _, new_ep = move(env, 70, left=True)
            continue
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))
        trial += 1
        if steps_taken >= 199:
            print("Failed to complete in trial")
            if steps_taken % 10 == 0:
                env.save_model("trial-{}.model")
            else:
                print("Completed in {} trials".format(trial))
                env.save_model("success.model")
                break

def debug(env):
    print ('Obs Size',env.observation_space)
    print('Action Space',env.action_space)


def move(env,num_steps, left=False, jump_prob=0.6 / 10.0, jump_repeat=4):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    total_rew = 0.0
    done = False
    steps_taken = 0
    jumping_steps_left = 0
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
        obs, rew, done, info = env.step(action)
        env.render()
        if left:
            im_stuck(env, obs, info)
        env.render()  # See the game played
        total_rew += rew
        steps_taken += 1
        if done:
            capture_moment(obs, 'died')
            break

    return total_rew, done



def im_in_control(env,obs):
    action = np.zeros((12,), dtype=np.bool)
    obs2, rew, _, _ = env.step(action)
    img1 = obs[1:224, 100:320, :]
    img2 = obs2[1:224, 100:320, :]
    is_fate = np.array_equal(img1,img2)
    print('lost control:',is_fate)
    if not is_fate:
        capture_moment(obs, '???')
    return is_fate

def im_stuck(env, obs,info):
    obs2, _, _, _ = env.step(np.zeros((12,), dtype=np.bool))
    obs2 = obs2[1:224, 100:320, :]
    obs1 = obs[1:224, 100:320, :]
    is_stuck_img.append(obs1)

    if not np.array_equal(obs1,obs2):
        if max(is_stuck_pos) <  info['screen_x']:
            is_stuck_pos.append(info['screen_x'])
            is_stuck_img.append(obs1)
            capture_moment(obs,'stranded')
            im_in_control(env,obs)
            env.im_stranded = True
            return
        else:
            env.im_stranded = False


def capture_moment(img, moment='unknown'):
    plt.imshow(img)
    plt.show()
    return

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
            _, _, done, _ = env.step(np.zeros((12,), dtype='bool'))
        else:
            obs, _, done, _ = env.step(sequence[idx])
        idx += 1
    return env.total_reward


#SEGA
class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        # self.env = env
     #Game History
        self.action_history = []
        self.reward_history = []
        self.sonic_history = []
        self.attempts = 1
        self.attempts_step = 0
        self.total_reward = 0
        self.total_steps_ever = 0
        self.last_observation = []
        self.im_stranded = False

    #Backtracking:
        self._cur_x = 0
        self._max_x = 0
    #RL variables for agent init
    # (sourced:https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
    #Model creation and target creation
        self.model = self.create_model()
        self.target_model = self.create_model()
    #Sonic stuff
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT",
                   "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'],
                   ['DOWN'], ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Conv2D(32, (3, 3), input_shape=state_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="binary_crossentropy", #"mean_squared_error"
                      optimizer='rmsprop', #Adam(lr=self.learning_rate)
                      metrics=['accuracy'])
        return model
    #A random act
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    #Learning
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        print('attempting replay')
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

    #Training
    def target_train(self):
        print('train')
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    #Saving
    def save_model(self, fn):
        self.model.save(fn)

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
        self.total_reward = 0
        self._cur_x = 0
        self._max_x = 0
        self.attempts_step = 0
        return self.env.reset(**kwargs)

    #Make Sonic Step
    def step(self, action):
        self.total_steps_ever += 1
        #self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.last_observation = obs
        self.attempts_step += 1
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        self._cur_x += rew
        self.rew1 = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        self.sonic_history.append({ 'attempt':self.attempts
        , 'step':self.attempts_step, 'action': [], 'current_x':int(self._cur_x)
        , 'reward':int(rew),'reward_2':int(self.rew1),  'total_reward':int(self.total_reward)
        , 'max_reward':int(self._max_x), 'stuck': self.im_stranded, 'done':done, 'img':'no'})
        if done or self.im_stranded:
            print(self.sonic_history[-3:])
            if done:
                self.attempts += 1
                self.attempts_step = 0

        return obs, rew, done, info



class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are
        only stored once. It exists purely to optimize memory usage which can
        be huge for DQN's 1M frames replay buffers. This object should only be
        converted to numpy array before being passed to the model. You'd not
        believe how complex the previous solution was.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack the k last frames.
        Returns a lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(shp[0], shp[1], shp[2] * k),
                                            dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class WarpFrame(gym.ObservationWrapper):
    #Create secondary observation space
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 320
        self.height = 224
        self.observation_space2 = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1),
                                            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO. This is incredibly important
    and effects performance a lot.
    """
    def reward(self, reward):
        return reward * 0.01

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
