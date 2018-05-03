#!/usr/bin/env python
#ENV_LOCAL
from retro_contest.local import make
#ENV_REMOTE
import gym_remote.client as grc
import gym_remote.exceptions as gre
#ENV_PLUS
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
import skimage
#FIFO
from collections import deque
#ML/RL
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
#GLOBALS
local_env = True
EMA_RATE = 0.2
EXPLOIT_BIAS = 0.15
TOTAL_TIMESTEPS = int(1e6)
completion = 9000 #Estimated End
batches = 4
#References
#https://github.com/keon/deep-q-learning/blob/master/dqn.py


def main():
    """Run JERK on the attached environment."""

    if local_env:
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
        LEVEL =  levels[3]#[random.randrange(0, 13, 1)]
        print(LEVEL)
        env = make(game='SonicTheHedgehog-Genesis', state=LEVEL)
    else:
        env = grc.RemoteEnv('tmp/sock')
    env = TrackedEnv(env)
    new_ep = True
    episode = 1
    solutions = []
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(state_size, action_size)
    agent = DQNAgent(state_size, action_size)
    env.pause_rl()
    while env.total_steps_ever <= TOTAL_TIMESTEPS:
        if new_ep: #If new episode....
            if env.rl: #If RL is predict with RL
                if not env.resume: #Resume RL Decisions
                    state = env.reset()
                else:
                    state = env.step(env.control())
                done = False
                while not done:
                    # env.render()
                    action = agent.act(state,env)
                    next_state, reward, done, _ = env.step(action)
                    reward = reward if not done else -10
                    #next_state = np.reshape(next_state, [1, state_size])
                    agent.remember(state, action, reward, next_state, done,env)
                    state = next_state
                env.pause_rl()
                continue
            #Else JERK replay
            if (solutions and
                    random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                best_pair = solutions[-1]
                new_rew = exploit(env, best_pair[1], agent)
                best_pair[0].append(new_rew)
                print('replayed best with reward %f' % new_rew)
                print(best_pair[0])
                continue
            else:
                env.reset()
                new_ep = False
        #Else JERK play
        rew, new_ep = move(env, 100)
        if not new_ep and rew <= 0:
            env.resume_rl()
            #print('backtracking due to negative reward: %f' % rew)
            # _, new_ep = move(env, 70, left=True)
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))
            episode += 1
    agent.save('hiram')

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
            action = env.control(0)
        else:
            action = env.control(1)
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
        if done:
            break
    return total_rew, done

def capture_moment(env,frames,done):
    if done:
        env.in_danger_pos.append(env.total_reward)
        img = env.observation_history[-1]
        show_moment(img)
    elif env.total_reward > max(env.is_stuck_pos):
        env.is_stuck_pos.append(env.total_reward)
        img = env.observation_history[-1]
        show_moment(img)
        # img = env.observation_history[frames+5]
        # show_moment(img)

def show_moment(img):
    plt.imshow(img)
    plt.show()


def exploit(env, sequence, agent):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    state = env.reset()
    done = False
    idx = 0
    env.rl = False
    while not done:
        if idx >= len(sequence):
            env.resume_rl()
            done = True
        else:
            new_state, rew, done, _ = env.step(sequence[idx])
            agent.remember(state, sequence[idx], rew, new_state, done, env) #Train agent from best sequence
            state = new_state
        idx += 1
    # agent.replay(batches)
    env.total_replays += 1
    env.replay_reward_history.append(env.total_reward)
    return env.total_reward

#DEFINE AGENT
class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_shape=(1,224,320), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, env):
        self.memory.append((state, action, reward, next_state, done,
                            env.safety,env.esteem,env.belonging,env.potential))

    def act(self, state, env):
        if np.random.rand() <= self.epsilon:
            return env.control()  #Fix
        print('RL Move')
        act_values = self.model.predict(state)
        print(act_values)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        print('Replaying')
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done, _, _, _, _ in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            print(target_f)
            target_f[0][action] = target_f
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


#DEFINE GYM
class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        #self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

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
        self.action_history = []
        self.reward_history = []
        self._cur_x = 0
        self._max_x = 0
        self.total_reward = 0
        self.rl_reward_history = []
        self.rl_total_reward = 0
        self.total_steps_ever = 0
        self.replay_reward_history = []
        self.total_replays = 0
        self.seq_replay = []
        self.memory = deque(maxlen=2000)
        self.is_stuck_pos = deque(maxlen=2000)
        self.is_stuck_pos.append(0)
        self.in_danger_pos = deque(maxlen=2000)
        self.in_danger_pos.append(0)
        self.track_evolution = deque(maxlen=2000)
        self.rl = False
        self.resume = False
        self.survival = False  # Tracks done, seeks to prevent done unless potential reached
        self.safety = False  # Tracks rings, seeks to prevent ring lose
        self.belonging = False  # Tracks collision, seeks to avoid bad friends
        self.esteem = False  # Tracks reward, seeks
        self.potential = False  # Tracks completion || reward > 9000

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
        self._cur_x = 0
        self._max_x = 0
        self.total_reward = 0
        self.steps = 0
        return self.env.reset(**kwargs)

    def resume_rl(self):
        self.rl = True
        self.resume = True

    def pause_rl(self):
        self.rl = False
        self.resume = False

    def maslow(self, done, info):
        if (self.total_reward >= completion) and done:
            self.potential = True
        else:
            self.potential = False
        if not done:
            self.survival = True
        if int(self.reward_history[-2]) < int(self.reward_history[-1]):
            self.esteem = True
        else:
            self.esteem = False
            self.is_stuck_pos.append(self.total_reward)
            self.resume_rl()
        if info['rings'] > 0:
            self.safety = True
        else:
            self.safety = False
        # self.track_evolution.append({'reward':self.total_reward,'survival':self.survival,'safety':self.safety,'belonging':self.belonging,
        #                              'esteem':self.esteem,'potential':self.potential})

    def process_frames(self,obs):
        x_t1 = skimage.color.rgb2gray(obs)
        #x_t1 = skimage.transform.resize(x_t1, (84, 84))
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        return x_t1

    def control(self, a = None):  # pylint: disable=W0221
        if not a:
            a = self.action_space.sample()
        return self._actions[a].copy()

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        obs = self.process_frames(obs)
        self.total_reward += rew
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        self.reward_history.append(self.total_reward)
        if self.steps > 1:
            self.maslow(done,info)
        self.steps += 1
        self.render()
        return obs, rew, done, info

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
