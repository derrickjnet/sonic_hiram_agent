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
#FIFO
from collections import deque
#ML/RL
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
#GLOBALS
local_env = True
EMA_RATE = 0.2
EXPLOIT_BIAS = 0.25
TOTAL_TIMESTEPS = int(1e6)
completion = 9000 #Estimated End
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
        LEVEL = levels[random.randrange(0, 13, 1)]
        print(LEVEL)
        env = make(game='SonicTheHedgehog-Genesis', state=LEVEL)
    else:
        env = grc.RemoteEnv('tmp/sock')
    env = TrackedEnv(env)
    new_ep = True
    solutions = []
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    survival_agent = DQNAgent(state_size, action_size)
    safety_agent = DQNAgent(state_size, action_size)
    esteem_agent = DQNAgent(state_size, action_size)
    belonging_agent = DQNAgent(state_size, action_size)
    fullfillment_agent = DQNAgent(state_size, action_size)
    while 1:
        #### If new episode brute or RL
        if new_ep:
            if env.rl:
                state = env.reset()
                for time in range(100):
                    # env.render()
                    action = env.action_space.sample() #agent.act(state)
                    next_state, reward, done, _ = env.step(action)
                    reward = reward if not done else -10
                    #next_state = np.reshape(next_state, [1, state_size])
                    agent.remember(state, action, reward, next_state, done,env)
                    state = next_state
                    time += 1
                    print('RL')
                env.rl = False
                continue
            #Take a chance replaying
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
                #Take a chance exploring
                env.reset()
                new_ep = False
                print('Episode End')
        rew, new_ep = move(env, 100)
        if not new_ep and rew <= 0:
            #print('backtracking due to negative reward: %f' % rew)
            _, new_ep = move(env, 70, left=True)
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))

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
    env.rl = True
    while not done:
        if idx >= len(sequence):
            new_state, rew, done, _ = env.step(np.zeros((12,), dtype='bool'))
        else:
            new_state, rew, done, _ = env.step(sequence[idx])
            agent.remember(state, sequence[idx], rew, new_state, done, env) #Train agent from best sequence
            state = new_state
        idx += 1
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
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, env):
        self.memory.append((state, action, reward, next_state, done,
                            env.safety,env.esteem,env.belonging,env.potential))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return #self.env.action_space.sample()  #Fix
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
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
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        self.total_steps_ever = 0
        self.seq_replay = []
        self.memory = deque(maxlen=2000)
        self.is_stuck_pos = deque(maxlen=2000)
        self.is_stuck_pos.append(0)
        self.in_danger_pos = deque(maxlen=2000)
        self.in_danger_pos.append(0)
        self.track_evolution = deque(maxlen=2000)
        self.rl = False
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
        self.total_reward = 0
        self.steps = 0
        return self.env.reset(**kwargs)

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
        if info['rings'] > 0:
            self.safety = True
        else:
            self.safety = False
        # self.track_evolution.append({'reward':self.total_reward,'survival':self.survival,'safety':self.safety,'belonging':self.belonging,
        #                              'esteem':self.esteem,'potential':self.potential})


    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        if self.steps > 1:
            self.maslow(done,info)
        self.steps += 1
        return obs, rew, done, info

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
