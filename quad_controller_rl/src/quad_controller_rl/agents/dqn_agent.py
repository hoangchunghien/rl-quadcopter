from quad_controller_rl.agents.base_agent import BaseAgent
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np
import os
import pandas as pd
from quad_controller_rl import util


FORCE_Z = 2  # Action force-z index
ACTION_SPLIT = 50

GAMMA = 0.9  # Discount rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.996
LEARNING_RATE = 0.001

MEMORY_SIZE = 10000
BATCH_SIZE = 64


class DQNAgent(BaseAgent):
    def __init__(self, task):
        self.task = task

        # Use only (x, y, z) of the state
        self.state_size = 3

        # For action calculation
        self.action_range = (self.task.action_space.high - self.task.action_space.low)[FORCE_Z]
        self.action_low = self.task.action_space.low[FORCE_Z]
        self.action_high = self.task.action_space.high[FORCE_Z]

        # Use only action force z, use only positive actions that matter
        self.action_map = np.arange(self.action_high, 15.0, -4.0)[::-1]
        self.action_size = len(self.action_map)


        # Replay memory
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Build a learning model
        self.model = self._build_model()

        # Exploration rate
        self.epsilon = 3.0

        self.reset_episode_vars()

        # Save episodes stats
        self.stats_filename = os.path.join(util.get_param('out'), "stats_{}.csv".format(util.get_timestamp()))
        self.stats_columns = ['episode', 'total_reward', 'epsilon']
        self.episode_num = 1
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model

    def step(self, state, reward, done):
        # Reduce the state vector
        state = self.preprocess_state(state)

        # Choose an action
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.add_memory(self.last_state, self.last_action, reward, state, done)

        # Learn if enough sample are available in the memory
        if len(self.memory) > BATCH_SIZE:
            self.replay(BATCH_SIZE)

        self.last_state = state
        self.last_action = action
        self.total_reward += reward

        if done:
            print("Score: {}, Epsilon: {}".format(self.total_reward, self.epsilon))

            # Write episode stats
            self.write_stats([self.episode_num, self.total_reward, self.epsilon])
            self.episode_num += 1

            # Save model weights after each 250 episodes
            if self.episode_num % 250 == 0:
                filename = os.path.join(util.get_param('out'), "landing_dqn_weights.h5")
                self.save_weights(filename)

            # Reset episode vars
            self.reset_episode_vars()

        return self.postprocess_action(action)

    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.vstack([state]))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Replay a mini batch of memory to gain experiences"""
        mini_batch = random.sample(self.memory, batch_size)

        # Increase experiences by learn from mini_batch memory
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(self.model.predict(np.vstack([next_state]))[0])
            target_f = self.model.predict(np.vstack([state]))
            target_f[0][action] = target
            self.model.fit(np.vstack([state]), target_f, epochs=1, verbose=0)

        # Experience increased, then decrease the exploration rate
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        return np.array(state[0:3])

    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)
        complete_action[FORCE_Z] = self.action_map[action]
        return complete_action

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0

    def write_stats(self, stats):
        """Write single episode stats to CSV file"""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)
        df_stats.to_csv(self.stats_filename, mode='a', index=False, header=not os.path.isfile(self.stats_filename))

    def load_weights(self, filename):
        """Load DQN model weights from file"""
        self.model.load_weights(filename)

    def save_weights(self, filename):
        """Save DQN model weights to file"""
        self.model.save_weights(filename)