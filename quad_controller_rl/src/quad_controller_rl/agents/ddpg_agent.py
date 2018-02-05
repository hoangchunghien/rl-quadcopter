from keras import layers, models, optimizers, initializers
from keras import backend as K
from quad_controller_rl.agents.base_agent import BaseAgent
import numpy as np
from collections import namedtuple, deque
import random
import os
import pandas as pd
from quad_controller_rl import util


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, size=1000):
        self.size = size
        self.memory = deque(maxlen=self.size)

    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class OUNoise:
    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        self.state = self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = action_high - action_low

        self.build_model()

    def build_model(self):
        """Build an actor network that map states to actions"""
        # Define input layer
        states = layers.Input(shape=(self.state_size,), name='states')
        h0 = layers.BatchNormalization()(states)

        # Add hidden layers
        h1 = layers.Dense(units=32, activation='relu',)(h0)
        h2 = layers.Dense(units=64, activation='relu',)(h1)
        h3 = layers.Dense(units=32, activation='relu')(h2)

        # Add raw output layer with sigmoid activation
        out = layers.Dense(self.action_size, activation='tanh', name='raw_actions')(h3)

        # Scale [0.0, 1.0] output for each action dimension to proper range
        out = layers.Lambda(lambda x: ((x+1)  * self.action_range/2) + self.action_low, name='actions')(out)

        # Create keras model
        self.model = models.Model(inputs=states, outputs=out)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * out)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op
        )


class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        """Build a critic network that maps (state, action) pairs -> Q-values"""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layers for state pathway
        sh0 = layers.BatchNormalization()(states)
        sh1 = layers.Dense(units=32, activation='relu')(sh0)
        sh2 = layers.Dense(units=64, activation='relu')(sh1)

        # Add hidden layers for action pathway
        ah1 = layers.Dense(32, activation='relu')(actions)
        ah2 = layers.Dense(64, activation='relu')(ah1)

        # Combine state and action pathways
        net = layers.Add()([sh2, ah2])
        net = layers.Activation('relu')(net)
        # net = layers.Dense(300, activation='relu')(net)

        # Add output layer to produce action values (Q-values)
        q_values = layers.Dense(units=1, name='q_values')(net)

        # Create a keras model
        self.model = models.Model(inputs=[states, actions], outputs=q_values)

        # Define optimizer and compile model for training
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients
        action_gradients = K.gradients(q_values, actions)

        # Define additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients
        )


class DDPGAgent(BaseAgent):
    """Reinforcement Learning Agent that learns using DDPG."""
    def __init__(self, task):

        self.task = task
        self.state_size = 3  # position only
        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        self.action_size = 3  # force only
        self.action_range = (self.task.action_space.high - self.task.action_space.low)[0:self.action_size]

        # Actor (Policy) model
        self.action_low = self.task.action_space.low[0:self.action_size]
        self.action_high = self.task.action_space.high[0:self.action_size]
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Init target model parameters with local parameters
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.0  # Discount factor
        self.tau = 0.001  # for soft update of target parameters

        self.reset_episode_vars()

        # Save episodes stats
        self.stats_filename = os.path.join(util.get_param('out'), "stats_{}.csv".format(util.get_timestamp()))
        self.stats_columns = ['episode', 'total_reward']
        self.episode_num = 1
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))

    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        return state[0:3]

    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)
        complete_action[0:3] = action
        # complete_action[2] = action[0]
        return complete_action

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0

    def step(self, state, reward, done):
        # Reduce the state vector
        state = self.preprocess_state(state)

        # Choose an action
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)

        # Learn if enough sample are available in the memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        self.last_state = np.copy(state)
        self.last_action = np.copy(action)
        self.total_reward += reward

        if done:
            print("Score: {}".format(self.total_reward))

            # Write episode stats
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1

            # Reset episode vars
            self.reset_episode_vars()

        return self.postprocess_action(action)

    def act(self, states):
        """Return actions for given state(s) as per current policy."""
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        return actions + self.noise.sample()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next_state actions and Q values from target model
        action_nexts = self.actor_target.model.predict_on_batch(next_states)
        q_target_nexts = self.critic_target.model.predict_on_batch([next_states, action_nexts])

        # Compute Q targets for current states and train local critic model
        q_targets = rewards + self.gamma * q_target_nexts * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=q_targets)

        # Train local actor model
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        # Soft update target model
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

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
