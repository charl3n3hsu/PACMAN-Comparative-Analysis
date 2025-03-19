pip install "gymnasium[atari]"
pip install ale-py
pip install tensorflow
pip install --upgrade gymnasium[atari] ale-py

import os
import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from enum import Enum
import matplotlib.pyplot as plt

logger = getLogger(__name__)

class ControllerType(Enum):
    MC = 0
    Sarsa = 1
    Sarsa_lambda = 2
    Q_learning = 3
    REINFORCE = 4
    ActorCritic = 5
    A3C = 6
    PPO = 7

class Config:
    def __init__(self, controller_type):
        self.controller = ControllerConfig(controller_type)
        self.trainer = TrainerConfig()

class TrainerConfig:
    def __init__(self):
        self.num_episodes = 10000
        self.batch_size = 50
        self.lr = 0.0001
        self.evaluate_interval = 500

class ControllerConfig:
    def __init__(self, controller_type):
        self.controller_type = controller_type
        self.epsilon = 0.5
        self.gamma = 0.9
        self.max_workers = 8


class SarsaControl(BaseController):
    def __init__(self, env, config: Config):
        super().__init__()
        self.env = env
        self.epsilon = config.controller.epsilon
        self.gamma = config.controller.gamma
        self.alpha = 0.01  # Learning rate
        self.model = self._initialize_model()
        self.max_workers = config.controller.max_workers
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.q_values_history = []
        self.epsilon_history = []

    def _initialize_model(self):
        model = Sequential([
            Flatten(input_shape=self.env.observation_space.shape),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.env.action_space.n, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def build_training_set(self, buf):
        Q_ = dict()
        states = np.array(buf.states)
        actions = np.array(buf.actions)
        rewards = np.array(buf.rewards)
        num_samples = len(rewards)

        inputs = np.zeros((num_samples,) + self.env.observation_space.shape)
        targets = np.zeros((num_samples, self.env.action_space.n))
        episode_q_values = []

        for i in range(num_samples):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = states[i + 1] if i + 1 < num_samples else None
            next_action = actions[i + 1] if i + 1 < num_samples else None

            inputs[i] = state
            targets[i] = self.model.predict(state[np.newaxis], verbose=0)

            if next_state is None or next_action is None:
                targets[i, action] = reward
            else:
                next_state_tuple = tuple(next_state.flatten())
                if next_state_tuple not in Q_:
                    Q_[next_state_tuple] = self.model.predict(next_state[np.newaxis], verbose=0)[0]
                targets[i, action] = reward + self.gamma * Q_[next_state_tuple][next_action]
                episode_q_values.append(np.max(Q_[next_state_tuple]))

        self.q_values_history.append(np.mean(episode_q_values) if episode_q_values else 0)
        return inputs, targets

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.epsilon_history.append(self.epsilon)

    def plot_metrics(self):
        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Average Q-value", color='tab:blue')
        ax1.plot(self.q_values_history, label="Average Q-value", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel("Epsilon", color='tab:red')
        ax2.plot(self.epsilon_history, label="Epsilon", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        plt.title("Sarsa Training Progress - Q-values and Epsilon")
        plt.show()

env = gym.make("ALE/MsPacman-v5", render_mode="human")

config = Config(ControllerType.Sarsa)

agent = SarsaControl(env, config)

num_episodes = 500
reward_history = []

for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32) / 255.0
    done = False
    total_reward = 0
    buffer = Buffer()

    while not done:
        action = agent.action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32) / 255.0
        buffer.add(state, action, reward)
        total_reward += reward
        state = next_state
        done = terminated or truncated

    reward_history.append(total_reward)
    inputs, targets = agent.build_training_set(buffer)
    agent.model.train_on_batch(inputs, targets)
    agent.update_epsilon()

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

agent.plot_metrics()
env.close()