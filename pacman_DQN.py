import gymnasium as gym
import numpy as np
import random
import cv2 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from collections import deque
import matplotlib.pyplot as plt  
import time 

# ------------------- ENVIRONMENT SETUP -------------------

class PacmanDQNAgent:
    def __init__(self, env_name="ALE/MsPacman-v5", gamma=0.99, alpha=0.0001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, memory_size=100000, frame_skip=4):
        self.env = gym.make(env_name, render_mode="rgb_array", frameskip=frame_skip) 
        self.action_space = self.env.action_space.n 
        self.state_shape = (84, 84, 1) 
        self.gamma = gamma  
        self.alpha = alpha 
        self.epsilon = epsilon  
        self.epsilon_min = epsilon_min 
        self.epsilon_decay = epsilon_decay 
        self.batch_size = batch_size 
        self.memory = deque(maxlen=memory_size)
        self.model = self.build_model()
        self.target_model = self.build_model()  
        self.update_target_model() 

    # ------------------- NEURAL NETWORK (CNN) -------------------

    def build_model(self):
        """Builds a CNN-based Deep Q-Network with explicit Input layer."""
        model = Sequential([
            Input(shape=self.state_shape),  
            Conv2D(32, (8, 8), strides=4, activation='relu'),
            Conv2D(64, (4, 4), strides=2, activation='relu'),
            Conv2D(64, (3, 3), strides=1, activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.action_space, activation='linear')  
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss="mse")
        return model

    def update_target_model(self):
        """Copies weights from policy model to target model."""
        self.target_model.set_weights(self.model.get_weights())

    # ------------------- STATE PREPROCESSING -------------------

    def preprocess_state(self, state):
        """Preprocesses the Pac-Man screen input (grayscale, resize, normalize)."""
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) 
        state = cv2.resize(state, (84, 84)) 
        state = state / 255.0 
        return np.expand_dims(state, axis=-1) 

    # ------------------- ACTION SELECTION (PACMAN EPSILON-GREEDY) -------------------

    def select_action(self, state):
        """Chooses an action using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample() 
        state = np.expand_dims(state, axis=0) 
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values) 

    # ------------------- EXPERIENCE REPLAY -------------------

    def store_experience(self, state, action, reward, next_state, done):
        """Stores experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def train_from_experience(self):
        """Trains the model using experiences from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return  

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            targets[i, actions[i]] = rewards[i] if dones[i] else rewards[i] + self.gamma * np.max(next_q_values[i])

        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

    # ------------------- TRAINING LOOP -------------------

    def train(self, num_episodes=5000, target_update_freq=10):
        """Main training loop."""
        rewards_history = [] 
        epsilon_history = []  
        episode_times = []  

        for episode in range(num_episodes):
            start_time = time.time()  
            state, _ = self.env.reset()
            state = self.preprocess_state(state)
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                done = terminated or truncated 
                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.train_from_experience()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if episode % target_update_freq == 0:
                self.update_target_model()

            rewards_history.append(total_reward)
            epsilon_history.append(self.epsilon)
            episode_time = time.time() - start_time
            episode_times.append(episode_time)
            print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}, Time: {episode_time:.2f}s")

            if total_reward >= max(rewards_history):
                self.model.save("best_pacman_model.h5")
                print(f"New best model saved with reward: {total_reward:.2f}")

        self.plot_training_progress(rewards_history, epsilon_history, episode_times)

    # ------------------- LINE PLOT: Rewards Per Episode for DQN -------------------

    def plot_training_progress(self, rewards_history, epsilon_history, episode_times):
        """Plots the rewards, epsilon values, and episode times over episodes."""
        plt.figure(figsize=(18, 6))

        # Plot rewards
        plt.subplot(1, 3, 1)
        plt.plot(rewards_history, label="Rewards", color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Rewards per Episode")
        plt.legend()

        # Plot epsilon values
        plt.subplot(1, 3, 2)
        plt.plot(epsilon_history, label="Epsilon", color="red")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon Value")
        plt.title("Epsilon Decay over Episodes")
        plt.legend()

        # Plot episode times
        plt.subplot(1, 3, 3)
        plt.plot(episode_times, label="Episode Time", color="green")
        plt.xlabel("Episode")
        plt.ylabel("Time (s)")
        plt.title("Time per Episode")
        plt.legend()

        plt.tight_layout()
        plt.show()

# ------------------- RUNNING THE OUTPUT -------------------

if __name__ == "__main__":
    agent = PacmanDQNAgent(frame_skip=4)  # Use frame skipping for faster training
    agent.train(num_episodes=500)  # Train for 1000 episodes