import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
import os
from multiprocessing import Pool

class MonteCarloAgent:
    def __init__(self, env, lr=0.001, gamma=0.99, epsilon=1.0, min_epsilon=0.01, decay_rate=0.9995):
        self.env = env
        self.action_count = env.action_space.n
        self.state_dim = env.observation_space.shape
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.memory = deque(maxlen=10000)
        self.model = self._initialize_model()

    def _initialize_model(self):
        model = Sequential([
            Flatten(input_shape=self.state_dim),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_count, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        return model

    def compute_return(self, rewards, step):
        G = sum((self.gamma ** i) * rewards[step + i] for i in range(len(rewards) - step))
        return G

    def store_experience(self, state, action, reward):
        self.memory.append((state, action, reward))

    def train(self):
        visit_count = defaultdict(int)
        Q_values = defaultdict(float)

        for idx, (state, action, reward) in enumerate(self.memory):
            state_key = tuple(state.flatten())  
            visit_count[(state_key, action)] += 1

            for t, (s, a, r) in enumerate(self.memory):
                if np.array_equal(s, state) and a == action and r == reward:
                    break

            G = self.compute_return([r for _, _, r in self.memory], t)
            Q_values[(state_key, action)] += (G - Q_values[(state_key, action)]) / visit_count[(state_key, action)]

        inputs, targets = [], []
        for (state_key, action), q_value in Q_values.items():
            state_array = np.array(state_key).reshape(self.state_dim)
            inputs.append(state_array)
            predicted_qs = self.model.predict(np.expand_dims(state_array, axis=0), verbose=0)[0]
            predicted_qs[action] = q_value
            targets.append(predicted_qs)

        if inputs:
            self.model.fit(np.array(inputs), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_rate

    def select_action(self, state, eval_mode=False):
        if not eval_mode and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_count)
        state = np.expand_dims(state, axis=0)
        predicted_actions = self.model.predict(state, verbose=0)
        return np.argmax(predicted_actions[0])

    def save(self, filename="mc_pacman.h5"):
        self.model.save(filename)

    def load(self, filename="mc_pacman.h5"):
        self.model.load_weights(filename)


def train_agent(episodes=500, max_steps=500):
    env = gym.make("ALE/MsPacman-ram-v5", render_mode="rgb_array")  
    agent = MonteCarloAgent(env)
    scores, epsilon_values = [], []

    for ep in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32) / 255.0
        total_reward, done, step_count = 0, False, 0

        while not done and step_count < max_steps:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32) / 255.0
            done = terminated or truncated

            agent.store_experience(state, action, reward)
            state = next_state
            total_reward += reward
            step_count += 1  

        agent.train()
        scores.append(total_reward)
        epsilon_values.append(agent.epsilon)

        print(f"Episode {ep + 1}/{episodes}, Score: {total_reward}, Steps: {step_count}, Epsilon: {agent.epsilon:.4f}")

        if ep % 50 == 0:
            agent.save()

    env.close()
    plot_training_results(scores, epsilon_values)
    return agent


def evaluate_agent(agent, test_episodes=5):
    env = gym.make("ALE/MsPacman-ram-v5", render_mode="human")  
    total_rewards = []

    for ep in range(test_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32) / 255.0
        total_reward, done = 0, False

        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32) / 255.0
            state = next_state
            total_reward += reward
            done = terminated or truncated

        print(f"Evaluation {ep + 1}/{test_episodes}, Score: {total_reward}")
        total_rewards.append(total_reward)

    env.close()
    avg_score = np.mean(total_rewards)
    print(f"Average Score: {avg_score}")
    return avg_score


def run_simulation(_):
    env = gym.make("ALE/MsPacman-ram-v5", render_mode="rgb_array", frameskip=10)
    agent = MonteCarloAgent(env)
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32) / 255.0
    done, total_reward = False, 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32) / 255.0
        done = terminated or truncated
        total_reward += reward
        state = next_state

    env.close()
    return total_reward


def train_parallel_episodes(num_episodes=100):
    with Pool(processes=4) as pool:
        results = pool.map(run_simulation, range(num_episodes))
    print("Parallel Training Complete | Average Reward:", np.mean(results))


def plot_training_results(scores, epsilon_values):
    os.makedirs("graph", exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(scores, marker='o')
    plt.title("Training Score Progression")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epsilon_values, marker='o', color='r')
    plt.title("Epsilon Decay Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("graph/pacman_mc_learning_curve.png")
    plt.show()


if __name__ == "__main__":
    agent = train_agent(episodes=500)
    agent.save("mc_pacman_final.h5")
    evaluate_agent(agent, test_episodes=5)
