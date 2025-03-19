import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize the Ms. Pac-Man environment
env = gym.make("ALE/MsPacman-v5", render_mode="human")

num_episodes = 200
rewards = []
epsilon_vals = []
epsilon = 0.5
speed = 0.001

# Run the environment
for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32) / 255.0  # Normalize
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32) / 255.0  # Normalize
        total_reward += reward
        state = next_state
        done = terminated or truncated
        time.sleep(speed)

    rewards.append(total_reward)
    epsilon_vals.append(epsilon)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Plots
plt.figure(figsize=(12, 6))

# Reward History Plot
plt.subplot(2, 2, 1)
plt.plot(range(num_episodes), rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress - Random Agent in Ms. Pac-Man")

# Epsilon History Plot
plt.subplot(2, 2, 2)
plt.plot(range(num_episodes), epsilon_vals, label="Epsilon", color='orange')
plt.xlabel("Episode")
plt.ylabel("Epsilon Value")
plt.title("Epsilon Progress - Random Agent")
plt.legend()

plt.tight_layout()
plt.show()

env.close()