import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
import os
from multiprocessing import Pool

class MonteCarloControl:
    def __init__(self, env, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995):
        self.env = env
        self.action_size = env.action_space.n
        self.state_shape = env.observation_space.shape 
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Flatten(input_shape=self.state_shape),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def compute_return(self, rewards, t):
        G = 0
        for i in range(len(rewards) - t):
            G += (self.gamma ** i) * rewards[t + i]
        return G

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def train(self):
        N = defaultdict(int)  
        Q = defaultdict(float) 

        for idx, (state, action, reward) in enumerate(self.memory):
            state_key = tuple(state.flatten())  # Convert state to tuple
            N[(state_key, action)] += 1

            for t, (s, a, r) in enumerate(self.memory):
                if np.array_equal(s, state) and a == action and r == reward:
                    break

            G = self.compute_return([r for _, _, r in self.memory], t)
            Q[(state_key, action)] += (G - Q[(state_key, action)]) / N[(state_key, action)]

        inputs = []
        targets = []
        for (state_key, action), q in Q.items():
            state_array = np.array(state_key).reshape(self.state_shape)
            inputs.append(state_array)
            target = self.model.predict(np.expand_dims(state_array, axis=0), verbose=0)[0]
            target[action] = q
            targets.append(target)

        if inputs:
            self.model.fit(np.array(inputs), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state, evaluate=False):
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        state = np.expand_dims(state, axis=0)
        action_values = self.model.predict(state, verbose=0)
        return np.argmax(action_values[0])

    def save_model(self, filename="mc_pacman.h5"):
        self.model.save(filename)

    def load_model(self, filename="mc_pacman.h5"):
        self.model.load_weights(filename)


def train_pacman_mc(episodes=500, target_update_freq=10, max_steps=500):
    env = gym.make("ALE/MsPacman-ram-v5", render_mode="human") 
    agent = MonteCarloControl(env)
    scores = []
    epsilon_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32) / 255.0 
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32) / 255.0
            done = terminated or truncated

            agent.remember(state, action, reward)
            state = next_state
            total_reward += reward
            steps += 1  

        agent.train()
        scores.append(total_reward)
        epsilon_history.append(agent.epsilon)

        print(f"Episode {episode + 1}/{episodes}, Score: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")

        if episode % target_update_freq == 0:
            agent.save_model()

    env.close()
    plot_results(scores, epsilon_history)
    return agent


def run_episode(_):
    env = gym.make("ALE/MsPacman-ram-v5", render_mode="human")
    agent = MonteCarloControl(env)
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32) / 255.0
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32) / 255.0
        done = terminated or truncated
        total_reward += reward
        state = next_state

    env.close()
    return total_reward


def train_parallel(num_episodes=100):
    with Pool(processes=4) as pool: 
        results = pool.map(run_episode, range(num_episodes))
    print("Average Reward:", np.mean(results))


def evaluate_pacman(agent, episodes=10):
    env = gym.make("ALE/MsPacman-ram-v5", render_mode="human")
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32) / 255.0
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32) / 255.0
            state = next_state
            total_reward += reward
            done = terminated or truncated

        print(f"Evaluation Episode {episode + 1}/{episodes}, Score: {total_reward}")
        total_rewards.append(total_reward)

    env.close()
    avg_score = np.mean(total_rewards)
    print(f"Average Score: {avg_score}")
    return avg_score


def plot_results(scores, epsilon_history):
    os.makedirs("graph", exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title("Training Score Progression")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.subplot(1, 2, 2)
    plt.plot(epsilon_history)
    plt.title("Epsilon Decay Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig("graph/pacman_mc_learning_curve.png")
    plt.show()


if __name__ == "__main__":
    agent = train_pacman_mc(episodes=3)
    agent.save_model("mc_pacman_final.h5")
    evaluate_pacman(agent, episodes=5)


results = '''
Episode 1/500, Score: 350.0, Steps: 196, Epsilon: 0.9995,
Episode 2/500, Score: 360.0, Steps: 184, Epsilon: 0.9990,
Episode 3/500, Score: 320.0, Steps: 185, Epsilon: 0.9985,
Episode 4/500, Score: 200.0, Steps: 174, Epsilon: 0.9980,
Episode 5/500, Score: 290.0, Steps: 193, Epsilon: 0.9975,
Episode 6/500, Score: 1040.0, Steps: 284, Epsilon: 0.9970,
Episode 7/500, Score: 430.0, Steps: 292, Epsilon: 0.9965,
Episode 8/500, Score: 260.0, Steps: 157, Epsilon: 0.9960,
Episode 9/500, Score: 350.0, Steps: 201, Epsilon: 0.9955,
Episode 10/500, Score: 380.0, Steps: 197, Epsilon: 0.9950,
Episode 11/500, Score: 210.0, Steps: 176, Epsilon: 0.9945,
Episode 12/500, Score: 230.0, Steps: 146, Epsilon: 0.9940,
Episode 13/500, Score: 160.0, Steps: 141, Epsilon: 0.9935,
Episode 14/500, Score: 360.0, Steps: 161, Epsilon: 0.9930,
Episode 15/500, Score: 330.0, Steps: 215, Epsilon: 0.9925,
Episode 16/500, Score: 420.0, Steps: 229, Epsilon: 0.9920,
Episode 17/500, Score: 290.0, Steps: 210, Epsilon: 0.9915,
Episode 18/500, Score: 510.0, Steps: 221, Epsilon: 0.9910,
Episode 19/500, Score: 420.0, Steps: 218, Epsilon: 0.9905,
Episode 20/500, Score: 230.0, Steps: 176, Epsilon: 0.9900,
Episode 21/500, Score: 1040.0, Steps: 281, Epsilon: 0.9896,
Episode 22/500, Score: 340.0, Steps: 257, Epsilon: 0.9891,
Episode 23/500, Score: 350.0, Steps: 208, Epsilon: 0.9886,
Episode 24/500, Score: 260.0, Steps: 188, Epsilon: 0.9881,
Episode 25/500, Score: 340.0, Steps: 189, Epsilon: 0.9876,
Episode 26/500, Score: 210.0, Steps: 168, Epsilon: 0.9871,
Episode 27/500, Score: 160.0, Steps: 160, Epsilon: 0.9866,
Episode 28/500, Score: 270.0, Steps: 188, Epsilon: 0.9861,
Episode 29/500, Score: 310.0, Steps: 196, Epsilon: 0.9856,
Episode 30/500, Score: 280.0, Steps: 172, Epsilon: 0.9851,
Episode 31/500, Score: 350.0, Steps: 181, Epsilon: 0.9846,
Episode 32/500, Score: 270.0, Steps: 151, Epsilon: 0.9841,
Episode 33/500, Score: 210.0, Steps: 187, Epsilon: 0.9836,
Episode 34/500, Score: 640.0, Steps: 297, Epsilon: 0.9831,
Episode 35/500, Score: 290.0, Steps: 211, Epsilon: 0.9826,
Episode 36/500, Score: 340.0, Steps: 192, Epsilon: 0.9822,
Episode 37/500, Score: 350.0, Steps: 178, Epsilon: 0.9817,
Episode 38/500, Score: 520.0, Steps: 228, Epsilon: 0.9812,
Episode 39/500, Score: 430.0, Steps: 242, Epsilon: 0.9807,
Episode 40/500, Score: 920.0, Steps: 301, Epsilon: 0.9802,
Episode 41/500, Score: 340.0, Steps: 201, Epsilon: 0.9797,
Episode 42/500, Score: 310.0, Steps: 194, Epsilon: 0.9792,
Episode 43/500, Score: 410.0, Steps: 207, Epsilon: 0.9787,
Episode 44/500, Score: 1870.0, Steps: 352, Epsilon: 0.9782,
Episode 45/500, Score: 360.0, Steps: 213, Epsilon: 0.9777,
Episode 46/500, Score: 250.0, Steps: 182, Epsilon: 0.9773,
Episode 47/500, Score: 340.0, Steps: 177, Epsilon: 0.9768,
Episode 48/500, Score: 400.0, Steps: 205, Epsilon: 0.9763,
Episode 49/500, Score: 810.0, Steps: 285, Epsilon: 0.9758,
Episode 50/500, Score: 240.0, Steps: 170, Epsilon: 0.9753,
Episode 51/500, Score: 390.0, Steps: 217, Epsilon: 0.9748,
Episode 52/500, Score: 220.0, Steps: 161, Epsilon: 0.9743,
Episode 53/500, Score: 700.0, Steps: 349, Epsilon: 0.9738,
Episode 54/500, Score: 230.0, Steps: 181, Epsilon: 0.9734,
Episode 55/500, Score: 350.0, Steps: 210, Epsilon: 0.9729,
Episode 56/500, Score: 220.0, Steps: 185, Epsilon: 0.9724,
Episode 57/500, Score: 620.0, Steps: 310, Epsilon: 0.9719,
Episode 58/500, Score: 240.0, Steps: 141, Epsilon: 0.9714,
Episode 59/500, Score: 450.0, Steps: 239, Epsilon: 0.9709,
Episode 60/500, Score: 430.0, Steps: 248, Epsilon: 0.9704,
Episode 61/500, Score: 340.0, Steps: 209, Epsilon: 0.9700,
Episode 62/500, Score: 280.0, Steps: 225, Epsilon: 0.9695,
Episode 63/500, Score: 330.0, Steps: 188, Epsilon: 0.9690,
Episode 64/500, Score: 290.0, Steps: 198, Epsilon: 0.9685,
Episode 65/500, Score: 350.0, Steps: 205, Epsilon: 0.9680,
Episode 66/500, Score: 170.0, Steps: 134, Epsilon: 0.9675,
Episode 67/500, Score: 270.0, Steps: 163, Epsilon: 0.9670,
Episode 68/500, Score: 330.0, Steps: 159, Epsilon: 0.9666,
Episode 69/500, Score: 330.0, Steps: 205, Epsilon: 0.9661,
Episode 70/500, Score: 410.0, Steps: 231, Epsilon: 0.9656,
Episode 71/500, Score: 160.0, Steps: 166, Epsilon: 0.9651,
Episode 72/500, Score: 350.0, Steps: 213, Epsilon: 0.9646,
Episode 73/500, Score: 260.0, Steps: 197, Epsilon: 0.9641,
Episode 74/500, Score: 260.0, Steps: 188, Epsilon: 0.9637,
Episode 75/500, Score: 400.0, Steps: 210, Epsilon: 0.9632,
Episode 76/500, Score: 230.0, Steps: 183, Epsilon: 0.9627,
Episode 77/500, Score: 2050.0, Steps: 500, Epsilon: 0.9622,
Episode 78/500, Score: 360.0, Steps: 226, Epsilon: 0.9617,
Episode 79/500, Score: 720.0, Steps: 273, Epsilon: 0.9613,
Episode 80/500, Score: 320.0, Steps: 178, Epsilon: 0.9608,
Episode 81/500, Score: 370.0, Steps: 196, Epsilon: 0.9603,
Episode 82/500, Score: 290.0, Steps: 153, Epsilon: 0.9598,
Episode 83/500, Score: 330.0, Steps: 182, Epsilon: 0.9593,
Episode 84/500, Score: 240.0, Steps: 171, Epsilon: 0.9589,
Episode 85/500, Score: 340.0, Steps: 211, Epsilon: 0.9584,
Episode 86/500, Score: 360.0, Steps: 217, Epsilon: 0.9579,
Episode 87/500, Score: 300.0, Steps: 204, Epsilon: 0.9574,
Episode 88/500, Score: 370.0, Steps: 226, Epsilon: 0.9569,
Episode 89/500, Score: 180.0, Steps: 186, Epsilon: 0.9565,
Episode 90/500, Score: 310.0, Steps: 193, Epsilon: 0.9560,
Episode 91/500, Score: 250.0, Steps: 151, Epsilon: 0.9555,
Episode 92/500, Score: 430.0, Steps: 225, Epsilon: 0.9550,
Episode 93/500, Score: 310.0, Steps: 225, Epsilon: 0.9546,
Episode 94/500, Score: 210.0, Steps: 185, Epsilon: 0.9541,
Episode 95/500, Score: 170.0, Steps: 181, Epsilon: 0.9536,
Episode 96/500, Score: 740.0, Steps: 260, Epsilon: 0.9531,
Episode 97/500, Score: 310.0, Steps: 185, Epsilon: 0.9526,
Episode 98/500, Score: 540.0, Steps: 241, Epsilon: 0.9522,
Episode 99/500, Score: 640.0, Steps: 245, Epsilon: 0.9517,
Episode 100/500, Score: 290.0, Steps: 198, Epsilon: 0.9512,
Episode 101/500, Score: 220.0, Steps: 160, Epsilon: 0.9507,
Episode 102/500, Score: 350.0, Steps: 187, Epsilon: 0.9503,
Episode 103/500, Score: 720.0, Steps: 315, Epsilon: 0.9498,
Episode 104/500, Score: 380.0, Steps: 181, Epsilon: 0.9493,
Episode 105/500, Score: 290.0, Steps: 189, Epsilon: 0.9488,
Episode 106/500, Score: 380.0, Steps: 221, Epsilon: 0.9484,
Episode 107/500, Score: 230.0, Steps: 175, Epsilon: 0.9479,
Episode 108/500, Score: 280.0, Steps: 197, Epsilon: 0.9474,
Episode 109/500, Score: 430.0, Steps: 196, Epsilon: 0.9469,
Episode 110/500, Score: 590.0, Steps: 244, Epsilon: 0.9465,
Episode 111/500, Score: 600.0, Steps: 301, Epsilon: 0.9460,
Episode 112/500, Score: 430.0, Steps: 260, Epsilon: 0.9455,
Episode 113/500, Score: 190.0, Steps: 177, Epsilon: 0.9451,
Episode 114/500, Score: 1200.0, Steps: 330, Epsilon: 0.9446,
Episode 115/500, Score: 440.0, Steps: 215, Epsilon: 0.9441,
Episode 116/500, Score: 420.0, Steps: 209, Epsilon: 0.9436,
Episode 117/500, Score: 410.0, Steps: 217, Epsilon: 0.9432,
Episode 118/500, Score: 260.0, Steps: 162, Epsilon: 0.9427,
Episode 119/500, Score: 320.0, Steps: 177, Epsilon: 0.9422,
Episode 120/500, Score: 180.0, Steps: 142, Epsilon: 0.9418,
Episode 121/500, Score: 210.0, Steps: 181, Epsilon: 0.9413,
Episode 122/500, Score: 1140.0, Steps: 310, Epsilon: 0.9408,
Episode 123/500, Score: 490.0, Steps: 221, Epsilon: 0.9403,
Episode 124/500, Score: 440.0, Steps: 197, Epsilon: 0.9399,
Episode 125/500, Score: 380.0, Steps: 198, Epsilon: 0.9394,
Episode 126/500, Score: 450.0, Steps: 197, Epsilon: 0.9389,
Episode 127/500, Score: 390.0, Steps: 205, Epsilon: 0.9385,
Episode 128/500, Score: 590.0, Steps: 227, Epsilon: 0.9380,
Episode 129/500, Score: 250.0, Steps: 198, Epsilon: 0.9375,
Episode 130/500, Score: 480.0, Steps: 219, Epsilon: 0.9371,
Episode 131/500, Score: 300.0, Steps: 177, Epsilon: 0.9366,
Episode 132/500, Score: 260.0, Steps: 184, Epsilon: 0.9361,
Episode 133/500, Score: 270.0, Steps: 181, Epsilon: 0.9356,
Episode 134/500, Score: 440.0, Steps: 189, Epsilon: 0.9352,
Episode 135/500, Score: 330.0, Steps: 207, Epsilon: 0.9347,
Episode 136/500, Score: 420.0, Steps: 204, Epsilon: 0.9342,
Episode 137/500, Score: 930.0, Steps: 272, Epsilon: 0.9338,
Episode 138/500, Score: 380.0, Steps: 203, Epsilon: 0.9333,
Episode 139/500, Score: 460.0, Steps: 292, Epsilon: 0.9328,
Episode 140/500, Score: 290.0, Steps: 220, Epsilon: 0.9324,
Episode 141/500, Score: 620.0, Steps: 228, Epsilon: 0.9319,
Episode 142/500, Score: 370.0, Steps: 205, Epsilon: 0.9314,
Episode 143/500, Score: 180.0, Steps: 130, Epsilon: 0.9310,
Episode 144/500, Score: 260.0, Steps: 201, Epsilon: 0.9305,
Episode 145/500, Score: 230.0, Steps: 181, Epsilon: 0.9300,
Episode 146/500, Score: 420.0, Steps: 221, Epsilon: 0.9296,
Episode 147/500, Score: 270.0, Steps: 159, Epsilon: 0.9291,
Episode 148/500, Score: 270.0, Steps: 216, Epsilon: 0.9287,
Episode 149/500, Score: 260.0, Steps: 131, Epsilon: 0.9282,
Episode 150/500, Score: 280.0, Steps: 160, Epsilon: 0.9277,
Episode 151/500, Score: 200.0, Steps: 160, Epsilon: 0.9273,
Episode 152/500, Score: 270.0, Steps: 204, Epsilon: 0.9268,
Episode 153/500, Score: 240.0, Steps: 195, Epsilon: 0.9263,
Episode 154/500, Score: 280.0, Steps: 195, Epsilon: 0.9259,
Episode 155/500, Score: 280.0, Steps: 237, Epsilon: 0.9254,
Episode 156/500, Score: 280.0, Steps: 146, Epsilon: 0.9249,
Episode 157/500, Score: 720.0, Steps: 294, Epsilon: 0.9245,
Episode 158/500, Score: 430.0, Steps: 265, Epsilon: 0.9240,
Episode 159/500, Score: 300.0, Steps: 211, Epsilon: 0.9236,
Episode 160/500, Score: 390.0, Steps: 203, Epsilon: 0.9231,
Episode 161/500, Score: 390.0, Steps: 203, Epsilon: 0.9226,
Episode 162/500, Score: 180.0, Steps: 150, Epsilon: 0.9222,
Episode 163/500, Score: 450.0, Steps: 265, Epsilon: 0.9217,
Episode 164/500, Score: 440.0, Steps: 221, Epsilon: 0.9213,
Episode 165/500, Score: 430.0, Steps: 233, Epsilon: 0.9208,
Episode 166/500, Score: 1240.0, Steps: 415, Epsilon: 0.9203,
Episode 167/500, Score: 490.0, Steps: 237, Epsilon: 0.9199,
Episode 168/500, Score: 380.0, Steps: 192, Epsilon: 0.9194,
Episode 169/500, Score: 440.0, Steps: 237, Epsilon: 0.9190,
Episode 170/500, Score: 430.0, Steps: 216, Epsilon: 0.9185,
Episode 171/500, Score: 310.0, Steps: 157, Epsilon: 0.9180,
Episode 172/500, Score: 400.0, Steps: 199, Epsilon: 0.9176,
Episode 173/500, Score: 380.0, Steps: 221, Epsilon: 0.9171,
Episode 174/500, Score: 170.0, Steps: 143, Epsilon: 0.9167,
Episode 175/500, Score: 530.0, Steps: 236, Epsilon: 0.9162,
Episode 176/500, Score: 330.0, Steps: 213, Epsilon: 0.9157,
Episode 177/500, Score: 200.0, Steps: 157, Epsilon: 0.9153,
Episode 178/500, Score: 520.0, Steps: 231, Epsilon: 0.9148,
Episode 179/500, Score: 510.0, Steps: 285, Epsilon: 0.9144,
Episode 180/500, Score: 410.0, Steps: 167, Epsilon: 0.9139,
Episode 181/500, Score: 330.0, Steps: 205, Epsilon: 0.9135,
Episode 182/500, Score: 250.0, Steps: 186, Epsilon: 0.9130,
Episode 183/500, Score: 660.0, Steps: 318, Epsilon: 0.9125,
Episode 184/500, Score: 270.0, Steps: 165, Epsilon: 0.9121,
Episode 185/500, Score: 220.0, Steps: 167, Epsilon: 0.9116,
Episode 186/500, Score: 210.0, Steps: 160, Epsilon: 0.9112,
Episode 187/500, Score: 490.0, Steps: 217, Epsilon: 0.9107,
Episode 188/500, Score: 220.0, Steps: 140, Epsilon: 0.9103,
Episode 189/500, Score: 240.0, Steps: 161, Epsilon: 0.9098,
Episode 190/500, Score: 320.0, Steps: 201, Epsilon: 0.9094,
Episode 191/500, Score: 640.0, Steps: 262, Epsilon: 0.9089,
Episode 192/500, Score: 380.0, Steps: 221, Epsilon: 0.9084,
Episode 193/500, Score: 330.0, Steps: 157, Epsilon: 0.9080,
Episode 194/500, Score: 240.0, Steps: 156, Epsilon: 0.9075,
Episode 195/500, Score: 370.0, Steps: 209, Epsilon: 0.9071,
Episode 196/500, Score: 610.0, Steps: 267, Epsilon: 0.9066,
Episode 197/500, Score: 390.0, Steps: 205, Epsilon: 0.9062,
Episode 198/500, Score: 1900.0, Steps: 313, Epsilon: 0.9057,
Episode 199/500, Score: 110.0, Steps: 112, Epsilon: 0.9053,
Episode 200/500, Score: 310.0, Steps: 197, Epsilon: 0.9048,
Episode 201/500, Score: 270.0, Steps: 224, Epsilon: 0.9044,
Episode 202/500, Score: 750.0, Steps: 263, Epsilon: 0.9039,
Episode 203/500, Score: 350.0, Steps: 210, Epsilon: 0.9035,
Episode 204/500, Score: 660.0, Steps: 247, Epsilon: 0.9030,
Episode 205/500, Score: 380.0, Steps: 184, Epsilon: 0.9026,
Episode 206/500, Score: 230.0, Steps: 138, Epsilon: 0.9021,
Episode 207/500, Score: 380.0, Steps: 196, Epsilon: 0.9017,
Episode 208/500, Score: 300.0, Steps: 212, Epsilon: 0.9012,
Episode 209/500, Score: 420.0, Steps: 213, Epsilon: 0.9008,
Episode 210/500, Score: 380.0, Steps: 224, Epsilon: 0.9003,
Episode 211/500, Score: 170.0, Steps: 121, Epsilon: 0.8999
'''