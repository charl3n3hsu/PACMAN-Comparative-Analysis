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


def plot_rewards_from_text(text, window_size=10):
    episodes = []
    rewards = []

    for line in text.strip().split("\n"):
        if "Episode" in line and "Total Reward" in line:
            parts = line.split(": Total Reward = ")
            episode = int(parts[0].split("Episode ")[1])
            reward = float(parts[1])
            episodes.append(episode)
            rewards.append(reward)

    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    coefficients = np.polyfit(episodes, rewards, 1)
    regression_line = np.polyval(coefficients, episodes)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Total Reward per Episode", alpha=0.5)

    plt.plot(episodes[:len(moving_avg)], moving_avg, label=f"Moving Average ({window_size} episodes)", color='red')

    plt.plot(episodes, regression_line, label="Linear Regression Trend", color='green', linestyle="dashed")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress - SARSA Agent in Ms. Pac-Man")
    plt.legend()
    plt.show()


printed_rewards = """
Episode 1: Total Reward = 790.0
Episode 2: Total Reward = 570.0
Episode 3: Total Reward = 360.0
Episode 4: Total Reward = 280.0
Episode 5: Total Reward = 300.0
Episode 6: Total Reward = 550.0
Episode 7: Total Reward = 360.0
Episode 8: Total Reward = 400.0
Episode 9: Total Reward = 400.0
Episode 10: Total Reward = 200.0
Episode 11: Total Reward = 380.0
Episode 12: Total Reward = 370.0
Episode 13: Total Reward = 200.0
Episode 14: Total Reward = 300.0
Episode 15: Total Reward = 620.0
Episode 16: Total Reward = 290.0
Episode 17: Total Reward = 180.0
Episode 18: Total Reward = 270.0
Episode 19: Total Reward = 230.0
Episode 20: Total Reward = 190.0
Episode 21: Total Reward = 180.0
Episode 22: Total Reward = 290.0
Episode 23: Total Reward = 330.0
Episode 24: Total Reward = 350.0
Episode 25: Total Reward = 200.0
Episode 26: Total Reward = 400.0
Episode 27: Total Reward = 440.0
Episode 28: Total Reward = 280.0
Episode 29: Total Reward = 280.0
Episode 30: Total Reward = 320.0
Episode 31: Total Reward = 400.0
Episode 32: Total Reward = 290.0
Episode 33: Total Reward = 200.0
Episode 34: Total Reward = 240.0
Episode 35: Total Reward = 290.0
Episode 36: Total Reward = 200.0
Episode 37: Total Reward = 300.0
Episode 38: Total Reward = 560.0
Episode 39: Total Reward = 320.0
Episode 40: Total Reward = 620.0
Episode 41: Total Reward = 300.0
Episode 42: Total Reward = 320.0
Episode 43: Total Reward = 580.0
Episode 44: Total Reward = 320.0
Episode 45: Total Reward = 260.0
Episode 46: Total Reward = 290.0
Episode 47: Total Reward = 370.0
Episode 48: Total Reward = 560.0
Episode 49: Total Reward = 500.0
Episode 50: Total Reward = 300.0
Episode 51: Total Reward = 430.0
Episode 52: Total Reward = 160.0
Episode 53: Total Reward = 720.0
Episode 54: Total Reward = 490.0
Episode 55: Total Reward = 340.0
Episode 56: Total Reward = 200.0
Episode 57: Total Reward = 430.0
Episode 58: Total Reward = 280.0
Episode 59: Total Reward = 360.0
Episode 60: Total Reward = 330.0
Episode 61: Total Reward = 260.0
Episode 62: Total Reward = 540.0
Episode 63: Total Reward = 580.0
Episode 64: Total Reward = 190.0
Episode 65: Total Reward = 290.0
Episode 66: Total Reward = 1200.0
Episode 67: Total Reward = 250.0
Episode 68: Total Reward = 350.0
Episode 69: Total Reward = 390.0
Episode 70: Total Reward = 600.0
Episode 71: Total Reward = 240.0
Episode 72: Total Reward = 440.0
Episode 73: Total Reward = 740.0
Episode 74: Total Reward = 270.0
Episode 75: Total Reward = 410.0
Episode 76: Total Reward = 280.0
Episode 77: Total Reward = 400.0
Episode 78: Total Reward = 320.0
Episode 79: Total Reward = 540.0
Episode 80: Total Reward = 480.0
Episode 81: Total Reward = 350.0
Episode 82: Total Reward = 200.0
Episode 83: Total Reward = 310.0
Episode 84: Total Reward = 700.0
Episode 85: Total Reward = 310.0
Episode 86: Total Reward = 1100.0
Episode 87: Total Reward = 680.0
Episode 88: Total Reward = 590.0
Episode 89: Total Reward = 310.0
Episode 90: Total Reward = 170.0
Episode 91: Total Reward = 280.0
Episode 92: Total Reward = 380.0
Episode 93: Total Reward = 330.0
Episode 94: Total Reward = 320.0
Episode 95: Total Reward = 490.0
Episode 96: Total Reward = 310.0
Episode 97: Total Reward = 250.0
Episode 98: Total Reward = 190.0
Episode 99: Total Reward = 410.0
Episode 100: Total Reward = 690.0
Episode 101: Total Reward = 200.0
Episode 102: Total Reward = 760.0
Episode 103: Total Reward = 980.0
Episode 104: Total Reward = 640.0
Episode 105: Total Reward = 470.0
Episode 106: Total Reward = 300.0
Episode 107: Total Reward = 190.0
Episode 108: Total Reward = 480.0
Episode 109: Total Reward = 480.0
Episode 110: Total Reward = 260.0
Episode 111: Total Reward = 190.0
Episode 112: Total Reward = 280.0
Episode 113: Total Reward = 320.0
Episode 114: Total Reward = 190.0
Episode 115: Total Reward = 460.0
Episode 116: Total Reward = 510.0
Episode 117: Total Reward = 1230.0
Episode 118: Total Reward = 300.0
Episode 119: Total Reward = 600.0
Episode 120: Total Reward = 210.0
Episode 121: Total Reward = 380.0
Episode 122: Total Reward = 400.0
Episode 123: Total Reward = 180.0
Episode 124: Total Reward = 220.0
Episode 125: Total Reward = 230.0
Episode 126: Total Reward = 620.0
Episode 127: Total Reward = 230.0
Episode 128: Total Reward = 500.0
Episode 129: Total Reward = 380.0
Episode 130: Total Reward = 1800.0
Episode 131: Total Reward = 300.0
Episode 132: Total Reward = 340.0
Episode 133: Total Reward = 380.0
Episode 134: Total Reward = 470.0
Episode 135: Total Reward = 760.0
Episode 136: Total Reward = 350.0
Episode 137: Total Reward = 480.0
Episode 138: Total Reward = 410.0
Episode 139: Total Reward = 430.0
Episode 140: Total Reward = 1850.0
Episode 141: Total Reward = 350.0
Episode 142: Total Reward = 450.0
Episode 143: Total Reward = 400.0
Episode 144: Total Reward = 540.0
Episode 145: Total Reward = 240.0
Episode 146: Total Reward = 290.0
Episode 147: Total Reward = 520.0
Episode 148: Total Reward = 590.0
Episode 149: Total Reward = 70.0
Episode 150: Total Reward = 240.0
Episode 151: Total Reward = 470.0
Episode 152: Total Reward = 460.0
Episode 153: Total Reward = 220.0
Episode 154: Total Reward = 330.0
Episode 155: Total Reward = 170.0
Episode 156: Total Reward = 190.0
Episode 157: Total Reward = 410.0
Episode 158: Total Reward = 190.0
Episode 159: Total Reward = 1070.0
Episode 160: Total Reward = 260.0
Episode 161: Total Reward = 500.0
Episode 162: Total Reward = 150.0
Episode 163: Total Reward = 410.0
Episode 164: Total Reward = 560.0
Episode 165: Total Reward = 1050.0
Episode 166: Total Reward = 320.0
Episode 167: Total Reward = 330.0
Episode 168: Total Reward = 370.0
Episode 169: Total Reward = 280.0
Episode 170: Total Reward = 260.0
Episode 171: Total Reward = 320.0
Episode 172: Total Reward = 190.0
Episode 173: Total Reward = 490.0
Episode 174: Total Reward = 1010.0
Episode 175: Total Reward = 300.0
Episode 176: Total Reward = 670.0
Episode 177: Total Reward = 180.0
Episode 178: Total Reward = 350.0
Episode 179: Total Reward = 200.0
Episode 180: Total Reward = 430.0
Episode 181: Total Reward = 430.0
Episode 182: Total Reward = 290.0
Episode 183: Total Reward = 800.0
Episode 184: Total Reward = 380.0
Episode 185: Total Reward = 330.0
Episode 186: Total Reward = 310.0
Episode 187: Total Reward = 360.0
Episode 188: Total Reward = 1000.0
Episode 189: Total Reward = 220.0
Episode 190: Total Reward = 360.0
Episode 191: Total Reward = 330.0
Episode 192: Total Reward = 130.0
Episode 193: Total Reward = 380.0
Episode 194: Total Reward = 550.0
Episode 195: Total Reward = 430.0
Episode 196: Total Reward = 230.0
Episode 197: Total Reward = 370.0
Episode 198: Total Reward = 460.0
Episode 199: Total Reward = 370.0
Episode 200: Total Reward = 320.0
Episode 201: Total Reward = 530.0
Episode 202: Total Reward = 650.0
Episode 203: Total Reward = 370.0
Episode 204: Total Reward = 150.0
Episode 205: Total Reward = 580.0
Episode 206: Total Reward = 290.0
Episode 207: Total Reward = 920.0
Episode 208: Total Reward = 310.0
Episode 209: Total Reward = 290.0
Episode 210: Total Reward = 940.0
Episode 211: Total Reward = 220.0
Episode 212: Total Reward = 200.0
Episode 213: Total Reward = 280.0
Episode 214: Total Reward = 380.0
Episode 215: Total Reward = 400.0
Episode 216: Total Reward = 220.0
Episode 217: Total Reward = 640.0
Episode 218: Total Reward = 630.0
Episode 219: Total Reward = 270.0
Episode 220: Total Reward = 550.0
Episode 221: Total Reward = 520.0
Episode 222: Total Reward = 190.0
Episode 223: Total Reward = 310.0
Episode 224: Total Reward = 580.0
Episode 225: Total Reward = 1010.0
Episode 226: Total Reward = 550.0
Episode 227: Total Reward = 290.0
Episode 228: Total Reward = 310.0
Episode 229: Total Reward = 100.0
Episode 230: Total Reward = 250.0
Episode 231: Total Reward = 240.0
Episode 232: Total Reward = 280.0
Episode 233: Total Reward = 320.0
Episode 234: Total Reward = 290.0
Episode 235: Total Reward = 260.0
Episode 236: Total Reward = 260.0
Episode 237: Total Reward = 960.0
Episode 238: Total Reward = 300.0
Episode 239: Total Reward = 660.0
Episode 240: Total Reward = 250.0
Episode 241: Total Reward = 420.0
Episode 242: Total Reward = 360.0
Episode 243: Total Reward = 760.0
Episode 244: Total Reward = 500.0
Episode 245: Total Reward = 690.0
Episode 246: Total Reward = 150.0
Episode 247: Total Reward = 200.0
Episode 248: Total Reward = 320.0
Episode 249: Total Reward = 290.0
Episode 250: Total Reward = 380.0
Episode 251: Total Reward = 610.0
Episode 252: Total Reward = 490.0
Episode 253: Total Reward = 380.0
Episode 254: Total Reward = 230.0
Episode 255: Total Reward = 190.0
Episode 256: Total Reward = 180.0
Episode 257: Total Reward = 370.0
Episode 258: Total Reward = 330.0
Episode 259: Total Reward = 200.0
Episode 260: Total Reward = 570.0
Episode 261: Total Reward = 140.0
Episode 262: Total Reward = 190.0
Episode 263: Total Reward = 230.0
Episode 264: Total Reward = 380.0
Episode 265: Total Reward = 640.0
Episode 266: Total Reward = 400.0
Episode 267: Total Reward = 370.0
Episode 268: Total Reward = 830.0
Episode 269: Total Reward = 150.0
Episode 270: Total Reward = 200.0
Episode 271: Total Reward = 430.0
Episode 272: Total Reward = 330.0
Episode 273: Total Reward = 1010.0
Episode 274: Total Reward = 280.0
Episode 275: Total Reward = 200.0
Episode 276: Total Reward = 400.0
Episode 277: Total Reward = 380.0
Episode 278: Total Reward = 250.0
Episode 279: Total Reward = 330.0
Episode 280: Total Reward = 370.0
Episode 281: Total Reward = 970.0
Episode 282: Total Reward = 770.0
Episode 283: Total Reward = 620.0
Episode 284: Total Reward = 200.0
Episode 285: Total Reward = 260.0
Episode 286: Total Reward = 190.0
Episode 287: Total Reward = 310.0
Episode 288: Total Reward = 150.0
Episode 289: Total Reward = 490.0
Episode 290: Total Reward = 1030.0
Episode 291: Total Reward = 280.0
Episode 292: Total Reward = 370.0
Episode 293: Total Reward = 400.0
Episode 294: Total Reward = 370.0
Episode 295: Total Reward = 170.0
Episode 296: Total Reward = 210.0
Episode 297: Total Reward = 350.0
Episode 298: Total Reward = 200.0
Episode 299: Total Reward = 170.0
Episode 300: Total Reward = 120.0
Episode 301: Total Reward = 340.0
Episode 302: Total Reward = 290.0
Episode 303: Total Reward = 1020.0
Episode 304: Total Reward = 390.0
Episode 305: Total Reward = 250.0
Episode 306: Total Reward = 370.0
Episode 307: Total Reward = 310.0
Episode 308: Total Reward = 290.0
Episode 309: Total Reward = 200.0
Episode 310: Total Reward = 200.0
Episode 311: Total Reward = 170.0
Episode 312: Total Reward = 410.0
Episode 313: Total Reward = 210.0
Episode 314: Total Reward = 710.0
Episode 315: Total Reward = 210.0
Episode 316: Total Reward = 220.0
Episode 317: Total Reward = 690.0
Episode 318: Total Reward = 420.0
Episode 319: Total Reward = 320.0
Episode 320: Total Reward = 490.0
Episode 321: Total Reward = 910.0
Episode 322: Total Reward = 170.0
Episode 323: Total Reward = 1140.0
Episode 324: Total Reward = 160.0
Episode 325: Total Reward = 1020.0
Episode 326: Total Reward = 260.0
Episode 327: Total Reward = 400.0
Episode 328: Total Reward = 970.0
Episode 329: Total Reward = 180.0
Episode 330: Total Reward = 1020.0
Episode 331: Total Reward = 560.0
Episode 332: Total Reward = 220.0
Episode 333: Total Reward = 470.0
Episode 334: Total Reward = 380.0
Episode 335: Total Reward = 620.0
Episode 336: Total Reward = 450.0
Episode 337: Total Reward = 1040.0
Episode 338: Total Reward = 140.0
Episode 339: Total Reward = 220.0
Episode 340: Total Reward = 200.0
Episode 341: Total Reward = 270.0
Episode 342: Total Reward = 480.0
Episode 343: Total Reward = 270.0
Episode 344: Total Reward = 280.0
Episode 345: Total Reward = 330.0
Episode 346: Total Reward = 640.0
Episode 347: Total Reward = 310.0
Episode 348: Total Reward = 220.0
Episode 349: Total Reward = 740.0
Episode 350: Total Reward = 170.0
Episode 351: Total Reward = 170.0
Episode 352: Total Reward = 280.0
Episode 353: Total Reward = 440.0
Episode 354: Total Reward = 260.0
Episode 355: Total Reward = 360.0
Episode 356: Total Reward = 330.0
Episode 357: Total Reward = 400.0
Episode 358: Total Reward = 290.0
Episode 359: Total Reward = 160.0
Episode 360: Total Reward = 280.0
Episode 361: Total Reward = 420.0
Episode 362: Total Reward = 540.0
Episode 363: Total Reward = 130.0
Episode 364: Total Reward = 100.0
Episode 365: Total Reward = 450.0
Episode 366: Total Reward = 200.0
Episode 367: Total Reward = 150.0
Episode 368: Total Reward = 140.0
Episode 369: Total Reward = 300.0
Episode 370: Total Reward = 140.0
Episode 371: Total Reward = 310.0
Episode 372: Total Reward = 150.0
Episode 373: Total Reward = 180.0
Episode 374: Total Reward = 340.0
Episode 375: Total Reward = 300.0
Episode 376: Total Reward = 670.0
Episode 377: Total Reward = 220.0
Episode 378: Total Reward = 150.0
Episode 379: Total Reward = 320.0
Episode 380: Total Reward = 120.0
Episode 381: Total Reward = 150.0
Episode 382: Total Reward = 270.0
Episode 383: Total Reward = 220.0
Episode 384: Total Reward = 290.0
Episode 385: Total Reward = 140.0
Episode 386: Total Reward = 280.0
Episode 387: Total Reward = 150.0
Episode 388: Total Reward = 120.0
Episode 389: Total Reward = 210.0
Episode 390: Total Reward = 120.0
Episode 391: Total Reward = 170.0
Episode 392: Total Reward = 320.0
Episode 393: Total Reward = 670.0
Episode 394: Total Reward = 300.0
Episode 395: Total Reward = 300.0
Episode 396: Total Reward = 1000.0
Episode 397: Total Reward = 270.0
Episode 398: Total Reward = 340.0
Episode 399: Total Reward = 260.0
Episode 400: Total Reward = 580.0
Episode 401: Total Reward = 1160.0
Episode 402: Total Reward = 580.0
Episode 403: Total Reward = 190.0
Episode 404: Total Reward = 190.0
Episode 405: Total Reward = 370.0
Episode 406: Total Reward = 360.0
Episode 407: Total Reward = 240.0
Episode 408: Total Reward = 230.0
Episode 409: Total Reward = 220.0
Episode 410: Total Reward = 200.0
Episode 411: Total Reward = 400.0
Episode 412: Total Reward = 320.0
Episode 413: Total Reward = 170.0
Episode 414: Total Reward = 180.0
Episode 415: Total Reward = 330.0
Episode 416: Total Reward = 150.0
Episode 417: Total Reward = 310.0
Episode 418: Total Reward = 140.0
Episode 419: Total Reward = 1280.0
Episode 420: Total Reward = 260.0
Episode 421: Total Reward = 300.0
Episode 422: Total Reward = 230.0
Episode 423: Total Reward = 240.0
Episode 424: Total Reward = 280.0
Episode 425: Total Reward = 390.0
Episode 426: Total Reward = 280.0
Episode 427: Total Reward = 120.0
Episode 428: Total Reward = 290.0
Episode 429: Total Reward = 280.0
Episode 430: Total Reward = 170.0
Episode 431: Total Reward = 540.0
Episode 432: Total Reward = 280.0
Episode 433: Total Reward = 290.0
Episode 434: Total Reward = 160.0
Episode 435: Total Reward = 110.0
Episode 436: Total Reward = 160.0
Episode 437: Total Reward = 320.0
Episode 438: Total Reward = 110.0
Episode 439: Total Reward = 190.0
Episode 440: Total Reward = 540.0
Episode 441: Total Reward = 130.0
Episode 442: Total Reward = 220.0
Episode 443: Total Reward = 190.0
Episode 444: Total Reward = 270.0
Episode 445: Total Reward = 230.0
Episode 446: Total Reward = 200.0
Episode 447: Total Reward = 310.0
Episode 448: Total Reward = 180.0
Episode 449: Total Reward = 660.0
Episode 450: Total Reward = 120.0
Episode 451: Total Reward = 340.0
Episode 452: Total Reward = 150.0
Episode 453: Total Reward = 1810.0
Episode 454: Total Reward = 310.0
Episode 455: Total Reward = 270.0
Episode 456: Total Reward = 350.0
Episode 457: Total Reward = 160.0
Episode 458: Total Reward = 120.0
Episode 459: Total Reward = 120.0
Episode 460: Total Reward = 250.0
Episode 461: Total Reward = 940.0
Episode 462: Total Reward = 120.0
Episode 463: Total Reward = 200.0
Episode 464: Total Reward = 490.0
Episode 465: Total Reward = 260.0
Episode 466: Total Reward = 170.0
Episode 467: Total Reward = 180.0
Episode 468: Total Reward = 110.0
Episode 469: Total Reward = 340.0
Episode 470: Total Reward = 330.0
Episode 471: Total Reward = 330.0
Episode 472: Total Reward = 280.0
Episode 473: Total Reward = 610.0
Episode 474: Total Reward = 120.0
Episode 475: Total Reward = 260.0
Episode 476: Total Reward = 350.0
Episode 477: Total Reward = 280.0
Episode 478: Total Reward = 120.0
Episode 479: Total Reward = 310.0
Episode 480: Total Reward = 150.0
Episode 481: Total Reward = 280.0
Episode 482: Total Reward = 110.0
Episode 483: Total Reward = 270.0
Episode 484: Total Reward = 170.0
Episode 485: Total Reward = 170.0
Episode 486: Total Reward = 350.0
Episode 487: Total Reward = 510.0
Episode 488: Total Reward = 190.0
Episode 489: Total Reward = 170.0
Episode 490: Total Reward = 380.0
Episode 491: Total Reward = 310.0
Episode 492: Total Reward = 300.0
Episode 493: Total Reward = 490.0
Episode 494: Total Reward = 220.0
Episode 495: Total Reward = 190.0
Episode 496: Total Reward = 250.0
Episode 497: Total Reward = 210.0
Episode 498: Total Reward = 170.0
Episode 499: Total Reward = 280.0
Episode 500: Total Reward = 120.0
"""

plot_rewards_from_text(printed_rewards)