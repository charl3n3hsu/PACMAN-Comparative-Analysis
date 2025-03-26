# Collaborative Project: PACMAN: A Comparative Analysis of Reinforcement Learning Approaches (DQN, SARSA, and Monte Carlo)

We tested different reinforcement learning models to train an agent to play Ms. Pac-Man. The data consists of game states from the Gymnasium Ms. Pac-Man environment, where each observation includes Pac-Man’s position, ghost locations, and pellet positions. We compared 4 models – a baseline random agent, SARSA, Monte Carlo Control and DQN. 

As expected, the baseline showed no learning or improvement and acted as a benchmark for other models. SARSA, an on-policy model, performed the worst, showing almost negative learning. The Monte Carlo Control model performed slightly better showing very weak improvement. In contrast, DQN performed best amongst all models, but exhibited some fluctuation due to exploration-exploitation tradeoff. 
