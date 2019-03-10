# OpenAIGym-Taxi-v2

My simple solution for the Taxi-v2 problem from openAI Gym (https://gym.openai.com/envs/Taxi-v2/). This project is part of the course Machine Learning Engineer Nanodegree. 

This simple solution is based on Temporal Differences Algorithms, in this case Q-Learning (SARSA MAX).

## Solution structure
- main.py: Execution of interaction between the agent and the environment.
- agent.py: Definition of the agent using Q-Learning, based on a epsilon-greedy updates.
- monitor.py: Support class to help during the debug and execution of the environment as measure of the rewards.

For more information about the environemnt check: https://arxiv.org/pdf/cs/9905014.pdf
