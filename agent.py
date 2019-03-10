import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 1
        self.gamma = 1
        self.epsilon = 0.0005

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # get epsilon-greedy action probabilities
        policy_s = self.epsilon_greedy_probs(self.Q[state])
        # pick next action
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # get epsilon-greedy action probabilities (for S')
        policy_s = self.epsilon_greedy_probs(self.Q[next_state])
        # update the knowledge
        self.Q[state][action] = self.update_Q(self.Q[state][action], np.dot(self.Q[next_state], policy_s), reward)
        
    def update_Q(self, Q, Q_next, reward):
        """ Update the agents action-value function
        Params
        ======
        - Q: the current action-value function
        - Qs: the next action-value function
        - reward: last reward received
        """
        return Q + (self.alpha * (reward + (self.gamma * Q_next) - Q))
    
    def epsilon_greedy_probs(self, Q):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[np.argmax(Q)] = 1 - self.epsilon + (self.epsilon /self.nA)
        return policy_s