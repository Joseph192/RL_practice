import numpy as np
import matplotlib.pyplot as plt

def simulate_bandit_greedy():
    # generate true action values
    q_star_values = np.random.normal(loc=0, scale=1, size=10)

    # initialize action-value estimates and action counts
    q_estimate = np.zeros(10)
    n = np.ones(10)
    reward_history = []

    for i in range(1000):   # 1000 steps
        action = np.argmax(q_estimate)

        # simulate true value for each action
        reward = np.random.normal(loc=q_star_values[action], scale=1)

        # update action-value estimate using sample-average method
        q_estimate[action] = q_estimate[action] + (reward - q_estimate[action]) / n[action]
        n[action] = n[action] + 1
        
        # record reward history
        reward_history.append(reward)

    return reward_history

def simulate_bandit_epsilon_greedy():
    # generate true action values
    q_star_values = np.random.normal(loc=0, scale=1, size=10)

    # initialize action-value estimates and action counts
    q_estimate = np.zeros(10)
    n = np.ones(10)
    reward_history = []

    for i in range(1000):   # 1000 steps
        if np.random.rand() < 0.1:  # epsilon = 0.1
            action = np.random.choice(range(10))  # explore
        else:
            action = np.argmax(q_estimate)

        # simulate true value for each action
        reward = np.random.normal(loc=q_star_values[action], scale=1)

        # update action-value estimate using sample-average method
        q_estimate[action] = q_estimate[action] + (reward - q_estimate[action]) / n[action]
        n[action] = n[action] + 1
        
        # record reward history
        reward_history.append(reward)

    return reward_history

average_greedy_reward_step = np.array(simulate_bandit_greedy())
average_epsilon_greedy_reward_step = np.array(simulate_bandit_epsilon_greedy())
count = 1

# 2000 simulations to average reward history
for i in range (1999):
    greedy_reward_history = np.array(simulate_bandit_greedy())
    average_greedy_reward_step = average_greedy_reward_step + (greedy_reward_history - average_greedy_reward_step) / count
    epsilon_greedy_reward_history = np.array(simulate_bandit_epsilon_greedy())
    average_epsilon_greedy_reward_step = average_epsilon_greedy_reward_step + (epsilon_greedy_reward_history - average_epsilon_greedy_reward_step) / count
    count = count + 1


# plot reward history
plt.plot(average_greedy_reward_step, label='Greedy')
plt.plot(average_epsilon_greedy_reward_step, label='Epsilon-Greedy')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward for Greedy Action Selection')
plt.legend()
plt.show()
