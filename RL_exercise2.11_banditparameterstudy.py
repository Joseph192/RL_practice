import torch
import matplotlib.pyplot as plt

steps = 200000

# simulate true action values for 10 armed bandit
true_action_values = torch.randn(10)

# initialize epsilon greedy parameter study variables
epsilon = 1/128
E_INIT = epsilon
step_size = 0.1

# number of different parameters to study
epsilon_count = 10
E_COUNT_INIT = epsilon_count

epsilons = [E_INIT * 2**i for i in range(epsilon_count)]  # "epsilon_count" different epsilon values to study

# initialize study reward averages
eps_greedy_reward_avg = torch.zeros(epsilon_count)

# simulate 1000 steps of epsilon-greedy algorithm
def epsilon_greedy(shift_variance, steps=100000):
    global epsilon
    global step_size
    global true_action_values
    # initialize action-value estimates and action counts
    q_estimate = torch.zeros(10)
    n = torch.ones(10)

    reward_history = []

    for i in range(steps):  
        if torch.rand(1).item() < epsilon:  # epsilon = 0.1
            action = torch.randint(0, 10, (1,)).item()  # explore
        else:
            action = torch.argmax(q_estimate).item()

        # simulate reward for action taken
        reward = torch.normal(true_action_values[action], 1).item()
        # true action value drift
        true_action_values += torch.normal(0, shift_variance, (10,))

        # update action-value estimate using recency-weighted method
        q_estimate[action] += step_size * (reward - q_estimate[action])
        n[action] += 1
        
        # record reward history past 1000 steps
        if i >= 100000:
            reward_history.append(reward)

    return reward_history

for i in range(epsilon_count):
    eps_greedy_reward = epsilon_greedy(0.01, steps)
    eps_greedy_reward_avg[i] = torch.mean(torch.tensor(eps_greedy_reward))
    print(epsilon)
    epsilon = epsilon * 2
    

# initialize parameter study variables for gradient bandit algorithm
step_size = 1/128
step_size_count = 10
STEP_SIZE_INIT = step_size
STEP_SIZE_COUNT_INIT = step_size_count

step_sizes = [step_size * 2**i for i in range(step_size_count)]  # "step_size_count" different step sizes to study

gradient_bandit_reward_avg = torch.zeros(step_size_count)

def gradient_bandit(shift_variance, steps=100000):
    global step_size
    global true_action_values
    # initialize action preferences and action counts
    h = torch.zeros(10)
    n = torch.ones(10)

    reward_history_1000 = []
    reward_history = []
    baseline = 0

    for i in range(steps):
        # compute action probabilities using softmax
        action_probs = torch.softmax(h, dim=0)

        # select action based on probabilities
        action = torch.multinomial(action_probs, 1).item()

        # simulate reward for action taken
        reward = torch.normal(true_action_values[action], 1).item()
        reward_history.append(reward)
        # true action value drift
        true_action_values += torch.normal(0, shift_variance, (10,))

        # update action preferences using gradient ascent
        baseline = (reward - baseline) / (i + 1) if reward_history else 0.0
        for a in range(10):
            if a == action:
                h[a] += step_size * (reward - baseline) * (1 - action_probs[a])
            else:
                h[a] -= step_size * (reward - baseline) * action_probs[a]
        
        n[action] += 1
        
        # record reward history past 1000 steps
        if i >= 100000:
            reward_history_1000.append(reward)

    return reward_history_1000

for i in range(step_size_count):
    gradient_bandit_reward = gradient_bandit(0.01, steps)
    gradient_bandit_reward_avg[i] = torch.mean(torch.tensor(gradient_bandit_reward))
    print(step_size)
    step_size = step_size * 2


plt.plot(step_sizes, gradient_bandit_reward_avg.numpy())
plt.plot(epsilons, eps_greedy_reward_avg.numpy())

plt.xscale('log', base=2)
plt.xlabel('Step Size')
plt.xlim(E_INIT/2, STEP_SIZE_INIT * 2**(STEP_SIZE_COUNT_INIT-1) * 2)
plt.xlabel('Parameter Value (Epsilon or Step Size)')

plt.yscale('linear')
plt.ylim(-15,15)
plt.ylabel('Average Reward After 100000 Steps')

plt.title('Parameter study of Bandit Algorithms')
plt.legend(['Gradient Bandit', 'Epsilon-Greedy'])
plt.show()

