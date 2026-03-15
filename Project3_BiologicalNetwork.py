import numpy as np
import random
import matplotlib.pyplot as plt

# Global Variables
gamma = 0.90
alpha = 0.25
epsilon = 0.15

# Connectivity matrix
C = np.array([
    [0,  0, -1,  0],
    [1,  0, -1, -1],
    [0,  1,  0,  0],
    [-1, 1,  1,  0]
])

# State Space 
states = [
    np.array([0, 0, 0, 0]), # binary 0
    np.array([0, 0, 0, 1]), # binary 1
    np.array([0, 0, 1, 0]), # binary 2
    np.array([0, 0, 1, 1]), # binary 3
    np.array([0, 1, 0, 0]), # binary 4
    np.array([0, 1, 0, 1]), # binary 5
    np.array([0, 1, 1, 0]), # binary 6
    np.array([0, 1, 1, 1]), # binary 7
    np.array([1, 0, 0, 0]), # binary 8
    np.array([1, 0, 0, 1]), # binary 9
    np.array([1, 0, 1, 0]), # binary 10
    np.array([1, 0, 1, 1]), # binary 11
    np.array([1, 1, 0, 0]), # binary 12
    np.array([1, 1, 0, 1]), # binary 13
    np.array([1, 1, 1, 0]), # binary 14
    np.array([1, 1, 1, 1])  # binary 15
]

# Action space
actions = [
    np.array([0,0,0,0]),  # a1
    np.array([0,1,0,0]),  # a2
    np.array([0,0,1,0]),  # a3
    np.array([0,0,0,1])   # a4
]

action_cost = [0,1,1,0]


# Reward function
def reward(s_next, a_index):
    cost = action_cost[a_index]
    reward = 5*s_next[0] + 5*s_next[1] + 5*s_next[2] + 5*s_next[3] - cost # This is R(s,a,s') 
    

    return reward 


# Epsilon-greedy policy
def epsilon_greedy(Q, state_index, epsilon):

    # exploration with prob e
    if random.random() < epsilon: 
        return random.randint(0,3)
    else:
        # exploitation (greedy) with prob 1-e
        return np.argmax(Q[state_index])


# helper function finds next state 
def next_state(s, a, p):

    # deterministic part
    v = C @ s
    C_sk_1 = (v > 0).astype(int) # threshold to 1 if greater than 0, 0 otherwise
    # noise
    noise = np.random.binomial(1, p, 4) # Bernoulli distribution
    # XOR operation
    s_next = (C_sk_1 ^ a ^ noise)
    return s_next


def visited_states(greedy_policy, p):
    test_episodes = 100
    state_visits = np.zeros(16)
    for episode in range(test_episodes):
        # random initial state
        s = random.choice(states).copy()
        for step in range(100):
            for idx, state in enumerate(states): # find the index of the given state 
                if np.array_equal(state, s):
                    s_index = idx
            # record visit to that state
            state_visits[s_index] += 1

            # greedy action
            a_index = greedy_policy[s_index]

             # Now, find next state given transition probabilities and best action
            s = next_state(s, actions[a_index], p)
    
    return state_visits



def SARSA_algorithm(gamma, alpha, epsilon, p):
    # initializations 
    all_rewards = [] 
    policies = []

    for run in range(2):
        # initializations for each run
        Q = np.zeros((16,4)) # 16 states, each have 4 possible actions
        episode_rewards = []

        for episode in range(1000):
            s = random.choice(states).copy() # start with a random first state, S of SARS'A'
            for idx, state in enumerate(states): # find the index of the given state 
                if np.array_equal(state, s):
                    s_index = idx
            a = epsilon_greedy(Q, s_index, epsilon) #A of SARS'A' using epsilon greedy policy
            total_reward = 0

            for step in range(100):
                s_next = next_state(s, actions[a], p) # find S' of SARS'A'
                s_next_index = int("".join(map(str,s_next)),2)

                r = reward(s_next, a) # This is R of SARS'A'

                a_next = epsilon_greedy(Q, s_next_index, epsilon) # This is A' of SARS'A'

                # SARSA update: Q(s,a) = Q(s,a) + alpha*[R+gamma*Q(s',a')-Q(s,a)]
                Q[s_index, a] = Q[s_index, a] + alpha * (r + gamma * Q[s_next_index, a_next] - Q[s_index, a])

                # Updates: Let s = s' and a = a'
                s = s_next
                s_index = s_next_index
                a = a_next

                total_reward += r # keep track of total reward 

            episode_rewards.append(total_reward) # keep track of total reward per episode 

        all_rewards.append(episode_rewards) # keep track of rewards per run 

        # Keep track of optimal policy for each run
        policy = np.argmax(Q, axis=1)   # best action for each state argmax(Q(s,a))
        policies.append(policy)

    
    # after training, record how many times each of the 16 states is visited during execution
    state_visits = visited_states(policy, p)
    print("State visitation counts:")
    print(state_visits.reshape(16,1))    
    
    action_labels = ['a1','a2','a3','a4'] 

    # Convert 0 to a1, 1 to a2, 2 to a3, and 3 to a4
    for run, policy in enumerate(policies):
        print(f"Run {run+1} optimal policy:")
        for s_index, action in enumerate(policy):
            print(states[s_index], "->", action_labels[action])
            print()

    
    # Average accumulated reward 
    avg_rewards = np.mean(all_rewards, axis=0) # compute average reward across episodes per run

    plt.plot(avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Accumulated Reward")
    plt.title("SARSA: Average Reward vs Episode (10 runs)")
    plt.show()
            

def main():

    SARSA_algorithm(gamma=0.9, alpha = 0.25, epsilon = 0.15, p=0.1)

if __name__ == "__main__":
    main()
