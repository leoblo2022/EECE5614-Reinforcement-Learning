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
    cost = action_cost[a_index] # get the cost for that action 
    # R(s,a,s') = 5s1' + 5s2' + 5s3' + 5s4' - c(a)
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


# helper function that finds next state based on Bernoulli noise and connectivity matrix 
def next_state(sk_1, ak_1, p):
    # deterministic part
    v_bar = C @ sk_1
    C_sk_1 = (v_bar > 0).astype(int) # threshold to 1 if greater than 0, 0 otherwise
    # noise
    nk = np.random.binomial(1, p, 4) # Bernoulli distribution for j=1,...,4
    # XOR operation
    s_k = (C_sk_1 ^ ak_1 ^ nk)
    return s_k

# This helper function records how many times each state is visted during 1 run
def visited_states(greedy_policy, p):
    # initialization
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
            a_index = greedy_policy[s_index] # a = pi_egreedy(s)

             # Now, find next state given transition probabilities and greedy action
            s = next_state(s, actions[a_index], p) # s = s'
    
    return state_visits


# SARSA ALGORITHM
def SARSA_algorithm(gamma, alpha, epsilon, p):
    # initializations 
    all_rewards = [] 
    policies = []
    num_episodes = 1000
    num_runs = 10
    num_steps = 100

    ########### TRAINING of ALGORITHM ####### 
    for run in range(num_runs):
        print(f"Run {run+1}") # keep track of which run we are on
        # initializations for each run
        Q = np.zeros((16,4)) # Initialize Q(s,a) arbitrarily 
        episode_rewards = []

        for episode in range(num_episodes):

             # STEP 1: THE S IN S-->A-->R-->S'-->A'
            s = random.choice(states).copy() # start with a random first state, S of SARS'A'
            for idx, state in enumerate(states): # find the index of the given state 
                if np.array_equal(state, s):
                    s_index = idx
            
            # STEP 2: THE A IN S-->A-->R-->S'-->A'
            a = epsilon_greedy(Q, s_index, epsilon) #A of SARS'A' using epsilon greedy policy
            total_reward = 0

            # Repeat, for each step of episode 
            for step in range(num_steps):

                # STEP 3: THE S' IN S-->A-->R-->S'-->A'
                s_next = next_state(s, actions[a], p) # find S' of SARS'A'
                s_next_index = int("".join(map(str,s_next)),2)

                # STEP 4: THE R IN S-->A-->R-->S'-->A'
                r = reward(s_next, a) # This is R of SARS'A'

                # STEP 5: THE A' IN S-->A-->R-->S'-->A'
                a_next = epsilon_greedy(Q, s_next_index, epsilon) # This is A' of SARS'A'

                # STEP 6: Perform SARSA update
                # SARSA update: Q(s,a) = Q(s,a) + alpha*[R+gamma*Q(s',a')-Q(s,a)]
                Q[s_index, a] = Q[s_index, a] + alpha * (r + gamma * Q[s_next_index, a_next] - Q[s_index, a])

                # STEP 7: Update states
                s = s_next # s = s' 
                s_index = s_next_index
                a = a_next # a = a'

                total_reward += r # keep track of accumulated reward 

            episode_rewards.append(total_reward) # keep track of  accumulated reward per episode 

        all_rewards.append(episode_rewards) # keep track of accumulared rewards across epusodes for each run  

        # Keep track of optimal policy for each run
        policy = np.argmax(Q, axis=1)   # best action for each state argmax(Q(s,a))
        policies.append(policy)

    ######### AFTER TRAINING ########

    # STEP 1: record how many times each of the 16 states is visited during execution
    state_visits = visited_states(policy, p)
    print("State visitation counts for SARSA algorithm:")
    print(state_visits.reshape(16,1))


    # STEP 2: Show the optimal policy for all independent runs 
    action_labels = ['a1','a2','a3','a4'] 
    # Convert 0 to a1, 1 to a2, 2 to a3, and 3 to a4
    for run, policy in enumerate(policies):
        print(f"Run {run+1} optimal policy:")
        for s_index, action in enumerate(policy):
            print(states[s_index], "->", action_labels[action]) # mapping of state to action
    print()
    
    # STEP 3: Show average accumulated reward across 10 runs with respect to episode number
    avg_rewards = np.mean(all_rewards, axis=0) # compute average reward across episodes per run
    plt.plot(avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Accumulated Reward")
    plt.title("SARSA: Average Reward vs Episode (10 runs)")
    plt.show()

    return avg_rewards
            
# SARSA ALGORITHM
def Q_Learning_algorithm(gamma, alpha, epsilon, p):
    # initializations 
    all_rewards = [] 
    policies = []
    num_episodes = 1000
    num_runs = 10
    num_steps = 100

    ########### TRAINING of ALGORITHM ####### 
    for run in range(num_runs):
        print(f"Run {run+1}") # keep track of which run we are on
        # initializations for each run
        Q = np.zeros((16,4)) # Initialize Q(s,a) arbitrarily 
        episode_rewards = []

        for episode in range(num_episodes):

             # STEP 1: find the state S (initialize randomly)
            s = random.choice(states).copy() # start with a random first state
            for idx, state in enumerate(states): # find the index of the given state 
                if np.array_equal(state, s):
                    s_index = idx
            
            total_reward = 0

            # Repeat, for each step of episode 
            for step in range(num_steps):

                # STEP 2: Choose a from s using policy derived from e_greedy
                a = epsilon_greedy(Q, s_index, epsilon) 

                # STEP 3: find the next state S' according to the action chosen
                s_next = next_state(s, actions[a], p) 
                s_next_index = int("".join(map(str,s_next)),2)

                # STEP 4: Observe the reward Rt+1
                r = reward(s_next, a) 

                # STEP 5: Q-learning update: Find max Q(s', a)
                Q_max = np.max(Q[s_next_index])

                # STEP 6: Perform Q-learning update
                # Q(s,a) = Q(s,a) + alpha*[R+gamma*maxQ(s',a)-Q(s,a)]
                Q[s_index, a] = Q[s_index, a] + alpha * (r + gamma * Q_max  - Q[s_index, a])

                # STEP 7: Update states
                s = s_next # s = s' 
                s_index = s_next_index

                total_reward += r # keep track of accumulated reward 

            episode_rewards.append(total_reward) # keep track of  accumulated reward per episode 

        all_rewards.append(episode_rewards) # keep track of accumulared rewards across epusodes for each run  

        # Keep track of optimal policy for each run
        policy = np.argmax(Q, axis=1)   # best action for each state argmax(Q(s,a))
        policies.append(policy)

    ######### AFTER TRAINING ########

    # STEP 1: record how many times each of the 16 states is visited during execution
    state_visits = visited_states(policy, p)
    print("State visitation counts for Q-learning algorithm:")
    print(state_visits.reshape(16,1))


    # STEP 2: Show the optimal policy for all independent runs 
    action_labels = ['a1','a2','a3','a4'] 
    # Convert 0 to a1, 1 to a2, 2 to a3, and 3 to a4
    for run, policy in enumerate(policies):
        print(f"Run {run+1} optimal policy:")
        for s_index, action in enumerate(policy):
            print(states[s_index], "->", action_labels[action]) # mapping of state to action
    print()
    
    # STEP 3: Show average accumulated reward across 10 runs with respect to episode number
    avg_rewards = np.mean(all_rewards, axis=0) # compute average reward across episodes per run
    plt.plot(avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Accumulated Reward")
    plt.title("Q-Learning: Average Reward vs Episode (10 runs)")
    plt.show()

    return avg_rewards


def main():
    print("Running the SARSA Algorithm")
    avg_rewards_sarsa = SARSA_algorithm(gamma=0.9, alpha = 0.25, epsilon = 0.15, p=0.1)

    print("\nRunning the Q-Learning Algorithm")
    avg_rewards_q_learning = Q_Learning_algorithm(gamma=0.9, alpha = 0.25, epsilon = 0.15, p=0.1)
    
    # plot all accumulated rewards on same plot for different algorithms
    plt.figure(figsize=(10,6))
    plt.plot(avg_rewards_sarsa, label="SARSA")
    plt.xlabel("Episode")
    plt.ylabel("Average Accumulated Reward")
    plt.title("Average Reward vs Episode (10 runs)")
    plt.plot(avg_rewards_q_learning,  label="Q-Learning")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
