import numpy as np

# Global Variables
gamma = 0.99
p = 0.045
theta = 0.01

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
    np.array([1,0,0,0]),  # a2
    np.array([0,1,0,0]),  # a3
    np.array([0,0,1,0]),  # a4
    np.array([0,0,0,1])   # a5
]

# This helper function builds the transition matrix M(a) for each action
def build_transition_matrices(states, actions, C, p):
    num_states = len(states)
    M = []

    for a in actions:
        M_a = np.zeros((num_states, num_states)) # Each M is a SxS (16x16) matrix 

        for i in range(num_states):
            s_i = states[i]
            # Compute Cs^i XOR a
            Csi = C @ s_i # @ is symbol for matrix multiplication
            Csi_bar_bool =  (Csi > 0) # maps each element greater than 0 to TRUE and anything else to FALSE
            Csi_bar = Csi_bar_bool.astype(int) # convert boolean to int (1 and 0)
            CsiXORa = np.bitwise_xor(Csi_bar, a) # component-wise modulo-2 addition XOR between Csibar and a

            for j in range(num_states):
                s_j = states[j]
                # Calculate the expression inside the || || symbol in equation 2
                exponent_expression = np.sum(np.abs(s_j - CsiXORa)) # This is || s^j - (Cs^i XOR a) ||

                # This is the whole expression M(a)ij = p^(|| s^j - (Cs^i XOR a) ||) (1-p)^(4-(|| s^j - (Cs^i XOR a) ||) )
                M_a[i, j] = (p ** exponent_expression) * ((1 - p) ** (4 - exponent_expression)) # Transition matrix for each a

        M.append(M_a) # This stores each M for each a

    return M

# This helper function calculates the Reward R(s,a,s') and expected immediate reward
def reward_function(states, actions, M):
    num_states = len(states) # should be 16
    r = []

    # Loop through each action a
    for a_index in range(len(actions)): 
        a = actions[a_index]
        r_a = np.zeros(num_states) # initialize reward to 0 for all states

        # For each action, loop through each state s
        for i in range(num_states):
            for j in range(num_states): # loop through next states s'
                s_prime = states[j] 
                Reward_s_a_sprime = 5*s_prime[0] + 5*s_prime[1] + 5*s_prime[2] + 5*s_prime[3] - np.sum(a) # This is R(s,a,s')
                # must multiply by transition probability at that state 
                # compute expected immediate reward
                r_a[i] += M[a_index][i, j] * Reward_s_a_sprime # R(s,a) = R(s,a) + M(a)ij * R(s, a, s') 

        r.append(r_a) # Keep track of all rewards for all states and actions

    return r

# Helper function that computes optimal policy from optimal value function.
# Located as final step of value iteration
def extract_policy(V, M, r, gamma):
    num_states = len(V)
    num_actions = len(M)

    policy = np.zeros(num_states, dtype=int)

    for i in range(num_states):

        Q_values = np.zeros(num_actions)

        for a in range(num_actions):
            Q_values[a] = r[a][i] + gamma * np.dot(M[a][i], V)

        policy[i] = np.argmax(Q_values)

    return policy


# Value Iteration (Matrix Form)
def value_iteration(M, r, gamma, theta):
    num_states = len(r[0]) # should be 16
    num_actions = len(M) # should be 4 
    print("Number of states", num_states)
    print("Number of actions", num_actions)

    V = np.zeros(num_states) # should be length of 16

    while True:
        delta = 0
        V_new = np.zeros(num_states)

        # Loop through each state
        for i in range(num_states):

            # Recall that Q(s,a) = max(R(s,a) + gamma*M(a)Vk)
            Q_values = np.zeros(num_actions)

            # Loop through each action
            for a in range(num_actions):
                # Value iteration backup: Vk+1 = max(R(s,a) + gamma*M(a)Vk)
                Q_values[a] = r[a][i] + gamma * np.dot(M[a][i], V)

            V_new[i] = np.max(Q_values) # find the max, this is Vk+1

        delta = np.max(np.abs(V_new - V))
        V = V_new

        # check for convergence/stability
        if delta < theta:
            break

    policy = extract_policy(V, M, r, gamma )
    
    return policy, V


def main():

    M = build_transition_matrices(states, actions, C, p)
    r = reward_function(states, actions, M)

    optimal_policy, V_star = value_iteration(M, r, gamma, theta)
  
    print("\nOptimal Policy:")
    for i in range(16):
        state_str = np.binary_repr(i, width=4)
        print(f"State {state_str} -> Action a{optimal_policy[i] + 1}")

    print("\nOptimal Value Function:")
    print(V_star)
    

if __name__ == "__main__":
    main()
