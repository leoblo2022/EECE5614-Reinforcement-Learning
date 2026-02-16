# First, import necessary libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

# GLOBAL VARIABLES
# Matrix is defined as 20x20 instead of 18x18 stated in the project description in order to treat borders as wall states
State_Matrix = \
    np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.nan],
            [np.nan, 1, 1, 1, 1, np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.nan],
            [np.nan, 1, 1, 1, 1, np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.nan],
            [np.nan, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, np.nan],
            [np.nan, 1, 1, np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.nan],
            [np.nan, 1, 1, np.nan, 1, 1, np.nan, 1, 1, np.nan, 1, 1, 1, 1, 1, np.nan, 1, 1, 1, np.nan],
            [np.nan, 1, 1, np.nan, 1, 1, np.nan, 1, 1, np.nan, 1, 1, np.nan, np.nan, np.nan, np.nan, 1, 1, 1, np.nan],
            [np.nan, 1, 1, 1, 1, 1, np.nan, 1, 1, np.nan, 1, 1, 1, 1, 1, np.nan, 1, 1, 1, np.nan],
            [np.nan, 1, 1, 1, 1, 1, np.nan, 1, 1, np.nan, 1, 1, 1, 1, 1, np.nan, 1, 1, 1, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 1, np.nan, 1, 1, np.nan, np.nan, 1, 1, 1, 1, np.nan, 1, 1, 1, np.nan],
            [np.nan, 1, 1, 1, 1, 1, np.nan, 1, 1, 1, np.nan, 1, 1, np.nan, 1, np.nan, np.nan, np.nan, 1, np.nan],
            [np.nan, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, np.nan, 1, 1, np.nan, 1, 1, 1, np.nan, 1, np.nan],
            [np.nan, 1, 1, 1, 1, 1, 1, np.nan, 1, 1, np.nan, 1, 1, np.nan, 1, 1, 1, np.nan, 1, np.nan],
            [np.nan, 1, 1, 1, 1, 1, 1, np.nan, 1, 1, np.nan, 1, 1, np.nan, 1, 1, 1, 1, 1, np.nan],
            [np.nan, 1, 1, 1, 1, 1, 1, np.nan, 1, 1, 1, 1, 1, np.nan, np.nan, np.nan, np.nan, 1, 1, np.nan],
            [np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.nan],
            [np.nan, np.nan, np.nan, 1, 1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 1, 1, np.nan],
            [np.nan, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])

  # define location of all states
oil_states = [(2,8),(2,16),(4,2),(5,6),(10,18),(15,10),(16,10),(17,14),(17,17),(18,7)]
bump_states = [(1,11),(1,12),(2,1),(2,2),(2,3),(5,1),(5,9),(5,17),(6,17),(7,2),(7,10),(7,11),(7,17),(8,17),(12,11),(12,12),(14,1),(14,2),(15,17),(15,18),(16,7)]
start_state = (15,4)
goal_state = (3,13)

actions = ['U', 'D', 'L', 'R']
action_dict = {
    'U': (-1, 0), # move up a row
    'D': (1, 0),  # move down a row
    'L': (0, -1), # move left a column
    'R': (0, 1) # move right a column
}

# HELPER FUNCTIONS
# function for coloring maze (for visualization purposes)
def coloring_blocks(heatmap, oil_states, bump_states, start_state, end_state):
    # Adding red oil blocks
    for i in range(len(oil_states)):
        heatmap.add_patch(Rectangle((oil_states[i][1], oil_states[i][0]), 1, 1,
                                    fill=True, facecolor='red', edgecolor='red', lw=0.25))
    # Adding salmon bump blocks
    for i in range(len(bump_states)):
        heatmap.add_patch(Rectangle((bump_states[i][1], bump_states[i][0]), 1, 1,
                                    fill=True, facecolor='lightsalmon', edgecolor='lightsalmon', lw=0.25))
    # Adding start block (Blue)
    heatmap.add_patch(Rectangle((start_state[1], start_state[0]), 1, 1,
                                fill=True, facecolor='lightblue', edgecolor='lightblue', lw=0.25))

    # Adding end block (Green)
    heatmap.add_patch(Rectangle((end_state[1], end_state[0]), 1, 1,
                                fill=True, facecolor='lightgreen', edgecolor='lightgreen', lw=0.25))

# Reward Function
def get_reward(s, next_s):
    if s == goal_state:
        return 0  # no more reward after reaches goal, it's done
    
    # any move costs -1
    reward = -1  # action cost
    # if next state is itself because it hit a wall
    if next_s == s:
        reward += -0.8

    if next_s in oil_states:
        reward += -5

    if next_s in bump_states:
        reward += -10

    if next_s == goal_state:
        reward += 200
        
    return reward

# Helper function for updating states after action is taken
def move_updates(state, action):
    i, j = state # get indices of current state 
    move_x, move_y = action_dict[action] # lookup action and directionality
    i_new, j_new = i + move_x, j + move_y 
    
    # if you hit a wall, stay in same state
    if np.isnan(State_Matrix[i_new, j_new]):
        return state
    else:
        return (i_new, j_new) #return the new indices of the next state
    
# determine next state location based on transition probabilities
def get_transitions(state, action, p):
    intended_next_state = move_updates(state, action)

    if state == goal_state:
        return [(1.0, goal_state)]  # goal-state is all absorbing, stay there once reached
    
    if action in ['U','D']:
        perpendicular_moves = ['L','R']
        next_left = move_updates(state, perpendicular_moves[0])
        next_right = move_updates(state, perpendicular_moves[1])
        return [
        (1-p, intended_next_state),
        (p/2, next_left),
        (p/2, next_right)
        ]
    else:
        perpendicular_moves = ['U','D']
        next_up = move_updates(state, perpendicular_moves[0])
        next_down = move_updates(state, perpendicular_moves[1])
        return [
        (1-p, intended_next_state),
        (p/2, next_up),
        (p/2, next_down)
        ]

def plot_path_on_maze(path):
    # Finally, create a fresh matrix for plotting the optimal path
    plt.subplots(figsize=(13,7.5))
    heatmap = sns.heatmap(State_Matrix, fmt=".2f", linewidths=0.25, linecolor='black',
                        cbar= False, cmap= 'rocket_r')
    heatmap.set_facecolor('black') # Color for the NA cells in the state matrix
    coloring_blocks(heatmap, oil_states, bump_states, start_state, goal_state)

    xs = []
    ys = []
    current_state = path[0][0]
    r, c = current_state
    xs.append(c + 0.5)
    ys.append(r + 0.5)
    for state_cr, direction in path:
        r, c = state_cr

        if direction == 'R':
            plt.arrow(c + 0.5, r + 0.5, 0.8, 0, width=0.04, color='black')
            r_new, c_new = r, c + 1
        elif direction == 'L':
            plt.arrow(c + 0.5, r + 0.5, -0.8, 0, width=0.04, color='black')
            r_new, c_new = r, c - 1
        elif direction == 'U':
            plt.arrow(c + 0.5, r + 0.5, 0, -0.8, width=0.04, color='black')
            r_new, c_new = r - 1, c
        elif direction == 'D':
            plt.arrow(c + 0.5, r + 0.5, 0, 0.8, width=0.04, color='black')
            r_new, c_new = r + 1, c
        else:
            r_new, c_new = r, c
            continue # may have hit a wall 

        xs.append(c_new + 0.5)
        ys.append(r_new + 0.5)

    print("length of xs", len(xs))
    print("length of ys", len(ys))
    return xs, ys

def Policy_Iteration(states, state_index, p, gamma, theta):

    n_states = len(states)
    # Initial random policy (pi_0) to all left
    policy = np.array(['L'] * n_states)

    # POLICY ITERATION (Vector Form) 
    converged = False
    iteration_count = 0
    T_max = 400

    while not converged:
        iteration_count +=1 #keep track of number of iterations
        
        # Step 1: POLICY EVALUATION (Approximate method, which is better for large state spaces)
        V = np.zeros(n_states) #initialize as vector Sx1

        while True:
            delta = 0 # initialize as 0

            # Iterate through all states 
            for s in states: 
                s_idx = state_index[s] # get index of each state
                v = V[s_idx]   # old value
            
                action = policy[s_idx] # get mapping of state to action for current policy 
                transitions = get_transitions(s, action, p) # get transition matrix P(s'|s,a)
            
                new_v = 0
                # formatted as [0.90, next_state]
                for prob, next_s in transitions:
                    next_idx = state_index[next_s] # index of s'
                    r = get_reward(s, next_s) # compute reward for next state R(s,a,s')
                    # Bellman Equation: V(s) = sum of p(s'|s,a)[R(s,a,s')+gamma*V(s')]
                    new_v += prob * (r + gamma * V[next_idx]) 
            
                V[s_idx] = new_v # update value 
            
                delta = max(delta, abs(v - new_v)) # compute delta
        
            if delta < theta: #convergence criteria
                break


        # Step 2: POLICY IMPROVEMENT 
        converged = True
        
        # Iterate through all states
        for s in states:
            s_idx = state_index[s]
            old_action = policy[s_idx]
            
            Q_values = []
            
            # Recall Bellman optimality: Q*(a,s) = sum p(s'|s,a)[R(s,a,s')+gamma*V(s')]
            for action in actions:
                q = 0
                # This returns p(s'|s,a) for the given state and action
                transitions = get_transitions(s, action, p)

                # for example, formatted as [0.90, next_state]
                for prob, next_s in transitions:
                    next_idx = state_index[next_s]
                    r = get_reward(s, next_s)
                    # This is p(s'|s,a)[R(s,a,s')+gamma*V(s')]
                    q += prob * (r + gamma * V[next_idx])
                    
                Q_values.append(q)
            
            best_action = actions[np.argmax(Q_values)]
            
            if best_action != old_action:
                converged = False
                
            # Update policy: pi_t-1 --> pi_t
            policy[s_idx] = best_action

    # Extract Optimal Path
    optimal_path = []
    current = start_state

    while current != goal_state:
        action = policy[state_index[current]]
        optimal_path.append((current, action))  # ← must store the tuple
        current = move_updates(current, action)


    # VISUALIZE RESULTS
    print("Iteration Count: ", iteration_count)
    # Create a fresh matrix for plotting the values
    # plot the value function values on the heat map
    plt.subplots(figsize=(13,7.5))
    Value_Matrix = np.full(State_Matrix.shape, np.nan)

    for (i, j) in states:
        idx = state_index[(i, j)]
        # Assign new 2D matrix with the value function value at the current state
        Value_Matrix[i, j] = V[idx]

    # Plot the new heatmap of the new value function values with the original state and coloring blocks
    heatmap = sns.heatmap(Value_Matrix, fmt=".2f", annot= Value_Matrix, linewidths=0.25, linecolor='black',
                        cbar= False, cmap= 'rocket_r', annot_kws={"size": 8})

    heatmap.set_facecolor('black') # Color for the NA cells in the state matrix
    coloring_blocks(heatmap, oil_states, bump_states, start_state, goal_state)
    plt.title("Optimal Value Function")
    plt.show()

    # Create a fresh matrix for plotting the voptimal policy (for all states)
    # plot the value function values on the heat map
    plt.subplots(figsize=(13,7.5))
    heatmap = sns.heatmap(State_Matrix, fmt=".2f", linewidths=0.25, linecolor='black',
                        cbar= False, cmap= 'rocket_r')
    heatmap.set_facecolor('black') # Color for the NA cells in the state matrix
    coloring_blocks(heatmap, oil_states, bump_states, start_state, goal_state)

    # go through each row and column
    for (r, c) in states:
        if (r, c) == goal_state:
            continue  # no arrow at goal

        action = policy[state_index[(r,c)]]

        if action == 'R':
            plt.arrow(c + 0.5, r + 0.5, 0.6, 0, width=0.04, color='black')
        elif action == 'L':
            plt.arrow(c + 0.5, r + 0.5, -0.6, 0, width=0.04, color='black')
        elif action == 'U':
            plt.arrow(c + 0.5, r + 0.5, 0, -0.6, width=0.04, color='black')
        elif action == 'D':
            plt.arrow(c + 0.5, r + 0.5, 0, 0.6, width=0.04, color='black')

    plt.title("Optimal Policy")
    plt.show()

    plot_path_on_maze(optimal_path)
    plt.title("Optimal Path")
    plt.show()
    print("Optimal Path length:", len(optimal_path), "steps")

    return policy

# This function simulates the execution of the optimal policy using sampled state transitions
def simulate_episode(policy, start_state, state_index, p, Tmax):
    
    state = start_state
    trajectory = []
    total_reward = 0
    
    # max limit is 400 steps
    for t in range(Tmax):
        
        # terminate if goal state is reached 
        if state == goal_state:
            break
        
        # get the appropriate action for each state 
        s_idx = state_index[state]
        action = policy[s_idx]
        
        transitions = get_transitions(state, action, p) # p(s'|a,s)
        
        # Sample next state using probabilities
        # example, probs = [0.90, 0.05, 0,05]
        probs = [prob for prob, _ in transitions]
        # get possible next states 
        next_states = [next_s for _, next_s in transitions]
        
        # based on probabilities, get next state 
        next_state = random.choices(next_states, weights=probs)[0]

        state_row, state_column = state
        next_state_row, next_state_column = next_state

        # determine which action the episode actually took due to stochasticity 
        if (next_state_row > state_row) and  (state_column == next_state_column):
            actual_action = 'D'
        elif (next_state_row < state_row) and  (state_column == next_state_column):
            actual_action  = 'U'
        elif (next_state_row == state_row) and  (state_column < next_state_column):
            actual_action  = 'R'
        elif (next_state_row == state_row) and  (state_column > next_state_column):
            actual_action  = 'L'
        else: 
            actual_action  = None # hit wall

        # keep track of total reward
        reward = get_reward(state, next_state)
        total_reward += reward
        
        trajectory.append((state, actual_action)) # keep track of sample trajectory of states and directions
        
        # update states
        state = next_state
    
    xs, ys = plot_path_on_maze(trajectory)
    plt.title("Sample Episode Path")
    plt.show()
    print("Sample Episode length:", len(trajectory), "steps")
    print("Total reward:", total_reward)

    return xs, ys


def plot_two_trajectories(x1, y1, x2, y2):
    plt.figure(figsize=(10,7.5))
    heatmap = sns.heatmap(State_Matrix, linewidths=0.25, linecolor='black', cbar=False, cmap='rocket_r')
    heatmap.set_facecolor('black')
    coloring_blocks(heatmap, oil_states, bump_states, start_state, goal_state)

    plt.plot(x1, y1, linestyle='-', marker='o', markersize=4, label='Episode 1')
    plt.plot(x2, y2, linestyle='--', marker='s', markersize=4, label='Episode 2')

    plt.legend()
    plt.title("Two Independent Sample Trajectories")
    plt.show()


def main():
    
    plt.subplots(figsize=(10,7.5))
    heatmap = sns.heatmap(State_Matrix, fmt=".2f", linewidths=0.25, linecolor='black',
                      cbar= False, cmap= 'rocket_r', vmin=0, vmax=255)
    heatmap.set_facecolor('black') # Color for the NaN cells in the state matrix
    plt.title('Maze Problem')

    # color blocks for visualization purposes
    coloring_blocks(heatmap, oil_states, bump_states, start_state, goal_state)
    plt.show()

    # determine location (indices) and number of valid states (anything but walls)
    states = []
    state_index = {}
    for i in range(State_Matrix.shape[0]):
        for j in range(State_Matrix.shape[1]):
            if not np.isnan(State_Matrix[i,j]): # if the cell is not a wall, record its index and add it to a list of valid states
                idx = len(states)
                states.append((i,j))
                state_index[(i,j)] = idx

    print("Base Scenario of Policy Iteration")
    print(" ")
    optimal_policy = Policy_Iteration(states, state_index, p=0.02, gamma=0.99, theta=0.01) # Base Scenario
    x1, y1 = simulate_episode(optimal_policy, start_state, state_index, p=0.02, Tmax=400)
    print(" ")
    print("Large Stochasticity Scenario of Policy Iteration")
    print(" ")
    optimal_policy = Policy_Iteration(states, state_index, p=0.4, gamma=0.99, theta=0.01) # Large Stochasticity  Scenario
    x1, y1 = simulate_episode(optimal_policy, start_state, state_index, p=0.4, Tmax=400)
    print(" ")
    print("Small Discount Factor Scenario of Policy Iteration")
    print(" ")
    #Policy_Iteration(states, state_index, p=0.02, gamma=0.4, theta=0.01) # Small Discount Factor  Scenario
    #print(" ")

    # EFFECT OF STOCHASTICITY
    optimal_policy = Policy_Iteration(states, state_index, p=0.02, gamma=0.99, theta=0.01) # Base Scenario
    x1, y1 = simulate_episode(optimal_policy, start_state, state_index, p=0.02, Tmax=400)
    x2, y2 = simulate_episode(optimal_policy, start_state, state_index, p=0.02, Tmax=400)
    plot_two_trajectories(x1, y1, x2, y2)
    optimal_policy = Policy_Iteration(states, state_index, p=0.2, gamma=0.99, theta=0.01) # high stochasticity
    x1, y1 = simulate_episode(optimal_policy, start_state, state_index, p=0.2, Tmax=400)
    x2, y2 = simulate_episode(optimal_policy, start_state, state_index, p=0.2, Tmax=400)
    plot_two_trajectories(x1, y1, x2, y2)
    optimal_policy = Policy_Iteration(states, state_index, p=0.4, gamma=0.99, theta=0.01) # very high stochasticity
    x1, y1 = simulate_episode(optimal_policy, start_state, state_index, p=0.4, Tmax=400)
    x2, y2 = simulate_episode(optimal_policy, start_state, state_index, p=0.4, Tmax=400)
    plot_two_trajectories(x1, y1, x2, y2)



if __name__ == "__main__":
    main()

