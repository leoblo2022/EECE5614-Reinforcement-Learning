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

bump_penalty = -10

# define action space
actions = ['U', 'D', 'L', 'R']
num_actions = len(actions)
action_to_idx = {a:i for i,a in enumerate(actions)}
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
        reward += bump_penalty

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
    
# This function determines next state location based on transition probabilities
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

# This function plots the optimal path from start state to goal state
def plot_optimal_path(path, toggle):
    # Finally, create a fresh matrix for plotting the optimal path
    if toggle == True:
        plt.subplots(figsize=(13,7.5))
        heatmap = sns.heatmap(State_Matrix, fmt=".2f", linewidths=0.25, linecolor='black', cbar= False, cmap= 'rocket_r')
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
            if toggle == True:
                plt.arrow(c + 0.5, r + 0.5, 0.8, 0, width=0.04, color='black')
            r_new, c_new = r, c + 1
        elif direction == 'L':
            if toggle == True:
                plt.arrow(c + 0.5, r + 0.5, -0.8, 0, width=0.04, color='black')
            r_new, c_new = r, c - 1
        elif direction == 'U':
            if toggle == True:
                plt.arrow(c + 0.5, r + 0.5, 0, -0.8, width=0.04, color='black')
            r_new, c_new = r - 1, c
        elif direction == 'D':
            if toggle == True:
                plt.arrow(c + 0.5, r + 0.5, 0, 0.8, width=0.04, color='black')
            r_new, c_new = r + 1, c
        else:
            r_new, c_new = r, c
            continue # may have hit a wall 

        xs.append(c_new + 0.5)
        ys.append(r_new + 0.5)

    return xs, ys

# This helper function plots the values V(s) at each state on the maze 
def plot_value_function(V, states, state_index):
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

# Helper function that plots the optimal policy pi(s) on the maze
def plot_optimal_policy(states, state_index, policy):
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

# This function simulates the execution of the optimal policy using sampled state transitions
def simulate_episode(policy, state_index, toggle, p, Tmax):
    
    state = start_state
    trajectory = []
    total_reward = 0
    cumulative_rewards = []
    
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

        cumulative_rewards.append(total_reward) # keep track of reward at each time step
        
        trajectory.append((state, actual_action)) # keep track of sample trajectory of states and directions
        
        # update states
        state = next_state

    xs, ys = plot_optimal_path(trajectory, toggle)
    if toggle == True:
        plt.title("Sample Episode Path")
        plt.show()
    print("Sample Episode length:", len(trajectory), "steps")
    print("Total reward:", total_reward, "\n")

    return xs, ys, cumulative_rewards


#This function calculates and plots the average cumulative reward across 10 simulations 
def average_cumulative_reward(policy, state_index, p):
    
    Tmax = 400
    trajectories = []
    taus = []
    num_episodes = 10
    
    # Run 10 simulations
    for i in range(num_episodes):
        x, y, cum_rewards = simulate_episode(policy, state_index, toggle=False, p=p, Tmax=400)
        tau = len(cum_rewards) # termination time (number of steps until reached goal)
        trajectories.append(cum_rewards) #keep track of all cumulative reward trajectories
        taus.append(tau) # keep track of all termination times
    
    # Define T_p
    T_p = max(taus)
    
    # Extend trajectories to length T_p
    cum_reward_full_length = []
    padding = []
    
    for cum_rewards in trajectories:
        if len(cum_rewards) < T_p:
            last_value = cum_rewards[-1] # accesses last element in array
            padding = [last_value] * (T_p - len(cum_rewards)) # find out length difference between Tp and cum_reward array
            cum_reward_full_length.append(cum_rewards + padding) # keep reward constant until it reaches Tp
        else:
            cum_reward_full_length.append(cum_rewards) # if length is same as Tp, keep the same 
    
    cum_reward_full_length = np.array(cum_reward_full_length)
    
    # Calculate Average cumulative reward curve
    G_bar = np.mean(cum_reward_full_length, axis=0)

    plt.plot(range(T_p), G_bar, label=f"p = {p}")
    plt.xlabel("Time step t")
    plt.ylabel("Average cumulative reward")
    plt.legend()
    plt.title("Average Cumulative Reward Curves")

# defines the e-greedy policy to find next action
def epsilon_greedy(Q, state, epsilon):

    r, c = state # index of state

    # exploration with prob e
    if random.random() < epsilon: 
         return random.choice(actions)

    # exploitation (greedy) with prob 1-e
    return actions[np.argmax(Q[r,c,:])] # find index of largest Q value for that state, this is your selected action A

# Performs the SARSA algorithm
def SARSA_algorithm(states, state_index, p, gamma, alpha):
    num_rows, num_cols = State_Matrix.shape
    max_steps = 1000 # each episode can only last up to 1000 steps
    all_Q = []  # initialize Q-table to 0
    goal_reached_count = 0
    num_episode = 0
    for run in range(1): #repeat for 10 runs 
        print(f"Run {run+1}") # keep track of which run we are on
        # Q(s,a)
        Q = np.zeros((num_rows, num_cols, len(actions)))
        for episode in range(1000): # limit to 1000 episodes for each run
            # starting state (blue square)
            state = start_state # "S" in SARSA
            r, c = state # keep track of location of current state
            action = epsilon_greedy(Q, state, epsilon=0.3) # choose action according to pi_e_greedy, this is A in SARS'A'
            a_idx = action_to_idx[action] # find the index of the action
            for step in range(max_steps):
                # transition probabilities
                transitions = get_transitions(state, action, p) # get transition probability p(s'|s,a)
                probs = [t[0] for t in transitions] # gets all probabilities based on action
                next_states = [t[1] for t in transitions] # gets all possible next states based on initial action
                idx = np.random.choice(len(next_states), p=probs) # randomly chooses next state based on transition probabilities
                next_state = next_states[idx] # This is S' in "SARS'A' 
                nr, nc = next_state # keep track of location of next state S'
                reward = get_reward(state, next_state) # calculate reward  R This is R in SARSA
                next_action = epsilon_greedy(Q, next_state, epsilon=0.3) # This is A' in SARS'A'
                nexta_idx = action_to_idx[next_action] # find the index of A'
                if np.isnan(State_Matrix[nr, nc]): # do not update walls (they are not valid states)
                    continue
                # SARSA update
                # Q(s,a) = Q(s,a) + alpha*[R + gamma*Q(s',a')-Q(s,a)]
                Q[r, c, a_idx] = Q[r, c, a_idx] + alpha * (reward + gamma * Q[nr, nc, nexta_idx] - Q[r, c, a_idx])
                # Update states/actions so that next state/action becomes current state/action, to prepare for next iteration
                state = next_state
                action = next_action
                if state == goal_state:
                    goal_reached_count+=1 # keep track of how many times the goal state was reached 
                    num_episode = episode
                    break # you have reached goal and are done
        all_Q.append(Q) #update Q-table 
    
    print("Goal reached:", goal_reached_count, "times")
    print("During training, the first valid path (from start to goal) was produced at episode number", num_episode)
    # average Q over runs
    Q_avg = np.mean(all_Q, axis=0) # This is average Q-values over all 10 runs
    Q = all_Q[0] # just get values from Q-table
    

    # derive optimal value function
    V = np.max(Q_avg, axis=2) # This is V*, which is max(Q(s,a)), last index is action

    # Flag to check if a run reached the goal
    successful_run_found = False

    for run_idx, Q in enumerate(all_Q): #look through Q-values of all 10 runs 
         # Build policy for this run 
        policy = [None] * len(states)
        for (r,c) in states: # loop through all states
            s = state_index[(r,c)] # get index of state
            best_action_index = np.argmax(Q[r,c,:]) #find the best action using argmax(Q(s,a))
            policy[s] = actions[best_action_index] # a = pi*(s)

        # Try to build the optimal path
        optimal_path = []
        current_state = start_state
        step_count = 0
        T_max = 1000

        while current_state != goal_state and step_count < T_max:
            s = state_index[current_state] # get index of current state
            action = policy[s] # get optimal action of state based on optimal policy
            optimal_path.append((current_state, action)) # append the state and action to optimal path
            next_state = move_updates(current_state, action) # get next state based on action
            current_state = next_state
            step_count += 1 # keep track of number of steps 

        if current_state == goal_state:
            print(f"Goal reached in run {run_idx+1}")
            successful_run_found = True
            break  # stop at the first successful run


    # If no run reached the goal, just take a random run
    if not successful_run_found:
        print("No run reached the goal. Using a random run for visualization.")
        run_idx = random.randint(0, len(all_Q)-1) # choose a random run (length of all_Q is 10)
        Q = all_Q[run_idx] #get the Q values for that specific run

        policy = [None] * len(states)
        for (r,c) in states:
            s = state_index[(r,c)]
            best_action_index = np.argmax(Q[r,c,:])
            policy[s] = actions[best_action_index]

        # build a path anyway (may not reach goal)
        optimal_path = []
        current_state = start_state
        step_count = 0
        T_max = 1000

        while current_state != goal_state and step_count < T_max:
            s = state_index[current_state]
            action = policy[s]
            optimal_path.append((current_state, action))
            current_state = move_updates(current_state, action)
            step_count += 1

    return V, policy, optimal_path
            

######################################################################################################################################
# **** MAIN FUNCTION ****
def main():
     # determine location (indices) and number of valid states (anything but walls)
    states = []
    state_index = {}
    for i in range(State_Matrix.shape[0]):
        for j in range(State_Matrix.shape[1]):
            if not np.isnan(State_Matrix[i,j]): # if the cell is not a wall, record its index and add it to a list of valid states
                idx = len(states)
                states.append((i,j))
                state_index[(i,j)] = idx

    # SARSA ALGORITHM
    V_star, optimal_policy, optimal_path = SARSA_algorithm(states, state_index, p=0.025, gamma=0.96, alpha=0.25)
    plot_optimal_policy(states, state_index, optimal_policy)
    plot_optimal_path(optimal_path, toggle=True)
    plt.title("Optimal Path")
    plt.show()
 


if __name__ == "__main__":
    main()