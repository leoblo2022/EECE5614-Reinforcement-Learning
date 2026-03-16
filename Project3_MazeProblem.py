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


# defines the e-greedy policy to find next action
def epsilon_greedy(Q, state, epsilon):

    r, c = state # index of state

    # exploration with prob e
    if random.random() < epsilon: 
         return random.choice(actions)

    # exploitation (greedy) with prob 1-e
    return actions[np.argmax(Q[r,c,:])] # find index of largest Q value for that state, this is your selected action A

# This helper function checks whether a purely greedy policy can find the goal state (a valid path)
def greedy_policy_valid_path_check(Q, start_state, goal_state, max_steps=1000):
    # initialize state and step count 
    current_state = start_state
    step_count = 0
    while step_count < max_steps:

        if current_state == goal_state:
            return True # it found a valid path!
        
        r, c = current_state # get index of current state
        # take the greedy action only 
        best_action_index = np.argmax(Q[r, c, :]) # argmax of Q(s,a)
        action = actions[best_action_index] # find the optimal action
        next_state = move_updates(current_state, action) # find the next state based on the optimal action
        current_state = next_state # update, s = s'
        step_count += 1

    return False # if it can't get to the goal state in 1000 steps or less, it isn't a valid path

# Helper function that attemps to build an optimal policy and path after training 
def build_optimal_policy_and_path(Q, states, state_index, run_idx):
    # Initialize flag
    successful_run_found = False
    # Initialize policy
    policy = [None] * len(states)
    for (r,c) in states: # loop through all states
        s = state_index[(r,c)] # get index of state
        best_action_index = np.argmax(Q[r,c,:]) #find the best action using argmax(Q(s,a))
        policy[s] = actions[best_action_index] # a = pi*(s)
    
    # Using optimal policy, try to build the optimal path
    optimal_path = []
    current_state = start_state
    step_count = 0
    T_max = 1000

    while current_state != goal_state and step_count < T_max:
        s = state_index[current_state] # get index of current state
        action = policy[s] # get optimal action of state based on optimal policy
        optimal_path.append((current_state, action)) # append the state and action to optimal path
        next_state = move_updates(current_state, action) # get next state based on action
        current_state = next_state # update s = s'
        step_count += 1 # keep track of number of steps 
    
    # successful path found in this run
    if current_state == goal_state:
        print(f"Goal reached in run {run_idx+1}")
        successful_run_found = True
        return successful_run_found, policy, optimal_path
    
    # no successful path found in this run 
    else:
        return successful_run_found, policy, optimal_path


# SARSA ALGORITHM
def SARSA_algorithm(states, state_index, p, gamma, alpha, epsilon):
    # Initializations 
    num_rows, num_cols = State_Matrix.shape
    num_runs = 10
    num_episodes = 1000
    max_steps = 1000 # each episode can only last up to 1000 steps
    all_Q = []  # initialize Q-table 
    first_valid_episode = 1000 
    valid_path_flag = False
    avg_rewards = np.zeros((num_runs, num_episodes)) # initialize accumulated reward for each episode and run

    ########### TRAINING of ALGORITHM ####### 
    for run in range(num_runs): #repeat for 10 runs 
        print(f"Run {run+1}") # keep track of which run we are on
        Q = np.zeros((num_rows, num_cols, len(actions))) # Initialize Q(s,a) arbitrarily 

        for episode in range(num_episodes): # limit to 1000 episodes for each run
            
            # STEP 1: THE S IN S-->A-->R-->S'-->A'
            state = start_state # blue starting square 
            r, c = state # keep track of index of current state

            # STEP 2: THE A IN S-->A-->R-->S'-->A'
            action = epsilon_greedy(Q, state, epsilon=epsilon) # choose action according to e_greedy policy, this is A in SARS'A'
            a_idx = action_to_idx[action] # find the index of the action

            episode_reward = 0
            # Repeat (for each step of episode)
            for step in range(max_steps):

                # STEP 3: THE S' IN S-->A-->R-->S'-->A'
                transitions = get_transitions(state, action, p) # get transition probability p(s'|s,a)
                probs = [t[0] for t in transitions] # gets all probabilities based on action
                next_states = [t[1] for t in transitions] # gets all possible next states based on initial action
                idx = np.random.choice(len(next_states), p=probs) # randomly chooses next state based on transition probabilities
                next_state = next_states[idx] # This is S' in "SARS'A' 
                nr, nc = next_state # keep track of location of next state S'

                # STEP 4: THE R IN S-->A-->R-->S'-->A'
                reward = get_reward(state, next_state) # calculate reward  R This is R in SARSA
                episode_reward += reward # accumulate reward for this episode

                # STEP 5: THE A' IN S-->A-->R-->S'-->A'
                next_action = epsilon_greedy(Q, next_state, epsilon=epsilon) # This is A' in SARS'A'
                nexta_idx = action_to_idx[next_action] # find the index of A'


                # STEP 6: Perform SARSA update
                # Q(s,a) = Q(s,a) + alpha*[R + gamma*Q(s',a')-Q(s,a)]
                if next_state == goal_state:
                    Q[r, c, a_idx] = Q[r, c, a_idx] + alpha * (reward - Q[r, c, a_idx])
                else: 
                    Q[r, c, a_idx] = Q[r, c, a_idx] + alpha * (reward + gamma * Q[nr, nc, nexta_idx] - Q[r, c, a_idx])

                # STEP 7: Update states
                state = next_state #s = s', becomes new S in SARSA
                r, c = state # update index as well
                action = next_action # a = a', becomes new A in SARSA
                a_idx = action_to_idx[action] # update action index as well
                
                # STEP 8: Repeat until S is terminal 
                if state == goal_state:
                    break # you have reached goal and are done
            
            # Store accumulated reward per run and episode 
            avg_rewards[run, episode] = episode_reward # store accumulated reward for that specific episode and run

            # Find the first episode to form a valid path during training 
            if not valid_path_flag: # if a valid path has not yet been found 
                if greedy_policy_valid_path_check(Q, start_state, goal_state): #if returns true 
                    first_valid_episode = episode # This is the first episode to find a valid path 
                    valid_path_flag = True
        
        all_Q.append(Q) #update Q-table for that run
    

    ######### AFTER TRAINING ########

    # STEP 1: compute average accumulated reward 
    # calculate the average reward across all 10 runs, per episode number
    average_reward_per_episode = np.mean(avg_rewards, axis=0)
    
    # STEP 2: Report first episode where the greedy policy produced valid path from start to goal
    if first_valid_episode < 1000:
        print("First episode where the greedy policy produced valid path from start to goal for SARSA is episode", first_valid_episode)
    else:
        print("A valid greedy path was never reached for the SARSA algorithm")

    # STEP 3: Find how many times among all runs, a path from start to goal has been obtained
    Q = all_Q[0] # just get values from Q-table 
    successful_run_found = False  # Initialize Flag to check if a run reached the goal

    for run_idx, Q in enumerate(all_Q): #look through Q-values of all 10 runs 
        successful_run_found, policy, path = build_optimal_policy_and_path(Q, states, state_index, run_idx)
        # If this run reached the goal plot the optimal policy and path for this run 
        if successful_run_found:
            optimal_policy = policy
            optimal_path = path

    if successful_run_found:
        return optimal_policy, optimal_path, average_reward_per_episode
    
    # If no run reached the goal, just take a random run
    if not successful_run_found:
        print("No run reached the goal. Using a random run for visualization.")
        run_idx = random.randint(0, len(all_Q)-1) # choose a random run (length of all_Q is 10)
        Q = all_Q[run_idx] #get the Q values for that specific run
        successful_run_found, policy, path = build_optimal_policy_and_path(Q, states, state_index, run_idx)
        return policy, path, average_reward_per_episode
            

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
    print("Evalulating the SARSA Algorithm")
    optimal_policy, optimal_path, avg_accum_reward = SARSA_algorithm(states, state_index, p=0.025, gamma=0.96, alpha=0.25, epsilon=0.1)
    plot_optimal_policy(states, state_index, optimal_policy)
    plot_optimal_path(optimal_path, toggle=True)
    plt.title("Optimal Path")
    plt.show()

    
    plt.figure(figsize=(10,6))
    plt.plot(avg_accum_reward)
    plt.xlabel("Episode")
    plt.ylabel("Average Accumulated Reward")
    plt.title("SARSA: Average Accumulated Reward vs Episode (10 Runs)")
    plt.show()
    

    print("Comparing different learning rates (alpha)")
    alpha_list = [0.05, 0.1, 0.25, 0.5]
    plt.figure(figsize=(10,6))
    for i in range(4):
        optimal_policy, optimal_path, avg_accum_reward = SARSA_algorithm(states, state_index, p=0.025, gamma=0.96, alpha=alpha_list[i], epsilon=0.1)
        plt.plot(avg_accum_reward, label=f"Alpha = {alpha_list[i]}")
        plt.xlabel("Episode")
        plt.ylabel("Average Accumulated Reward")
        plt.title("Average Accumulated Reward for Different Learning Rates using SARSA")
        plt.legend()
    plt.show()



if __name__ == "__main__":
    main()