# First, import necessary libraries
from collections import deque
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# GLOBAL VARIABLES
# Matrix is defined as 10x10 instead of 8x8 stated in the project description in order to treat borders as wall states
State_Matrix = \
    np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 1, 1, 1, 1, 1, 1, 1, 1, np.nan],
            [np.nan, 1, 1, 1, np.nan, 1, 1, 1, 1, np.nan],
            [np.nan, 1, 1, np.nan, np.nan, np.nan, np.nan, 1, np.nan, np.nan],
            [np.nan, 1, 1, 1, np.nan, 1, 1, 1, 1, np.nan],
            [np.nan, 1, 1, 1, 1, 1, np.nan, 1, 1, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 1, 1, np.nan, 1, 1, np.nan],
            [np.nan, 1, 1, 1, 1, 1, np.nan, 1, 1, np.nan],
            [np.nan, 1, 1, 1, 1, 1, np.nan, 1, 1, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])

# define location of all states
red_states = [(3,2), (6,4), (7,3), (7,7)]
yellow_states = [(1,2), (3,7), (5,2), (5,7)]
start_state = (7,2)
goal_state = (1,6)

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

# Hyperparameters
N_epi = 1000
T_epi = 50
gamma = 0.97
alpha = 0.01 # 10^-2
replay_memory_size = 10000
batch_size = 64
N_QU = 5
eta = 0.01 # 10^-2

# DEFINE ALL CLASSES
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    

class ReplayMemory:
    def __init__(self, capacity):
        self.replayMemory = deque(maxlen=capacity)

    # add the most recent experience (s,a,r,s',done) to replay memory
    def push(self, s, a, r, s_next, done):
        self.replayMemory.append((s, a, r, s_next, done))

    # randomly select a minibatch of size Nbatch from the experiences stored in the replay memory
    def sample(self, batch_size):
        batch = random.sample(self.replayMemory, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.array(s_next),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.replayMemory)
    

# HELPER FUNCTIONS
# function for coloring maze (for visualization purposes)
def coloring_blocks(heatmap, red_states, yellow_states, start_state, end_state):
    # Adding red blocks
    for i in range(len(red_states)):
        heatmap.add_patch(Rectangle((red_states[i][1], red_states[i][0]), 1, 1,
                                    fill=True, facecolor='red', edgecolor='red', lw=0.25))
    # Adding yllow blocks
    for i in range(len(yellow_states)):
        heatmap.add_patch(Rectangle((yellow_states[i][1], yellow_states[i][0]), 1, 1,
                                    fill=True, facecolor='yellow', edgecolor='yellow', lw=0.25))
    # Adding start block (Blue)
    heatmap.add_patch(Rectangle((start_state[1], start_state[0]), 1, 1,
                                fill=True, facecolor='blue', edgecolor='blue', lw=0.25))

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

    if next_s in red_states:
        reward += -10

    if next_s in yellow_states:
        reward += -5

    if next_s == goal_state:
        reward += 100
        
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

# Helper function that takes a state's index and normalizes it to between 0 and 1
def normalize_state(state, grid_size):
    x, y = state
    return np.array([x / (grid_size - 1), y / (grid_size - 1)], dtype=np.float32)

# Uses epsilon greedy of the Q-values to determine action a+t
def select_action_epsilon_greedy(state, epsilon, Q_network):
    if random.random() < epsilon:
        return random.randint(0, 3)  # 4 actions
    else:
        with torch.no_grad():
            state_t = torch.tensor(state).unsqueeze(0)
            q_values = Q_network(state_t)
            return q_values.argmax().item()

# Performs the training step by minimizing the loss function
def train_step(Q_network, target_network, optimizer, batch, gamma):
    states, actions, rewards, next_states, dones = batch # get all (s,a,r,s',done) tuples from minibatch
    states = torch.tensor(states)
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards).unsqueeze(1)
    next_states = torch.tensor(next_states)
    dones = torch.tensor(dones).unsqueeze(1)

    # Current Q-values in Q_network
    q_values = Q_network(states).gather(1, actions)

    # Target Q-values in target network
    with torch.no_grad():
        max_next_q = target_network(next_states).max(1, keepdim=True)[0]
        # Zi = ri + gamma* max(Q^w-)*(1-donei)
        z = rewards + gamma * max_next_q * (1 - dones)


    # Define the loss function: L(z,w,w-) = summation of (z-Qw(si,ai))^2
    loss = F.mse_loss(q_values, z)

    # Minimize the loss function using a stochastic gradient optimization technique (using Adam opimizer)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# soft update approach which provides a slow update that helps ensure the stability of the training process,
def soft_update(Q_network, target_network, eta):
    for target_w, Q_w in zip(target_network.parameters(), Q_network.parameters()):
        target_w.data.copy_(
            eta * Q_w.data + (1 - eta) * target_w.data
        )

# Helper function that computes the optimal policy pi(s), path, and state values V(s) after training
def compute_policy_path_and_values(Q_network, states, state_index, grid_size=10):
    policy = {}      # maps (i,j) → action ('U','D','L','R')
    state_values = {}  # maps (i,j) → max Q-value

    Q_network.eval()

    with torch.no_grad():
        # Loop through all states
        for state in states:
            state_norm = normalize_state(state, grid_size)
            state_t = torch.tensor(state_norm).unsqueeze(0)

            q_values = Q_network(state_t).squeeze()  # get Q-values from Q network

            # Find best action by using the argmax of the Q-values
            best_action_idx = torch.argmax(q_values).item()
            best_action = actions[best_action_idx]

            # Add final state values and optimal policy pi(s) for each state 
            policy[state] = best_action
            state_values[state] = torch.max(q_values).item()

    # Next, compute optimal path, starting from start_state
    state = start_state
    path = [state]

    for i in range(100): # define max steps so it doesn't continue forever
        if state == goal_state:
            break
        
        # extract the optimal policy for each state in the path
        action = policy[state]
        next_state = move_updates(state, action) # compute the next state based on state and action

        path.append(next_state) # append to final optimal path
        state = next_state # update s = s'

    return policy, path, state_values

# Deep Q Network Algorithm
def deep_Q_network(Q_network, target_network, states, state_index, optimizer, memory):
    # Initialize average and episodic reward, loss, and length
    Epi_Rewards = []
    Epi_Losses = []
    Epi_Lengths = []
    Avg_Rewards = []
    Avg_Losses = []
    Avg_Lengths = []

    target_network.load_state_dict(Q_network.state_dict())

    # TRAINING LOOP
    for episode in range(N_epi):
        episode_reward = 0
        episode_loss = 0
        episode_steps = 0

        # epsilon decay
        epsilon = max(0.1, 0.995 ** episode)

        # random initial state
        state = random.choice(states)
        idx = state_index[state]
        state_norm = normalize_state(state, grid_size=10)

        for t in range(T_epi):
            episode_steps += 1
            # Choose action based on epsilon greedy of Q-values produced by Q_network
            action_idx = select_action_epsilon_greedy(state_norm, epsilon, Q_network)
            action = actions[action_idx]  # convert index → string

            # get next state based on transition probabilities
            transitions = get_transitions(state, action, p=0.025)
            probs = [t[0] for t in transitions] # gets all probabilities based on action
            next_states = [t[1] for t in transitions] # gets all possible next states based on initial action
            idx = np.random.choice(len(next_states), p=probs) # randomly chooses next state based on transition probabilities
            next_state = next_states[idx] # get next state
            # normalize next state index 
            next_state_norm = normalize_state(next_state, grid_size=10)

            # Observe reward
            reward = get_reward(state, next_state)
            episode_reward += reward

            # update the done variable 
            if next_state == goal_state: 
                done = 1
            else:
                done = 0

            # Add most recent (s,a,r,s',done) to replay memory D
            memory.push(state_norm, action_idx, reward, next_state_norm, done)

            # update: S = S'
            state = next_state
            state_norm = next_state_norm    # normalized

            # Training step
            if len(memory) >= batch_size and t % N_QU == 0:
                batch = memory.sample(batch_size)
                loss = train_step(Q_network, target_network, optimizer, batch, gamma)
                soft_update(Q_network, target_network, eta)
                episode_loss += loss

            if done:
                break

        # After each episode ends, append episode reward, loss, and length
        Epi_Rewards.append(episode_reward)
        Epi_Losses.append(episode_loss)
        Epi_Lengths.append(episode_steps)

        # Compute moving average of accumulated reward, loss, and length
        k = len(Epi_Rewards)
        m = min(25, k)
        sum_rewards = 0
        sum_loss = 0
        sum_length = 0
        for j in range(m):
            sum_rewards = sum_rewards + Epi_Rewards[k - 1 - j]
            sum_loss = sum_loss + Epi_Losses[k - 1 - j] 
            sum_length = sum_length + Epi_Lengths[k - 1 - j]
        avg_reward = sum_rewards / m
        avg_loss = sum_loss / m
        avg_length = sum_length / m

        Avg_Rewards.append(avg_reward)
        Avg_Losses.append(avg_loss)
        Avg_Lengths.append(avg_length)

    policy, path, state_values = compute_policy_path_and_values(Q_network, states, state_index, grid_size=10)
    
    return Avg_Rewards, Avg_Losses, Avg_Lengths, policy, path, state_values

# Helper function that plots the optimal policy pi(s) on the maze
def plot_optimal_policy(states, state_index, policy):
    # plot the value function values on the heat map
    plt.subplots(figsize=(13,7.5))
    heatmap = sns.heatmap(State_Matrix, fmt=".2f", linewidths=0.25, linecolor='black',
                        cbar= False, cmap= 'rocket_r')
    heatmap.set_facecolor('black') # Color for the NA cells in the state matrix
    coloring_blocks(heatmap, red_states, yellow_states, start_state, goal_state)

    # go through each row and column
    for (r, c) in states:
        if (r, c) == goal_state:
            continue  # no arrow at goal

        action = policy[(r, c)]

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

# This helper function plots the values V(s) at each state on the maze 
def plot_value_function(state_values):
    # Create a fresh matrix for plotting the values
    # plot the value function values on the heat map
    plt.subplots(figsize=(13,7.5))
    Value_Matrix = np.full(State_Matrix.shape, np.nan)

    for (i, j), val in state_values.items():
       Value_Matrix[i, j] = val


    # Plot the new heatmap of the new value function values with the original state and coloring blocks
    heatmap = sns.heatmap(State_Matrix, fmt=".2f", annot= Value_Matrix, linewidths=0.25, linecolor='black', cbar= False, cmap= 'rocket_r', annot_kws={"size": 8})
    heatmap.set_facecolor('black') # Color for the NA cells in the state matrix
    coloring_blocks(heatmap, red_states, yellow_states, start_state, goal_state)
    plt.title("Optimal Value Function")
    plt.show()

# This function plots the optimal path from start state to goal state
def plot_optimal_path(path):
    # Finally, create a fresh matrix for plotting the optimal path
    plt.subplots(figsize=(13,7.5))
    heatmap = sns.heatmap(State_Matrix, fmt=".2f", linewidths=0.25, linecolor='black', cbar= False, cmap= 'rocket_r')
    heatmap.set_facecolor('black') # Color for the NA cells in the state matrix
    coloring_blocks(heatmap, red_states, yellow_states, start_state, goal_state)

    for k in range(len(path)-1):
        r, c = path[k]
        r_next, c_next = path[k+1]

        dr = r_next - r
        dc = c_next - c

        plt.arrow(c + 0.5, r + 0.5, dc * 0.8, dr * 0.8, width=0.04, color='black')

    plt.title("Optimal Path")
    plt.show()

######################################################################################################################################
# **** MAIN FUNCTION ****
def main():
    plt.subplots(figsize=(10,7.5))
    heatmap = sns.heatmap(State_Matrix, fmt=".2f", linewidths=0.25, linecolor='black',
                      cbar= False, cmap= 'rocket_r', vmin=0, vmax=255)
    heatmap.set_facecolor('black') # Color for the NaN cells in the state matrix
    plt.title('Maze Problem')

    # color blocks for visualization purposes
    coloring_blocks(heatmap, red_states, yellow_states, start_state, goal_state)
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
    

    Q_network = DQN() # Q-network: first fully connected feed-forward neural network
    target_network = DQN() # Target network: second fully connected feed-forward neural network

    # Adam is used as the optimization technique
    optimizer = optim.Adam(Q_network.parameters(), lr=alpha)
    # Create the class mreplay memory D 
    memory = ReplayMemory(replay_memory_size)

    Avg_Rewards, Avg_Losses, Avg_Lengths, optimal_policy, optimal_path, state_values = deep_Q_network(Q_network, target_network, states, state_index, optimizer, memory)
    plot_optimal_policy(states, state_index, optimal_policy)
    plot_value_function(state_values) # plot optimal V(s) on the maze
    plot_optimal_path(optimal_path)


    plt.figure(figsize=(10,6))
    plt.plot(Avg_Rewards)
    plt.title("DQN Average Reward")
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(Avg_Losses)
    plt.title("DQN Average Loss")
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(Avg_Lengths)
    plt.title("DQN Average Episode Length")
    plt.show()

if __name__ == "__main__":
    main()