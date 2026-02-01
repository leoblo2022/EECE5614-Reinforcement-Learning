#import necessary libraries 
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# this helper function updates the estimate of Q_at and also tracks accumulated reward
def update_Q_and_accR(Accumulated_reward, total_reward, reward, k, Q_a, alpha, policy_type):
    # track accumulated reward 
    total_reward = total_reward + reward
    Accumulated_reward[k] = (1/(k+1))*total_reward
    # update current estimate of Q(a1)
    if policy_type == 0: # is e-greedy algorithm (not gradient-bandit)
        Q_a = Q_a + alpha*(reward - Q_a)
    else: 
        Q_a = 0 # is gradient-bandit or UCB policy, don't need previous Q_a
    return total_reward, Accumulated_reward, Q_a

# e-GREEDY ACTION-SELECTION FUNCTION
def e_greedy_algorithm(epsilon, a, init):
    # initialize estimate of each lever as 0
    num_steps = 1000
    Q_a1 = init[0]
    Q_a2 = init[1]
    Accumulated_reward = [0] * num_steps
    total_reward = 0

    for k in range(num_steps):
        e = epsilon # epsilon-greedy parameter 
        # select the corresponding alpha parameter
        if a==1:
            alpha = 1
        elif a==2:
            alpha = 0.9**(k+1)
        elif a==3:
            alpha = 1/(1+math.log(1+(k+1))) 
        elif a==4:
            alpha = 1/(k+1)
        else:
            alpha = 0.1

        # with probability 1-epsilon (EXPLOITATION)
        if random.random() > e: 
            # choose the higher current estimate
            if max(Q_a1, Q_a2) == Q_a1:
                # draw a reward value based on the true Q(a1) distribution
                reward_1 = np.random.normal(8, math.sqrt(20))
                # update Q_a estimare and track accumulated reward
                total_reward, Accumulated_reward, Q_a1 = update_Q_and_accR(Accumulated_reward, total_reward, reward_1, k, Q_a1, alpha, 0)


            # if Q(a2) is currently the higher estimate 
            else:
                # draw a reward value based on the true Q(a2) distribution
                # choose randomly between two Gaussians mixture 
                if np.random.choice([True, False]):
                    # draw a reward from the first Gaussian mixture 
                    reward_2 = np.random.normal(8, math.sqrt(15))
                    # update Q_a estimare and track accumulated reward
                    total_reward, Accumulated_reward, Q_a2 = update_Q_and_accR(Accumulated_reward, total_reward, reward_2, k, Q_a2, alpha, 0)

                else:
                    # draw a reward from the second  Gaussian mixture 
                    reward_2  = np.random.normal(14, math.sqrt(10))
                    # update Q_a estimare and track accumulated reward
                    total_reward, Accumulated_reward, Q_a2 = update_Q_and_accR(Accumulated_reward, total_reward, reward_2, k, Q_a2, alpha, 0)

        # else with probability epsilon (EXPLORATION)
        else: 
            # select randomly between lever 1 and lever 2
            if  random.random() > 0.5:
                reward_1 = np.random.normal(loc=8, scale=math.sqrt(20))
                # update Q_a estimare and track accumulated reward
                total_reward, Accumulated_reward, Q_a1 = update_Q_and_accR(Accumulated_reward, total_reward, reward_1, k, Q_a1, alpha, 0)
            else:
                if np.random.choice([True, False]):
                    # draw a reward from the first Gaussian mixture 
                    reward_2 = np.random.normal(8, math.sqrt(15))
                    # update Q_a estimare and track accumulated reward
                    total_reward, Accumulated_reward, Q_a2 = update_Q_and_accR(Accumulated_reward, total_reward, reward_2, k, Q_a2, alpha, 0)
                else:
                    # draw a reward from the second  Gaussian mixture 
                    reward_2  = np.random.normal(14, math.sqrt(10))
                    # update Q_a estimare and track accumulated reward
                    total_reward, Accumulated_reward, Q_a2 = update_Q_and_accR(Accumulated_reward, total_reward, reward_2, k, Q_a2, alpha, 0)
    return Accumulated_reward, Q_a1, Q_a2

# OPTIMISIC INITIALIZATION FUNCTION
def optimistic_initialization(e, alpha, optim):
    total_reward = [0] * 1000
    Q_a1_list = [0]*100
    Q_a2_list = [0]*100
    num_runs = 100
    if optim == 0:
        init= [0, 0]
    elif optim == 1:
        init = [8, 11]
    else:
        init = [20, 20]
    for runs in range(num_runs):
        Accumulated_reward, Q_a1, Q_a2 = e_greedy_algorithm(e, alpha, init) # initialize to [0,0]
        Q_a1_list[runs] = Q_a1
        Q_a2_list[runs] = Q_a2
        for k in range(1000):
                    # sum accumulated reward over all runs 
                    total_reward[k] = total_reward[k] + Accumulated_reward[k]
    # calculate the average of 100 runs using the summed accumulared reward
    Average_reward = [element / 100 for element in total_reward]
    # calculate the average action value of Q(a1) and Q(a2) after 100 runs 
    Q_a1_mean = np.mean(Q_a1_list)
    Q_a2_mean = np.mean(Q_a2_list)
    return Average_reward, Q_a1_mean, Q_a2_mean, init

# GRADIENT-BANDIT POLICY FUNCTION
def gradient_bandit(H_a1, H_a2, alpha, k, total_reward, Accumulated_reward):
    # calculate policy probabilities
    pi_a1 = (np.exp(H_a1) / (np.exp(H_a1) + np.exp(H_a2)))
    pi_a2 = 1-pi_a1
    # select next action based on policy probabilities 
    if random.random() < pi_a1: 
        # select a_1 as action
        # draw reward from a_1 
        reward = np.random.normal(8, math.sqrt(20))
        # calculated accumulated average reward
        total_reward, Accumulated_reward, Q_a2 = update_Q_and_accR(Accumulated_reward, total_reward, reward, k, 0, alpha, 1)
        # update preferences, where a1 is selected action 
        H_a1 = H_a1 + alpha*(reward-Accumulated_reward[k])*(1-pi_a1)
        H_a2 = H_a2 - alpha*(reward-Accumulated_reward[k])*(pi_a2)
    else:
        # select a_2 as action
        # draw reward from a_2 (need to implement)
        if np.random.choice([True, False]):
            # draw a reward from the first Gaussian mixture 
            reward = np.random.normal(8, math.sqrt(15))
            # update Q_a estimare and track accumulated reward
            total_reward, Accumulated_reward, Q_a2 = update_Q_and_accR(Accumulated_reward, total_reward, reward, k, 0, alpha, 1)
        else:
            # draw a reward from the second  Gaussian mixture 
            reward  = np.random.normal(14, math.sqrt(10))
            total_reward, Accumulated_reward, Q_a2 = update_Q_and_accR(Accumulated_reward, total_reward, reward, k, 0, alpha, 1)
        # update preferences, where a2 is selected action 
        H_a2 = H_a2 + alpha*(reward-Accumulated_reward[k])*(1-pi_a2)
        H_a1 = H_a1 - alpha*(reward-Accumulated_reward[k])*(pi_a1)
    return H_a1, H_a2, total_reward, Accumulated_reward

# Upper Confidence Bound (UCB) Policy Function
def upper_confidence_bound(Q_a1, Q_a2, c, k, N_a1, N_a2, total_reward, Accumulated_reward):
    # select action
    action = np.argmax([Q_a1+(c*math.sqrt((math.log(k+1))/(N_a1))), Q_a2+(c*math.sqrt((math.log(k+1))/(N_a2)))])
    # if action 1 is selected
    if action == 0:
        reward = np.random.normal(8, math.sqrt(20))
        Q_a1 = Q_a1 + (1/N_a1)*(reward - Q_a1)
        N_a1 = N_a1 + 1
        total_reward, Accumulated_reward, num = update_Q_and_accR(Accumulated_reward, total_reward, reward, k, 0, 0, 1)
    # if action 2 is selected
    else:
        if np.random.choice([True, False]):
            # draw a reward from the first Gaussian mixture 
            reward = np.random.normal(8, math.sqrt(15))
        else:
            # draw a reward from the second  Gaussian mixture 
            reward  = np.random.normal(14, math.sqrt(10))
        # update preferences, where a2 is selected action 
        Q_a2 = Q_a2 + (1/N_a2)*(reward - Q_a2)
        N_a2 = N_a2 + 1
        total_reward, Accumulated_reward, num = update_Q_and_accR(Accumulated_reward, total_reward, reward, k, 0, 0, 1)
    return Q_a1, Q_a2, N_a1, N_a2, total_reward, Accumulated_reward

def calculate_average_accum_reward(policy_type):
    # Part 4: Upper Confidence Bound Policy (UCB)
    alpha = 0.1
    c=5
    num_runs = 100
    num_steps = 1000
    global_reward = [0]*1000
    time_array = list(range(1, 1001))
    for run in range(num_runs):
        N_a1 = 1
        N_a2 = 1
        Q_a1 = 0
        Q_a2 = 0
        H_a1 = 0
        H_a2 = 0
        total_reward = 0
        Accumulated_reward = [0] * 1000
        if policy_type == 1:
            Accumulated_reward_greedy, Q_a1, Q_a2 = e_greedy_algorithm(0.1, 0.1, init=[0,0])
        for k in range(num_steps):
            if policy_type == 1:
                global_reward[k] = global_reward[k] + Accumulated_reward_greedy[k]
            elif policy_type == 2:
                H_a1, H_a2, total_reward, Accumulated_reward = gradient_bandit(H_a1, H_a2, alpha, k, total_reward, Accumulated_reward)
                global_reward[k] = global_reward[k] + Accumulated_reward[k]
            elif policy_type == 3:
                Q_a1, Q_a2, N_a1, N_a2, total_reward, Accumulated_reward = upper_confidence_bound(Q_a1, Q_a2, c, k, N_a1, N_a2, total_reward, Accumulated_reward)
                global_reward[k] = global_reward[k] + Accumulated_reward[k]
    # calculate the average of 100 runs using the summed accumulated reward
    Average_reward = [element / 100 for element in global_reward]
    # calculate the average of 100 runs using the summed accumulated reward      
    plt.xlabel('Time (t)')
    plt.ylabel('Average Accumulated Reward')
    plt.title('Average Accumulated Reward for policies')
    if policy_type == 1:
        plt.plot(time_array, Average_reward, label='e-Greedy')
    elif policy_type == 2:
        plt.plot(time_array, Average_reward, label='Gradient_bandit')
    elif policy_type == 3:      
        plt.plot(time_array, Average_reward, label='Upper Confidence Bound (UCB)')

###########################################################################################################################
def main():
    
    # PART 1: e-greedy algorithm
    epsilon = [0, 0.1, 0.2, 0.5]
    num_runs = 100
    # create x-axis
    time_array = my_array = list(range(1, 1001))
    # go through every combination of epsilon and step size (alpha)
    for i in range(4):
        if i+1==1:
            learn_rate = "1"
        elif i+1==2:
            learn_rate  = "0.9^(k)"
        elif i+1==3:
            learn_rate  = "1/1+ln(1+k)" 
        else:
            learn_rate  = "1/k"    
        print(" ")
        print("For Learning Rate: ", learn_rate)
        for j in range(4):
            # initialized summed accumulared reward and action values
            total_reward = [0] * 1000
            Q_a1_list = [0]*100
            Q_a2_list = [0]*100
            # repeat for 100 independent runs 
            for runs in range(num_runs):
                # call e-greedy function
                Accumulated_reward, Q_a1, Q_a2 = e_greedy_algorithm(epsilon[j], i+1, init=[0,0])
                Q_a1_list[runs] = Q_a1
                Q_a2_list[runs] = Q_a2
                for k in range(1000):
                    # sum accumulated reward over all runs 
                    total_reward[k] = total_reward[k] + Accumulated_reward[k]
            # calculate the average of 100 runs using the summed accumulated reward
            Average_reward = [element / 100 for element in total_reward]
            # calculate the average action value of Q(a1) and Q(a2) after 100 runs 
            Q_a1_mean = np.mean(Q_a1_list)
            Q_a2_mean = np.mean(Q_a2_list)
            # produce plots here
            plt.xlabel('Time (t)')
            plt.ylabel('Average Accumulated Reward')
            plt.title('Average Accumulated Reward for Different e-greedy Values and Learning Rates')
            plt.plot(time_array, Average_reward, label=f'e = {epsilon[j]}')
            # Display average of action value Q(a1) and Q(a2) after 100 runs 
            print("Average of Q(a1) for e =", epsilon[j], ": ", Q_a1_mean)
            print("Average of Q(a2) for e =", epsilon[j], ": ", Q_a2_mean)

        plt.legend()
        plt.show()
    #############################################################################################################
    
    # PART 2: Optimistic Initialization 
    alpha = 0.1
    e = 0.1
    for optimistic in range(3):
        Average_reward, Q_a1_mean, Q_a2_mean, init = optimistic_initialization(e, alpha, optimistic)
        plt.xlabel('Time (t)')
        plt.ylabel('Average Accumulated Reward')
        plt.title('Average Accumulated Reward for Different optimistic initializations')
        plt.plot(time_array, Average_reward, label=f'Q = {init}')
        print(" ")
        print("Average of Q(a1) for optimistic initialization: [", init[0], " " , init[1], "]:", Q_a1_mean)
        print("Average of Q(a2) for optimistic initialization: [", init[0], " " , init[1], "]:", Q_a2_mean)
    
    plt.legend()
    plt.show()
        
    #############################################################################################################

    # Part 3: Gradient-Bandit Policy
    print(" ")
    print("Plotting Gradient-Bandit Policy compared to e-greedy")
    alpha = 0.1
    num_runs = 100
    num_steps = 1000
    global_reward = [0]*1000
    total_reward_greedy = [0]*1000
    for runs in range(num_runs):
        H_a1 = 0
        H_a2 = 0
        total_reward = 0
        Accumulated_reward = [0] * 1000
        Accumulated_reward_greedy, Q_a1, Q_a2 = e_greedy_algorithm(0.1, 0.1, init=[0,0])
        for k in range(num_steps):
            H_a1, H_a2, total_reward, Accumulated_reward = gradient_bandit(H_a1, H_a2, alpha, k, total_reward, Accumulated_reward)
            global_reward[k] = global_reward[k] + Accumulated_reward[k]
            total_reward_greedy[k] = total_reward_greedy[k] + Accumulated_reward_greedy[k]
    # calculate the average of 100 runs using the summed accumulated reward
    Average_reward = [element / 100 for element in global_reward]
    # calculate the average of 100 runs using the summed accumulated reward
    Average_reward_greedy = [element / 100 for element in total_reward_greedy]
    # produce plots here
    plt.xlabel('Time (t)')
    plt.ylabel('Average Accumulated Reward')
    plt.title('Average Accumulated Reward for Gradient-Bandit Policy vs. e-Greedy Policy')
    plt.plot(time_array, Average_reward, label='Gradient-Bandit Policy')
    plt.plot(time_array, Average_reward_greedy, label='e-Greedy Policy')
    plt.legend()
    plt.show()


    ########################################################################################################################

    # Part 4: Upper Confidence Bound Policy (UCB)
    print(" ")
    print("Plotting UCB policies with different exploration rates c")
    num_runs = 100
    num_steps = 1000
    c_list = [2, 5, 100]
    for i in range(3):
        global_reward = [0]*1000
        c = c_list[i] # select value for exploration rate c   
        for runs in range(num_runs):
            N_a1 = 1
            N_a2 = 1
            Q_a1 = 0
            Q_a2 = 0
            total_reward = 0
            Accumulated_reward = [0] * 1000
            for k in range(num_steps):
                Q_a1, Q_a2, N_a1, N_a2, total_reward, Accumulated_reward = upper_confidence_bound(Q_a1, Q_a2, c, k, N_a1, N_a2, total_reward, Accumulated_reward)
                global_reward[k] = global_reward[k] + Accumulated_reward[k]
        # calculate the average of 100 runs using the summed accumulated reward
        Average_reward = [element / 100 for element in global_reward]
        # calculate the average of 100 runs using the summed accumulated reward      
        plt.xlabel('Time (t)')
        plt.ylabel('Average Accumulated Reward')
        plt.title('Average Accumulated Reward for Different UCB policies')
        plt.plot(time_array, Average_reward, label=f'c={c_list[i]}')
    plt.legend()
    plt.show()

    print(" ")
    print("Plotting e-greedy")
    calculate_average_accum_reward(1)
    print(" ")
    print("Plotting gradient-bandit")
    calculate_average_accum_reward(2)
    print(" ")
    print("Plotting UCB policy")
    calculate_average_accum_reward(3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()








