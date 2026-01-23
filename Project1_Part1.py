#import necessary libraries 
import math
import random
import numpy as np

def e_greedy_algorithm(epsilon, a):
    # initialize estimate of each lever as 0
    num_steps = 1000
    Q_a1 = 0
    Q_a2 = 0
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
        else:
            alpha = 1/(k+1)

        # with probability 1-epsilon (EXPLOITATION)
        if random.random() > e: 
            # choose the higher current estimate
            if max(Q_a1, Q_a2) == Q_a1:
                # draw a reward value based on the true Q(a1) distribution
                reward_1 = np.random.normal(8, math.sqrt(20))
                total_reward = total_reward + reward_1
                Accumulated_reward[k] = (1/(k+1))*total_reward
                # update current estimate of Q(a1)
                Q_a1 = Q_a1 + alpha*(reward_1 - Q_a1)

            # if Q(a2) is currently the higher estimate 
            else:
                # draw a reward value based on the true Q(a2) distribution
                # choose randomly between two Gaussians mixture 
                if np.random.choice([True, False]):
                    # draw a reward from the first Gaussian mixture 
                    reward_2 = np.random.normal(8, math.sqrt(15))
                    total_reward = total_reward + reward_2
                    Accumulated_reward[k] = (1/(k+1))*total_reward
                    Q_a2 = Q_a2 + alpha*(reward_2 - Q_a2)
                else:
                    # draw a reward from the second  Gaussian mixture 
                    reward_2  = np.random.normal(14, math.sqrt(10))
                    total_reward = total_reward + reward_2
                    Accumulated_reward[k] = (1/(k+1))*total_reward
                    Q_a2 = Q_a2 + alpha*(reward_2 - Q_a2)

        # else with probability epsilon (EXPLORATION)
        else: 
            # select randomly between lever 1 and lever 2
            if  random.random() > 0.5:
                reward_1 = np.random.normal(loc=8, scale=math.sqrt(20))
                total_reward = total_reward + reward_1
                Accumulated_reward[k] = (1/(k+1))*total_reward
                Q_a1 = Q_a1 + alpha*(reward_1 - Q_a1)
            else:
                if np.random.choice([True, False]):
                    # draw a reward from the first Gaussian mixture 
                    reward_2 = np.random.normal(8, math.sqrt(15))
                    total_reward = total_reward + reward_2
                    Accumulated_reward[k] = (1/(k+1))*total_reward
                    Q_a2 = Q_a2 + alpha*(reward_2 - Q_a2)
                else:
                    # draw a reward from the second  Gaussian mixture 
                    reward_2  = np.random.normal(14, math.sqrt(10))
                    total_reward = total_reward + reward_2
                    Accumulated_reward[k] = (1/(k+1))*total_reward
                    Q_a2 = Q_a2 + alpha*(reward_2 - Q_a2)
    return Accumulated_reward

def main():
    epsilon = [0, 0.1, 0.2, 0.5]

    # go through every combination of epsilon and step size (alpha)
    for i in range(4):
        for j in range(4):
        # call e-greedy function
            Accumulated_reward = e_greedy_algorithm(epsilon[i], j+1)
   

if __name__ == "__main__":
    main()








