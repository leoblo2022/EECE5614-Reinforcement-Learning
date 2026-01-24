#import necessary libraries 
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# this helper function updates the estimate of Q_at and also tracks accumulated reward
def update_Q_and_accR(Accumulated_reward, total_reward, reward, k, Q_a, alpha):
    # track accumulated reward 
    total_reward = total_reward + reward
    Accumulated_reward[k] = (1/(k+1))*total_reward
    # update current estimate of Q(a1)
    Q_a = Q_a + alpha*(reward - Q_a)
    return total_reward, Accumulated_reward, Q_a


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
                # update Q_a estimare and track accumulated reward
                total_reward, Accumulated_reward, Q_a1 = update_Q_and_accR(Accumulated_reward, total_reward, reward_1, k, Q_a1, alpha)


            # if Q(a2) is currently the higher estimate 
            else:
                # draw a reward value based on the true Q(a2) distribution
                # choose randomly between two Gaussians mixture 
                if np.random.choice([True, False]):
                    # draw a reward from the first Gaussian mixture 
                    reward_2 = np.random.normal(8, math.sqrt(15))
                    # update Q_a estimare and track accumulated reward
                    total_reward, Accumulated_reward, Q_a2 = update_Q_and_accR(Accumulated_reward, total_reward, reward_2, k, Q_a2, alpha)

                else:
                    # draw a reward from the second  Gaussian mixture 
                    reward_2  = np.random.normal(14, math.sqrt(10))
                    # update Q_a estimare and track accumulated reward
                    total_reward, Accumulated_reward, Q_a2 = update_Q_and_accR(Accumulated_reward, total_reward, reward_2, k, Q_a2, alpha)

        # else with probability epsilon (EXPLORATION)
        else: 
            # select randomly between lever 1 and lever 2
            if  random.random() > 0.5:
                reward_1 = np.random.normal(loc=8, scale=math.sqrt(20))
                # update Q_a estimare and track accumulated reward
                total_reward, Accumulated_reward, Q_a1 = update_Q_and_accR(Accumulated_reward, total_reward, reward_1, k, Q_a1, alpha)
            else:
                if np.random.choice([True, False]):
                    # draw a reward from the first Gaussian mixture 
                    reward_2 = np.random.normal(8, math.sqrt(15))
                    # update Q_a estimare and track accumulated reward
                    total_reward, Accumulated_reward, Q_a2 = update_Q_and_accR(Accumulated_reward, total_reward, reward_2, k, Q_a2, alpha)
                else:
                    # draw a reward from the second  Gaussian mixture 
                    reward_2  = np.random.normal(14, math.sqrt(10))
                    # update Q_a estimare and track accumulated reward
                    total_reward, Accumulated_reward, Q_a2 = update_Q_and_accR(Accumulated_reward, total_reward, reward_2, k, Q_a2, alpha)
    return Accumulated_reward, Q_a1, Q_a2

def main():
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
                Accumulated_reward, Q_a1, Q_a2 = e_greedy_algorithm(epsilon[j], i+1)
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
        # produce plots here
            plt.xlabel('Time (t)')
            plt.ylabel('Average Accumulated Reward')
            plt.title('Average Accumulated Reward for Different e-greedy Values and Learning Rates')
            plt.legend(['e = 0', 'e = 0.1', 'e = 0.2', 'e = 0.5'])
            plt.plot(time_array, Average_reward)
            # Display average of action value Q(a1) and Q(a2) after 100 runs 
            print("Average of Q(a1) for e =", epsilon[j], ": ", Q_a1_mean)
            print("Average of Q(a2) for e =", epsilon[j], ": ", Q_a2_mean)

        plt.show()
        
if __name__ == "__main__":
    main()








