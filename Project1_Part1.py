#import necessary libraries 
import math
import random
import numpy as np

num_steps = 1000
epsilon = [0, 0.1, 0.2, 0.5]
#alpha = [1, 0.9**k, 1/(1+math.log(1+k)), 1/k]

Q_a1 = 0
Q_a2 = 0
alpha = 0.5
for k in range(num_steps):
    e = epsilon[1]
    if random.random() > e:
        if max(Q_a1, Q_a2) == Q_a1:
            reward_1 = np.random.normal(8, math.sqrt(20))
            Q_a1 = Q_a1 + alpha*(reward_1 - Q_a1)
        else:
            if np.random.choice([True, False]):
                # Draw from the first component
                reward_2 = np.random.normal(8, math.sqrt(15))
                Q_a2 = Q_a2 + alpha*(reward_2 - Q_a2)
            else:
                # Draw from the second component
                reward_2  = np.random.normal(14, math.sqrt(10))
                Q_a2 = Q_a2 + alpha*(reward_2 - Q_a2)
    else: 
        if  random.random() > 0.5:
            reward_1 = np.random.normal(loc=8, scale=math.sqrt(20))
            Q_a1 = Q_a1 + alpha*(reward_1 - Q_a1)
        else:
            if np.random.choice([True, False]):
                # Draw from the first component
                reward_2 = np.random.normal(8, math.sqrt(15))
                Q_a2 = Q_a2 + alpha*(reward_2 - Q_a2)
            else:
                # Draw from the second component
                reward_2  = np.random.normal(14, math.sqrt(10))
                Q_a2 = Q_a2 + alpha*(reward_2 - Q_a2)


print("Q_a1", str(Q_a1))
print("Q_a2", str(Q_a2))




