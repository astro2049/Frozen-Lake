# Frozen Lake

> *Group M*

## 0 - Code

Based on the correctness and clarity of your code and the output of your main function, you will receive the following number of points for accomplishing each of the following tasks:

1. Implementing the frozen lake environment [10/100]

2. Implementing policy iteration [7.5/100]

3. Implementing value iteration [7.5/100]

4. Implementing Sarsa control [7.5/100]

5. Implementing Q-learning control [7.5/100]

6. Implementing Sarsa control using linear function approximation [7.5/100]

7. Implementing Q-learning control using linear function approximation [7.5/100]

8. Implementing deep Q-network learning [15/100]
   

**Important**: An incorrect implementation of the frozen lake environment will compromise the points received for correct implementations of reinforcement learning algorithms, since correctness will be mostly assessed based on the output of the main function. If you are not able to implement the frozen lake environment correctly, you may use the transition probabilities from *p.npy* (see Sec. 1).

## 1 - Report

Explain how your code for this assignment is organized and justify implementation decisions that deviate significantly from what we suggested. Must be excellently organized.

Additionally, you will receive the following number of points for answering each of the following questions:

1. How many iterations did policy iteration require before returning an optimal policy for the big frozen lake? How many iterations did value iteration require? [2.5/100]
2. For each model-free reinforcement learning algorithm (Sarsa control, Q-learning control, linear Sarsa control, linear Q-learning control, deep Q-network learning) store the return (sum of **discounted** rewards) obtained during each episode of interaction with the small frozen lake. For each of these algorithms, include a plot that shows the episode number on the x-axis and a moving average of these values on the y-axis. Use a moving average window of length 20. **Hint**: `np.convolve(returns array, np.ones(20)/20, mode=’valid’)`. [10/100]
3. Try to minimize the number of episodes required to find an optimal policy for the *small* frozen lake by tweaking the parameters of Sarsa control and Q-learning control (learning rate and exploration factor). Describe your results. Then try to find an optimal policy for the *big* frozen lake by tweaking the parameters of Sarsa control and Q-learning control. Even if you fail, describe your results. [10/100]
4. In linear action-value function approximation, how can each element of the parameter vector ***θ*** be interpreted when each possible pair of state *s* and action *a* is represented by a different feature vector *ϕ(s, a)* where all elements except one are zero? Explain why the tabular model-free reinforcement learning algorithms that you implemented are a special case of the non-tabular model-free reinforcement learning algorithms that you implemented. [2.5/100]
5. During deep Q-network training, why is it necessary to act according to an *ϵ*-greedy policy instead of a greedy policy (with respect to Q) [2.5/100]
6. How do the authors of deep Q-network learning [Mnih et al., 2015] explain the need for a target Q-network in addition to an online Q-network? [2.5/100]

## 2 - Individual Reflection

Briefly describe the role that each member of the group had in the submission.