import contextlib
from matplotlib import pyplot as plt
import numpy as np
import random
import torch
import collections


# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1
        # TODO:
        self.side_len = self.lake.shape[0]
        self.hole_states = np.where(self.lake_flat == '#')
        self.goal_state = self.lake.size - 1
        self.start_state = 0

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        # TODO:
        if state == self.absorbing_state or np.isin(state, self.hole_states) or state == self.goal_state:
            return next_state == self.absorbing_state
        else:
            val = 0
            slips = []
            slips.append(state) if state - self.side_len < 0 else slips.append(state - self.side_len)
            slips.append(state) if state % self.side_len == 0 else slips.append(state - 1)
            slips.append(state) if state + self.side_len > self.goal_state else slips.append(state + self.side_len)
            slips.append(state) if state % self.side_len == (self.side_len - 1) else slips.append(state + 1)
            for s in slips:
                if next_state == s:
                    val += self.slip / self.side_len

            if action == 0:
                state -= self.side_len
                if state < 0:
                    state += self.side_len
            elif action == 1:
                if state % self.side_len != 0:
                    state -= 1
            elif action == 2:
                state += self.side_len
                if state > self.goal_state:
                    state -= self.side_len
            elif action == 3:
                if state % self.side_len != (self.side_len - 1):
                    state += 1
            if next_state == state:
                val += 1 - self.slip

            return val

    def r(self, next_state, state, action):
        # TODO:
        if state == self.goal_state:
            return 1
        else:
            return 0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


def play(env):
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}.'.format(r))


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=float)

    # TODO:
    ''' 
    References:
    1) Lecture slides

    2) Reinforcement learning 1: Policy iteration, value iteration and the Frozen Lake (2020) Jacob Higgins. 
    Available at: https://jacobhiggins.github.io/posts/2020/06/blog-post-1/

    3) Ng, R. Dynamic Programming, Dynamic Programming - Deep Learning Wizard. 
    Available at: https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
    '''
    # Loop until delta < theta or iteration < max iterations.
    for i in range(max_iterations):
        delta = 0

        # Loop through states
        for state in range(env.n_states):
            action = policy[state]
            state_value = value[state]

            # Sum of value
            value[state] = sum(
                [env.p(next_state, state, action) * ((env.r(next_state, state, action)) + gamma * value[next_state])
                 for next_state in range(env.n_states)])
            # Calculating delta
            delta = max(delta, np.abs(value[state] - state_value))

        # If state values changes are smaller than theta, stop policy evaluation
        if delta < theta:
            break

    return value


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    # TODO:
    ''' 
    References:
    1) Lecture slides

    2) Reinforcement learning 1: Policy iteration, value iteration and the Frozen Lake (2020) Jacob Higgins. 
    Available at: https://jacobhiggins.github.io/posts/2020/06/blog-post-1/
    
    3) Ng, R. Dynamic Programming, Dynamic Programming - Deep Learning Wizard. 
    Available at: https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
    '''
    # Each state
    for state in range(env.n_states):
        policy_of_state = np.zeros(env.n_actions)

        # Each action
        for action in range(env.n_actions):
            # All possible next states from this state-action pair
            policy_of_state[action] = sum(
                [env.p(next_state, state, action) * (env.r(next_state, state, action) + gamma * value[next_state])
                 for next_state in range(env.n_states)])

        # Maximum policy
        # Set new policy that maximizes policy
        policy[state] = np.argmax(policy_of_state)

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    # TODO:
    ''' 
    References:
    1) Lecture slides

    2) Reinforcement learning 1: Policy iteration, value iteration and the Frozen Lake (2020) Jacob Higgins. 
    Available at: https://jacobhiggins.github.io/posts/2020/06/blog-post-1/
    
    3) Ng, R. Dynamic Programming, Dynamic Programming - Deep Learning Wizard. 
    Available at: https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
    '''
    value = np.zeros(env.n_states)
    for i in range(max_iterations):
        # Get state values
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)

        # To create new policy, obtain q-values, and maximise q-values for each state to produce the best possible action
        new_policy = policy_improvement(env, value, gamma)

        # If successive policy estimates for the value function converge, stop
        if np.array_equal(policy, new_policy):
            break

        policy = new_policy
    print ("Iterations: ", i)
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    # TODO:
    ''' 
    References:
    1) Lecture slides

    2) Reinforcement learning 1: Policy iteration, value iteration and the Frozen Lake (2020) Jacob Higgins. 
    Available at: https://jacobhiggins.github.io/posts/2020/06/blog-post-1/
    
    3) Ng, R. Dynamic Programming, Dynamic Programming - Deep Learning Wizard. 
    Available at: https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
    '''
    for i in range(max_iterations):
        delta = 0

        # Loop through each state
        for state in range(env.n_states):
            # Old state value
            state_value = value[state]
            policy_of_state = np.zeros(env.n_actions)
            # New state value = max of q-value
            for action in range(env.n_actions):
                policy_of_state[action] = sum(
                    [env.p(next_state, state, action) * (env.r(next_state, state, action) + gamma * value[next_state])
                     for next_state in range(env.n_states)])

            value[state] = max(policy_of_state)
            # Calculating delta
            delta = max(delta, abs(value[state] - state_value))

        # If state values changes are smaller than theta or itereations are greater than max iterations, stop
        if delta < theta:
            break

    # Select policy with optimal state values
    policy = policy_improvement(env, value, gamma)
    print ("Iterations: ", i)
    return policy, value



def epsilonGreedyAlgo(epsilon, currentActionValues, env):
    #Decide wether to choose exploration or exploitation.
    #random number < epsilon --- chose Random number --- Probablity of Exploration
    #random number > epsilon --- Choose current highest values --- probablity of Exploitation
    if (random.uniform(0,1))<epsilon:
        x=random.randint(0,env.n_actions-1)
        return x
    else:
        #get Maximum value indices from the array
        maxActionsList=np.argwhere(currentActionValues==np.max(currentActionValues))
        if maxActionsList.size>1:
            #Choose random index to make sure all maximum values are treated equally
            return maxActionsList[random.randint(0,maxActionsList.size-1)]
        else:
            return int(maxActionsList)
        #return np.argmax(currentActionValues)'''


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
 
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
 
    q = np.zeros((env.n_states, env.n_actions))

    # Array to store rewards
    reward_store_array = []
    for i in range(max_episodes):
        stateS = env.reset()
        actionValueList=q[stateS]
        actionA=epsilonGreedyAlgo(epsilon[i], actionValueList, env)
        done=False
        return_rewards = 0
        while not done:
            ns,reward,done=env.step(actionA)
           
            na=epsilonGreedyAlgo(epsilon[i],q[ns], env)
            q[stateS, actionA]=q[stateS,actionA]+(eta[i]*(reward+(gamma*(q[ns,na]))-q[stateS,actionA]))
            stateS=ns
            actionA=na
            # Storing reward    
            return_rewards = return_rewards + (gamma**env.n_steps) * reward
        reward_store_array.append(return_rewards)

    # Plotting graph
    graph_plot = np.convolve(reward_store_array, np.ones(20)/20, mode='valid')
    plt.clf()
    font_head = {'family':'Times New Roman','color':'black','size':20}
    font_axis = {'family':'Times New Roman','color':'black','size':15}
    plt.title("Sarsa", fontdict = font_head)
    plt.xlabel("Episode Number", fontdict = font_axis)
    plt.ylabel("Average Value", fontdict = font_axis)
    plt.plot(np.arange(1, len(graph_plot) + 1), graph_plot)
    plt.savefig('Graphs/Sarsa_Plot.png')
       
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
 
    return policy, value
 
 
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
 
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
 
    q = np.zeros((env.n_states, env.n_actions))

    # Array to store rewards
    reward_store_array = []
    for i in range(max_episodes):
        stateS = env.reset()
        actionList=q[stateS]
        actionA=epsilonGreedyAlgo(epsilon[i], actionList, env)
        return_rewards = 0
        done=False
        while not done:
            ns,reward,done=env.step(actionA)
            na=epsilonGreedyAlgo(epsilon[i],q[ns],env)

            maxActionsList=np.argwhere(q[ns]==np.max(q[ns]))
            if maxActionsList.size>1:
                #Choose random index to make sure all maximum values are treated equally
                maxNextAction= maxActionsList[random.randint(0,maxActionsList.size-1)]
            else:
                maxNextAction= int(maxActionsList)

            q[stateS, actionA]=q[stateS,actionA]+eta[i]*(reward+gamma*(q[ns,maxNextAction])-q[stateS,actionA])
            stateS=ns
            actionA=na

        # Storing reward    
            return_rewards = return_rewards + (gamma**env.n_steps) * reward
        reward_store_array.append(return_rewards)
 
    # Plotting graph
    graph_plot = np.convolve(reward_store_array, np.ones(20)/20, mode='valid')
    plt.clf()
    font_head = {'family':'Times New Roman','color':'black','size':20}
    font_axis = {'family':'Times New Roman','color':'black','size':15}
    plt.title("Q-Learning", fontdict = font_head)
    plt.xlabel("Episode Number", fontdict = font_axis)
    plt.ylabel("Average Value", fontdict = font_axis)
    plt.plot(np.arange(1, len(graph_plot) + 1), graph_plot)
    plt.savefig('Graphs/Q-Learning_Plot.png')

    policy = q.argmax(axis=1)
    value = q.max(axis=1)
 
    return policy, value


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(env.n_features)
    ''' 
    References:
    1) Lecture slides

    2) Artificial Intelligence 2E, SARSA with Linear Function Approximation. 
    Available at: https://artint.info/2e/html/ArtInt2e.Ch12.S9.SS1.html. 
    
    3) Ofyildirim, Reinforcement-learning: A repo for reinforcement learning algorithms, GitHub. 
    Available at: https://github.com/ofyildirim/reinforcement-learning. 
    '''
    # Array to store rewards
    reward_store_array = []
    for i in range(max_episodes):
        e = np.zeros(env.n_features)
        features = env.reset()
        q = features.dot(theta)
        # TODO:
        action = epsilonGreedyFunction(random_state, env, q, epsilon[i])
        return_rewards = 0
        done = False
        while not done:
            e = (gamma * e) + features[action]
            features_dash, reward, done = env.step(action)
            delta = reward - q[action]
            q = features_dash.dot(theta)
            action_dash = epsilonGreedyFunction(random_state, env, q, epsilon[i])
            
            # Update variables' value
            delta = delta + gamma * q[action_dash]
            theta = theta + (eta[i] * delta * e)
            action = action_dash
            features = features_dash

        # Storing reward    
            return_rewards = return_rewards + (gamma**env.env.n_steps) * reward
        reward_store_array.append(return_rewards)

    # Plotting graph
    graph_plot = np.convolve(reward_store_array, np.ones(20)/20, mode='valid')
    plt.clf()
    font_head = {'family':'Times New Roman','color':'black','size':20}
    font_axis = {'family':'Times New Roman','color':'black','size':15}
    plt.title("Linear Sarsa", fontdict = font_head)
    plt.xlabel("Episode Number", fontdict = font_axis)
    plt.ylabel("Average Value", fontdict = font_axis)
    plt.plot(np.arange(1, len(graph_plot) + 1), graph_plot)
    plt.savefig('Graphs/Linear_Sarsa_Plot.png')

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(env.n_features)
    gamma_decay = np.linspace(gamma, 0, max_episodes)
    ''' 
    References:
    1) Lecture slides

    2) Understanding Q-learning and linear function approximation, Seita's Place. 
    Available at: https://danieltakeshi.github.io/2016/10/31/going-deeper-into-reinforcement-learning-understanding-q-learning-and-linear-function-approximation/. 
    
    3) Ofyildirim, Reinforcement-learning: A repo for reinforcement learning algorithms, GitHub. 
    Available at: https://github.com/ofyildirim/reinforcement-learning. 
    '''
    # Array to store rewards
    reward_store_array = []
    for i in range(max_episodes):
        e = np.zeros(env.n_features)
        features = env.reset()
        q = features.dot(theta)
        # TODO:
        action = epsilonGreedyFunction(random_state, env, q, epsilon[i])
        return_rewards = 0
        done = False
        while not done:
            features_dash, reward, done = env.step(action)
            delta = reward - q[action]
            q = features_dash.dot(theta)
            action_dash = epsilonGreedyFunction(random_state, env, q, epsilon[i])
            q_value_max = np.max(q)
            # Get max value
            list_greedy_action = np.array(np.where(q == q_value_max)).flatten()

            if(np.isin(action_dash, list_greedy_action)):
                action_star = action_dash
            else:
                action_star = random_state.choice(list_greedy_action)
            
            e = e + features[action]
            # Update variables' value
            delta = delta + gamma * q[action_star]
            theta = theta + (eta[i] * delta * e)

            if(action_dash == action_star):
                e = (gamma_decay[i] * e)
            else:
                e = np.zeros(env.n_features)

            action = action_dash
            features = features_dash

        # Storing reward    
            return_rewards = return_rewards + (gamma**env.env.n_steps) * reward
        reward_store_array.append(return_rewards)  

    # Plotting graph
    graph_plot = np.convolve(reward_store_array, np.ones(20)/20, mode='valid')
    plt.clf()
    font_head = {'family':'Times New Roman','color':'black','size':20}
    font_axis = {'family':'Times New Roman','color':'black','size':15}
    plt.title("Linear Q-Learning", fontdict = font_head)
    plt.xlabel("Episode Number", fontdict = font_axis)
    plt.ylabel("Average Value", fontdict = font_axis)
    plt.plot(np.arange(1, len(graph_plot) + 1), graph_plot)
    plt.savefig('Graphs/Linear_Q-Learning_Plot.png')     
           
    return theta 

def epsilonGreedyFunction(random_state, env, q, epsilon):
    ''' 
    References:
    1) Lecture slides

    2) Roberts, S. The Epsilon-Greedy Algorithm (ε-Greedy), Medium. Towards Data Science.
    Available at: https://towardsdatascience.com/bandit-algorithms-34fd7890cb18.
    '''
    action_random = random_state.choice(env.n_actions)
    # Get max value
    list_greedy_action = np.array(np.where(q == np.max(q))).flatten()
    # Get random action from the list
    action_greedy = random_state.choice(list_greedy_action)
    
    if random_state.uniform(0, 1) < epsilon:
        # Number is smaller than epsilon then select a random no. for exploration
        return action_random
    else:
        # Number is greater than epsilon then select the current highest value for exploitation
        return action_greedy


class FrozenLakeImageWrapper:
    def __init__(self, env):
        self.env = env

        lake = self.env.lake
        print("Lake:")
        print(lake)
        print('')

        self.n_actions = self.env.n_actions
        self.state_shape = (4, lake.shape[0], lake.shape[1])

        lake_image = [(lake == c).astype(float) for c in ['&', '#', '$']]
        print(lake_image)

        self.state_image = {env.absorbing_state:
                                np.stack([np.zeros(lake.shape)] + lake_image)}
        #for state in range(lake.size):
        #   for
        stateImgSize=lake.size
        i=0
        ch2=ch3=ch4=np.zeros((lake.shape[0],lake.shape[1]))
        ch2[np.where(lake=='&')]=1
        ch3[np.where(lake=='#')]=1
        ch4[np.where(lake=='$')]=1
        for k in range((lake.shape[0])):
            for j in range ((lake.shape[1])):
                print(i,j)
                ch1=np.zeros((lake.shape[0],lake.shape[1]))
                ch1[k][j]=1
                
                self.state_image[i]=np.array([ch1,ch2,ch3,ch4])
                i=i+1
        print(self.state_image)
        



    # TODO:



    def encode_state(self, state):
        return self.state_image[state]

    def decode_policy(self, dqn):
        states = np.array([self.encode_state(s) for s in range(self.env.n_states)])
        q = dqn(states).detach().numpy()  # torch.no_grad omitted to avoid import

        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


class DeepQNetwork(torch.nn.Module):
    def __init__(self, env, learning_rate, kernel_size, conv_out_channels,
                 fc_out_features, seed):
        torch.nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.conv_layer = torch.nn.Conv2d(in_channels=env.state_shape[0],
                                          out_channels=conv_out_channels,
                                          kernel_size=kernel_size, stride=1)

        h = env.state_shape[1] - kernel_size + 1
        w = env.state_shape[2] - kernel_size + 1

        self.fc_layer = torch.nn.Linear(in_features=h * w * conv_out_channels,
                                        out_features=fc_out_features)
        self.output_layer = torch.nn.Linear(in_features=fc_out_features,
                                            out_features=env.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)

        # TODO:
        x=torch.nn.functional.relu(self.conv_layer(x))
        
        flat = torch.nn.Flatten()
        out=flat(x)
        
        #print(out)
        out=torch.nn.functional.relu(self.fc_layer(out))
        return self.output_layer(out)


    def train_step(self, transitions, gamma, tdqn):
        states = np.array([transition[0] for transition in transitions])
        actions = np.array([transition[1] for transition in transitions])
        rewards = np.array([transition[2] for transition in transitions])
        next_states = np.array([transition[3] for transition in transitions])
        dones = np.array([transition[4] for transition in transitions])

        q = self(states)
        q = q.gather(1, torch.Tensor(actions).view(len(transitions), 1).long())
        q = q.view(len(transitions))

        with torch.no_grad():
            next_q = tdqn(next_states).max(dim=1)[0] * (1 - dones)

        target = torch.Tensor(rewards) + gamma * next_q

        # TODO: the loss is the mean squared error between `q` and `target`
        loss = 0
        #print(type(q))
        q = q.double()
        meanSquaredLoss=torch.nn.MSELoss()
        loss=meanSquaredLoss(q,target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, buffer_size, random_state):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.random_state = random_state

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def draw(self, batch_size):
        batch=collections.deque()
        for i in range(batch_size):
            if len(self.buffer)>0:
                batch.append(self.buffer.pop())
            else:
                break
        return batch



# TODO:


def deep_q_network_learning(env, max_episodes, learning_rate, gamma, epsilon,
                            batch_size, target_update_frequency, buffer_size,
                            kernel_size, conv_out_channels, fc_out_features, seed):
    random_state = np.random.RandomState(seed)
    replay_buffer = ReplayBuffer(buffer_size, random_state)

    dqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels,
                       fc_out_features, seed=seed)
    tdqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels,
                        fc_out_features, seed=seed)

    epsilon = np.linspace(epsilon, 0, max_episodes)

    reward_store_array = []

    for i in range(max_episodes):
        state = env.reset()
        return_rewards = 0
        done = False
        while not done:
            if random_state.rand() < epsilon[i]:
                action = random_state.choice(env.n_actions)
            else:
                with torch.no_grad():
                    q = dqn(np.array([state]))[0].numpy()

                qmax = max(q)
                best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
                action = random_state.choice(best)

            next_state, reward, done = env.step(action)

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.draw(batch_size)
                dqn.train_step(transitions, gamma, tdqn)

        if (i % target_update_frequency) == 0:
            tdqn.load_state_dict(dqn.state_dict())

        # Storing reward    
            return_rewards = return_rewards + (gamma**env.env.n_steps) * reward
        reward_store_array.append(return_rewards)

    # Plotting graph
    graph_plot = np.convolve(reward_store_array, np.ones(20)/20, mode='valid')
    plt.clf()
    font_head = {'family':'Times New Roman','color':'black','size':20}
    font_axis = {'family':'Times New Roman','color':'black','size':15}
    plt.title("Deep Q-Network Learning", fontdict = font_head)
    plt.xlabel("Episode Number", fontdict = font_axis)
    plt.ylabel("Average Value", fontdict = font_axis)
    plt.plot(np.arange(1, len(graph_plot) + 1), graph_plot)
    plt.savefig('Graphs/Deep_Q-Network_Learning_Plot.png')

    return dqn



def main():
    seed = 0

    # Big lake
    # lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '#', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '#', '#', '.', '.', '.', '#', '.'],
    #         ['.', '#', '.', '.', '#', '.', '#', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '$']]

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9
    '''
    print('# Model-based algorithms')

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 4000

    print('')
    
    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta=0.5, gamma=gamma,
                          epsilon=0.5, seed=seed)
    env.render(policy, value)

    print('')
    
    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta=0.5, gamma=gamma,
                               epsilon=0.5, seed=seed)
    env.render(policy, value)

    print('')
      
    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')

    parameters = linear_sarsa(linear_env, max_episodes, eta=0.5, gamma=gamma,
                              epsilon=0.5, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')

    parameters = linear_q_learning(linear_env, max_episodes, eta=0.5, gamma=gamma,
                                   epsilon=0.5, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')
    '''
    max_episodes = 4000
    image_env = FrozenLakeImageWrapper(env)

    print('## Deep Q-network learning')

    dqn = deep_q_network_learning(image_env, max_episodes, learning_rate=0.001,
                                  gamma=gamma, epsilon=0.2, batch_size=32,
                                  target_update_frequency=4, buffer_size=256,
                                  kernel_size=3, conv_out_channels=4,
                                  fc_out_features=8, seed=4)
    policy, value = image_env.decode_policy(dqn)
    image_env.render(policy, value)
    


if __name__ == '__main__':
    main()