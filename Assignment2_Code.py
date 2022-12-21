import numpy as np
import contextlib

#Configuresnumpyprintoptions
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args,**kwargs)
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
            self.pi = np.full(n_states, 1./n_states)
 
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        return self.state

    def step(self, action):
        if action<0 or action>=self.n_actions:
            raise Exception("Invalid action.")

        self.n_steps += 1
        done = (self.n_steps>=self.max_steps)
        self.state, reward = self.draw(self.state, action)
        return self. state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()

class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):

        """
lake: A matrix that represents the lake. For example:
lake=[['&', '.', '.', '.'],
      ['.', '#', '.', '#'],
      ['.', '.', '.', '#'],
      ['#', '.', '.', '$']]
slip: The probability that the agent will slip
maxsteps: The maximum number of time steps in an episode
seed: A seed to control the random number generator(optional)
start(&), frozen(.), hole(#), goal($)        
        """
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.slip = slip
        n_states = self.lake.size+1
        n_actions = 4
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        self.Absorb_State = n_states-1
        #TODO:
        self.Hole_States = []
        i=0
        while i <= len(self.lake_flat):
            if self.lake_flat[i] == '#': 
                self.Hole_States.append(i)
            i+=1    
        self.Goal_State = np.where(self.lake_flat == '$')
        self.Start_State = np.where(self.lake_flat == '&')

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed = seed)

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.Absorb_State) or done
        return state, reward, done 

    def p(self, next_state, state, action):
        #TODO:
        return 
    
    def r(self, next_state, state, action):
        #TODO:
        if state == self.Goal_State(state):
            return 1
        else:
            return 0    

    def render(self, policy = None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            if self.state < self.Absorb_State:
                lake[self.state]='@'

            print(lake.reshape(self.lake.shape))
        else:
            #UTF−8 arrows look nicer, but cannot be used in LaTeX
            #https://www.w3schools.com/charsets/refutfarrows.asp
            actions = ['^', '<', '_', '>']
            print('Lake:')
            print(self.lake)
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            print('Value:')
            with _printoptions(precision = 3, suppress = True):
                print(value[:-1].reshape(self.lake.shape))

    def play(env):
        actions = ['w','a','s','d']
        state = env.reset()
        env.render()
        done = False
        while not done:
            c = input('\nMove: ')
            if c not in actions:
                raise Exception('Invalid action.')
            state, r, done = env.step(actions.index(c))
            env.render()
            print('Reward: {0}.'.format(r))
          

    def policy_evaluation(env, policy, gamma, theta, max_iterations) :
        value = np.zeros(env.n_states, dtype=np.float)
        #TODO:
        #Loop until delta < theta or iteration < max iterations.
        local_iterations = 0
        while local_iterations <= max_iterations:
            delta = 0

        #Loop through states
            for state in range(env.n_states):
                action = policy[state]
                State_Value = value[state]

            #Sum of value
                for state_ in range(env.n_states):
                    value[state] = sum([env.p(state_, state, action)] * (env.r(state_, state, action)) + gamma * value[state_])
                    #Calculating delta 
                    delta = max(delta, np.abs(State_Value - value[state]))

        #Stop policy evaluation if state values changes are smaller than theta or itereations are greater than max iterations
            if delta < theta:
                break
            local_iterations+=1

        return value

    def policy_improvement(env, value, gamma):
        policy = np.zeros(env.n_states, dtype=int)
        #TODO:
        #Each state
        for state in range(env.n_states):
            Policy_of_State = np.zeros(env.n_actions).tolist()
            
        #Each action
            for action in range(env.n_actions):
			#All possible next states from this state-action pair
                for state_ in range(env.n_states):
                    Policy_of_State[action] =  sum([env.p(state_, state, action)] * (env.r(state_, state, action)) + gamma * value[state_])  
    #This state
		#Maximum policy
            policy[state] = Policy_of_State.index(max(Policy_of_State))
        #Set new policy to this action that maximizes policy
            
        return policy

    def policy_iteration(env, gamma, theta, max_iterations, policy=None):
        if policy is None :
            policy = np.zeros(env.n_states, dtype=int)
        else:
            policy = np.array(policy, dtype=int)
        #TODO:
        value = np.zeros(env.n_states)
        local_iterations = 0
        while local_iterations <= max_iterations:
            #Get new policy by getting q-values and maximizing q-values per state to get best action per state
            New_Policy = env.policy_improvement(env, value, gamma)

            #Get state values
            value = env.policy_evaluation(env, policy, gamma, theta, max_iterations)

            #Stop if the value function estimates for successive policies has converged
            if np.array_equal(policy, New_Policy):
                break

            policy = New_Policy

            local_iterations += 1

        return policy, value


    def value_iteration(env, gamma, theta, max_iterations, value=None):
        if value is None :
            value = np.zeros(env.n_states)
        else:
            value = np.array(value, dtype=np.float)
        #TODO:
        local_iterations = 0
        while local_iterations <= max_iterations:
            delta = 0

        #Loop through each state
            for state in range(env.n_states):
            #Old state value
                State_Value = value[state]
                Policy_of_State = np.zeros(env.n_actions).tolist()
             #New state value = max of q-value
                for action in range(env.n_actions):
                    for state_ in range(env.n_states):
                        Policy_of_State[action] =  sum([env.p(state_, state, action)] * (env.r(state_, state, action)) + gamma * value[state_]) 
                    
                value[state] = max(Policy_of_State)
                #Calculating delta
                delta = max(delta, abs(value[state] - State_Value))

        #Stop if state values changes are smaller than theta or itereations are greater than max iterations
            if delta < theta:
                break
            local_iterations += 1

        #Extract policy with optimal state values
        policy = env.policy_improvement(env, value, gamma)

        return policy, value      


    