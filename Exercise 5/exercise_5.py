# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy as np
import time

# Environment
env = gym.make("Taxi-v3")
state = env.reset()

# Training 
def train():
    # Training parameters for Q learning
    alpha = 0.9 # Learning rate
    gamma = 0.9 # Future reward discount factor
    num_of_episodes = 1000
    num_of_steps = 500 # per each episode
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q_table = np.zeros([n_states,n_actions])
    tot_reward = 0

    for episode in range(num_of_episodes):
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        for step in range(num_of_steps):
            action = np.argmax(Q_table[state])
            state2, reward, done, info = env.step(action)
            Q_table[state,action] += alpha * ((reward + (gamma*(np.max(Q_table[state2])))) - Q_table[state,action])
            tot_reward += reward
            state = state2
            if done == True:
                    break
    return Q_table

# Testing
def test(Q_table):
    state = env.reset()
    done = None
    total_reward = 0
    total_action = 0
    while done!= True:
        action = np.argmax(Q_table[state])
        state, reward, done, info = env.step(action)
        total_reward += reward
        total_action +=1
        env.render()
        time.sleep(1)
    return total_action, total_reward

if __name__ == '__main__':
    Q_table_trained = train()

    total_action = []
    total_reward = []
    for i in range(10):
        action,reward = test(Q_table_trained)
        total_action.append(action)
        total_reward.append(reward)
        
    print('Average total reward over 10 episodes:',np.mean(total_reward))
    print('Average total actions over 10 episodes:',np.mean(total_action))