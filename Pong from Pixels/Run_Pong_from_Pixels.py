#""" Run an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
# import cPickle as pickle
import pickle
import gym
import time

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
use_trained_model = True # resume from previous checkpoint?
# use_trained_model = False # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if use_trained_model:
  # model = pickle.load(open('save_one_week.p', 'rb'))
  model = pickle.load(open('save_two_weeks.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
    
def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

env = gym.make("Pong-v0")
# env.mode = 'human'


for i_episode in range(1):
    observation = env.reset()
    prev_x = None # used in computing the difference frame
    for t in range(10000):
        env.render()
        time.sleep(0.01)

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x
        
        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

        # action = env.action_space.sample()

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            time.sleep(600)
            break