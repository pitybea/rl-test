""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop
import theano
import theano.tensor as T
import numpy as np
import gym

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
gamma = 0.99
resume = False # resume from previous checkpoint?
render = False
D = 80 * 80 # input dimensionality: 80x80 grid

def build_model():
  model = Sequential()
  model.add(Dense(H, input_dim = D, activation = 'relu', init = 'lecun_uniform'))
  model.add(Dense(2, activation ='softmax'))
  model.load_weights('_weights.h5')
  my = RMSprop(lr = 0.0001)
  model.compile(loss = 'categorical_crossentropy', optimizer = my)
  return model

model = build_model()

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, len(r))):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs, drs, ys = [],[],[]
data_buffer, label_buffer = [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
  if render: env.render()
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  aprob = model.predict(np.array([x], np.float32))[0][1]
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  xs.append(x) # observation
  #y = 1 if action == 2 else 0 # a "fake label"

  ys.append([0.0, 1.0]) if action == 2 else ys.append([1.0, 0.0])
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1
    discounted_epr = discount_rewards(drs)

    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    data_buffer += xs
    labels = (np.array(ys).T * discounted_epr).T 

    label_buffer += labels.tolist()

    xs, drs, ys = [],[],[] # reset array memory

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      model.fit(np.array(data_buffer, np.float32), np.array(label_buffer, np.float32), batch_size = 512, nb_epoch = 1)
      data_buffer, label_buffer = [],[]
    if episode_number % (batch_size * 10) == 0:
      model.save_weights('_weights.h5', overwrite = True)
    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
