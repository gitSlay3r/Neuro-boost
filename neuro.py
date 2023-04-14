import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import random
import tensorflow as tf

# Load the data
data = np.genfromtxt('/Users/AhtemiichukMaxim/PycharmProjects/Quick_Protons/data (1).csv', delimiter=',')

# Define the input and output dimensions
input_dim = 16
output_dim = 5

# Split the data into training and validation sets
train_data = data[1:5000, :]
val_data = data[5000:, :]

# Define the neural network architecture
model = Sequential()
model.add(Dense(32, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(18, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(output_dim))

# Compile the model with the Adam optimizer and mean squared error loss
model.compile(loss='mse', optimizer=Adam(lr=0.001))

# Define the reinforcement learning parameters
epsilon = 0.1  # Exploration rate
gamma = 0.9  # Discount factor
batch_size = 5
replay_memory_size = 10000
num_epochs = 10

# Define the function to select the next action using an epsilon-greedy policy
def select_action(state):
    if np.random.rand() < epsilon:
        # Choose a random action
        return np.random.uniform(low=0.0, high=1.0, size=output_dim)
    else:
        # Use the neural network to predict the next action
        return model.predict(state.reshape(1, -1)).flatten()

# Define the function to update the neural network using the Bellman equation
def update_network(replay_memory):
    # Sample a minibatch from the replay memory
    # Filter out non-tuple elements in replay_memory
    replay_memory = [t for t in replay_memory if isinstance(t, tuple) and len(t) == 5]

    # Check that there are enough elements in replay_memory to create a minibatch
    if len(replay_memory) < batch_size:
        return
    # Create minibatch
    minibatch = random.sample(replay_memory, batch_size)
    # Compute the target Q-values using the Bellman equation
    targets = np.zeros((batch_size, output_dim))
    states = np.zeros((batch_size, input_dim))
    print(minibatch)
    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
        target = reward
        if not done:
            next_action = select_action(next_state)
            target += gamma * model.predict(next_state.reshape(1, -1)).flatten()[0]
        targets[i] = model.predict(state.reshape(1, -1))
        targets[i][np.argmax(action)] = target
        states[i] = state

    # Train the neural network on the minibatch
    model.train_on_batch(states, targets)

# Initialize the replay memory with random actions
replay_memory = []
for i in range(replay_memory_size):
    state = np.random.uniform(low=0.0, high=1.0, size=input_dim)
    action = np.random.uniform(low=0.0, high=1.0, size=output_dim)
    reward = -1.0
    next_state = np.random.uniform(low=0.0, high=1.0, size=input_dim)
    done = False
    replay_memory.append((state, action, reward, next_state, done))

# Train the neural network using Q-learning with experience replay
for epoch in range(num_epochs):
    total_reward = 0.0
    for i in range(train_data.shape[0]):
        state = train_data[i, :input_dim]
        max_error = train_data[i, input_dim]
        action = select_action(state)
        new_max_error = max_error + np.random.uniform(low=-0.1, high=0.1)
        reward = -1.0 if abs(new_max_error) > abs(max_error) else 1.0
        next_state = np.concatenate((train_data[i, :input_dim-1], [new_max_error]))
        done = False if epoch < num_epochs-1 else True
        replay_memory.append((state, action, reward, next_state, done))
        total_reward += reward
    if len(replay_memory) > replay_memory_size:
        replay_memory.pop(0)
    update_network(replay_memory)

# Evaluate the performance of the neural network on the validation set
val_total_reward = 0.0
for i in range(val_data.shape[0]):
    state = val_data[i, :input_dim]
    max_error = val_data[i, input_dim]
    action = select_action(state)
    new_max_error = max_error + np.random.uniform(low=-0.1, high=0.1)
    reward = -1.0 if abs(new_max_error) > abs(max_error) else 1.0
    next_state = np.concatenate((val_data[i, :input_dim - 1], [new_max_error]))
    done = True
    val_total_reward += reward

print("Epoch: {}/{}, Total Reward: {:.2f}, Validation Reward: {:.2f}"
      .format(epoch + 1, num_epochs, total_reward, val_total_reward))
#%%
model.save("TRASH_v3.h5")