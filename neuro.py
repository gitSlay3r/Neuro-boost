import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import random
import tensorflow as tf

data = np.genfromtxt('data (1).csv', delimiter=',')

#  входные и выходные размерности
input_dim = 16
output_dim = 5

train_data = data[1:5000, :]
val_data = data[5000:, :]

# Определения архитектуры
model = Sequential()
model.add(Dense(32, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(18, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(output_dim))

model.compile(loss='mse', optimizer=Adam(lr=0.001))

epsilon = 0.1
gamma = 0.9
batch_size = 5
replay_memory_size = 10000
num_epochs = 10

# Определить функцию для выбора следующего действия с помощью epsilon-greedy policy
def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.uniform(low=0.0, high=1.0, size=output_dim)
    else:
        return model.predict(state.reshape(1, -1)).flatten()


def update_network(replay_memory):
    replay_memory = [t for t in replay_memory if isinstance(t, tuple) and len(t) == 5]

    if len(replay_memory) < batch_size:
        return

    minibatch = random.sample(replay_memory, batch_size)
    targets = np.zeros((batch_size, output_dim))
    states = np.zeros((batch_size, input_dim))
    print(minibatch)
    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
        target = reward
        if not done:
            # next_action = select_action(next_state)
            target += gamma * model.predict(next_state.reshape(1, -1)).flatten()[0]
        targets[i] = model.predict(state.reshape(1, -1))
        targets[i][np.argmax(action)] = target
        states[i] = state

    model.train_on_batch(states, targets)


replay_memory = []
for i in range(replay_memory_size):
    state = np.random.uniform(low=0.0, high=1.0, size=input_dim)
    action = np.random.uniform(low=0.0, high=1.0, size=output_dim)
    reward = -1.0
    next_state = np.random.uniform(low=0.0, high=1.0, size=input_dim)
    done = False
    replay_memory.append((state, action, reward, next_state, done))


for epoch in range(num_epochs):
    total_reward = 0.0
    for i in range(train_data.shape[0]):
        # state = train_data[i, :input_dim]
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
model.save("n_v.h5")
