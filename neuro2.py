from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import pandas as pd
from keras.models import load_model

data = pd.read_csv('data.csv')

X = data.drop(['MBERRLIM', 'TOLLIN', 'TOLVARNEWT', 'CHECKP', 'MBERRCTRL'], axis=1)
y = data[['MBERRLIM', 'TOLLIN', 'TOLVARNEWT', 'CHECKP', 'MBERRCTRL']]

# создание модели
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

# избежать переобучения
early_stop = EarlyStopping(monitor='val_loss', patience=20, mode='min')

history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop])

# для оценки модели
loss = model.evaluate(X, y)
print('Loss:', loss)
# %%
model.save("m_1.h5")
# %%

model = load_model('m_1.h5')

new_data = pd.read_csv('data.csv')
new_data_test = new_data.loc[:, ['MBERRLIM', 'TOLLIN', 'TOLVARNEWT', 'CHECKP', 'MBERRCTRL']]
y_orig = new_data_test.iloc[:200]
new_data_1 = new_data_test.iloc[:800]


def new_predict(model, X_new):
    y_pred = model.predict(X_new)
    y_pred_mean = y_pred.mean(axis=0)
    for i in range(5):
        for j in range(len(y_pred)):
            if y_pred[j][i] >= y_pred_mean[i]:
                y_pred[j][i] = 1
            else:
                y_pred[j][i] = 0
    return y_pred

X_new = new_data.drop(['MBERRLIM', 'TOLLIN', 'TOLVARNEWT', 'CHECKP', 'MBERRCTRL'], axis=1)
y_pred = new_predict(model, X_new)
#%%
y_pred = pd.DataFrame(y_pred)
## наложение выхода нейронки на данные(оставит нужное и занулит остальное -ключевые параметры-)
result = y_pred.applymap(lambda x: data.values[y_pred.index.get_loc(x), y_pred.columns.get_loc(x)] if x == 1 else 0)
