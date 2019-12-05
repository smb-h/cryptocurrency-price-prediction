import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from .data_splitter import train_test_split
from .plot import line_plot


endpoint = 'https://min-api.cryptocompare.com/data/histoday'
api_key = "fbd094f08b216fbd27e436c1134c34aeac9cf6b823f425ce4e67c43d322b7081"
res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=500&api_key=' + api_key)



# print(res.content)
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 'close'

# print(hist.head(5))

# split data
train, test = train_test_split(hist, test_size=0.2)


pd.plotting.register_matplotlib_converters()

line_plot(train[target_col], test[target_col], 'training', 'test', title='')




# buid neural network model
def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

# initial data in neurons in LSTM layer
np.random.seed(42)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 100
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'


# train model
train, test, X_train, X_test, y_train, y_test = prepare_data(
    hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)


# Mean Absolute Error
targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, y_test)

# plot and predict prices
preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
line_plot(targets, preds, 'actual', 'prediction', lw=3)






