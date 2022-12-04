import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from numpy.random import seed
seed(10)
tensorflow.random.set_seed(10)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


sns.set()
np.random.seed(7)

# configuration
pd.set_option('display.max_columns', None)
cnx = sqlite3.connect('C:\\Users\Szydo\\Desktop\\magisterka materiały\\do wrzucenia\\bazarefactored.db')
measurement = pd.read_sql_query("SELECT * FROM measurement_refactored", cnx)


# timestamp edition
measurement['date'] = pd.to_datetime(measurement["date"], dayfirst=True)
measurement.set_index('date', inplace=True)

# date period resampling '60min' - to hours, 'D' - to daily etc.
#measurement = measurement.resample('60min').sum()

measurement = measurement[['active_energy']]

dataset = measurement.values
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(5, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')
#model.fit(trainX, trainY, epochs=10, batch_size=10, verbose=2)
history = model.fit(trainX, trainY, epochs=40, batch_size=8,  verbose=2)
# plot history
plt.plot(history.history['loss'], label='train')
plt.title('Funkcja straty błędu MAE względem liczby epok')
plt.xlabel('epoki')
plt.ylabel('strata')
plt.tight_layout()
plt.legend()
plt.show()
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
#trainRMSE = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
#print('Train Score: %.2f RMSE' % (trainRMSE))
test_mae = mean_absolute_error(testY[0], testPredict[:, 0])
test_rmse = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
test_rae = np.sum(np.abs(np.subtract(testY[0], testPredict[:, 0]))) / \
            np.sum(np.abs(np.subtract(testY[0], np.mean(testY[0]))))
test_rse = np.sum(np.square(np.subtract(testY[0], testPredict[:, 0]))) / \
            np.sum(np.square(np.subtract(testY[0], np.mean(testY[0]))))
test_r2 = r2_score(testY[0], testPredict[:, 0])
print('Test Score: %.3f MAE' % (test_mae))
print('Test Score: %.3f RMSE' % (test_rmse))
print('Test Score: %.3f RAE' % (test_rae))
print('Test Score: %.3f RSE' % (test_rse))
print('Test Score: %.3f R2' % (test_r2))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), label='Pomiar')
plt.plot(trainPredictPlot, label='Predykcja zbioru trenującego')
plt.plot(testPredictPlot, label='Predykcja zbioru testującego')
plt.title('Predykcja vs dane rzeczywiste')
plt.legend()
plt.show()

pd.DataFrame(trainPredict).to_csv('C:\\Users\Szydo\\Desktop\\magisterka materiały\\do wrzucenia\\prediction_train.csv')
pd.DataFrame(testPredict).to_csv('C:\\Users\Szydo\\Desktop\\magisterka materiały\\do wrzucenia\\prediction_test.csv')
