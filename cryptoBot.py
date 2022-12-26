#Imports
import config
import pickle
import numpy as np
import requests
import threading
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.models import load_model
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import keras.backend as K
import datetime
import time as t
import scipy.stats as stats
import cbpro
import pandas as pd
from PIL import Image
from time import sleep

#Log into coinbase
client = cbpro.AuthenticatedClient(config.coinbasePublic, config.coinbaseSecretKey, config.coinbasePassPhrase)

#Create stocks
class Coin:
    def __init__(self, symbol):
        self.symbol = symbol
        self.alreadyHave = False
        self.prices = []
        self.predictedPrices = []
        self.priceBoughtAt = 0
        self.quantityBought = 0

#Coins
coins = []

#Neural Network Settings, predictionRequired must be lower than dataPoints(Restart entire setup procedure if you change anything here, also do not change the predictedPoints)
predictionRequired = 100
predictAhead = 60
predictedPoints = predictAhead + 200

#Number of data points and refresh rate in seconds, dataPoints should stay 500, initial data is amount of data fed into training
dataPoints = 500
refreshRate = 300
initialData = 400

#Multiplier for trades, 1 means it will buy all it can, 0.5 means it will trade with half the money in one trade and could spend the other half on another or split it up more
budgetTolerance = 0.25

#Number of coins you want to track
numCoins = 100

#Do not change
ranOnce = False

def collectInitialData():
    coins = client.get_products()
    numCoins = 100
    for coin in coins:
        stocks.append(Stock(coin['id']))
        if numCoins == 100:
            break

    for stock in stocks:
        endTime = datetime.datetime.utcnow()
        startTime = endTime - datetime.timedelta(minutes=60 * 5)
        count = 0
        while True:
            historical = pd.DataFrame(client.get_product_historic_rates(product_id=stock.symbol, start=startTime, end=endTime))
            historical.columns= ["Date","Open","High","Low","Close","Volume"]
            historical['Date'] = pd.to_datetime(historical['Date'], unit='s')
            for i in range(len(historical)):
                if int(str(historical.iloc[i]['Date']).split(':')[1]) % 5 == 0:
                    stock.prices.insert(0, float(historical.iloc[i]['Close']))
                    count += 1
                    if count == 500:
                        break

            if count == 500:
                break
            
            endTime = startTime
            startTime = startTime = endTime - datetime.timedelta(minutes=60 * 5)

    with open('crypto.txt', 'wb') as fh:
        pickle.dump(stocks, fh, pickle.HIGHEST_PROTOCOL)


def on_ready():
    try:
        if ranOnce == False:
            collectInitialData()
        
    except Exception as e:
        print("On Ready: " + str(e))      


def collectData():
    while True:
        start = t.time()
        #Update prices
        for stock in stocks:
            price = None
            while price == None:
                try:
                    price = requests.get('https://api.pro.coinbase.com/products/' + stock.symbol + '/ticker').json()
                    price = price['price']
                except:
                    pass               
            stock.prices.append(price)
            while len(stock.prices) > dataPoints:
                stock.prices.pop(0)

        #Write information into file
        with open("crypto.txt", "wb") as filehandler:
            pickle.dump(stocks, filehandler, pickle.HIGHEST_PROTOCOL)
        
        end = t.time()

        #Get wait time
        newRefresh = round(refreshRate - (end - start))

        if newRefresh > 0:
            await sleep(newRefresh)        


def train():
    while True:
        global stocks        
        for stock in stocks:
            K.clear_session()
            if len(stock.prices) == dataPoints:
                fileName = "./models/" + str(stock.symbol) + "model.h5"
                #Prices used to train and predict next points
                prices = []
                for i in range(initialData):
                    prices.append(stock.prices[i])

                #Prepare data using first 400 points
                scaler = MinMaxScaler(feature_range=(0, 1))
                trainPrices = np.array(prices)
                scaled_data = scaler.fit_transform(trainPrices.reshape(-1, 1))

                x_train = []
                y_train = []

                for x in range(predictionRequired, len(scaled_data) - predictAhead):
                    x_train.append(scaled_data[x - predictionRequired:x, 0])
                    y_train.append(scaled_data[x + predictAhead, 0])

                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    
                try:
                    model = load_model(fileName)

                except:    
                    #Build model
                    model = Sequential()

                    #Experiment with layers, more layers longer time to train
                    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                    model.add(Dropout(0.2))
                    model.add(LSTM(units=50, return_sequences=True))
                    model.add(Dropout(0.2))
                    model.add(LSTM(units=50))
                    model.add(Dropout(0.2))
                    model.add(Dense(units=1)) #Prediction of next closing value

                    model.compile(optimizer='adam', loss='mean_squared_error')
                    #Epoch = how many times model sees data, batchsize = how many units it sees at once

                callbacks = [EarlyStopping(monitor='val_loss', patience=100)]          
                model.fit(x_train, y_train, epochs=1000, validation_split=0.2, callbacks=callbacks)
                # model.fit(x_train, y_train, epochs=1000)
                model.save(fileName)
                
                #Get prices to predict data
                prices = []
                for i in range(0, initialData - predictAhead):
                    prices.append(stock.prices[i])

                #Get predicted prices for points 400 - 500
                for i in range(initialData - predictAhead + 1, dataPoints - predictAhead):
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    predictPrices = np.array(prices).reshape(-1, 1)
                    scaler = scaler.fit(predictPrices)
                    total_dataset = predictPrices

                    model_inputs = np.array(total_dataset[len(total_dataset) - predictionRequired:]).reshape(-1, 1)
                    model_inputs = scaler.transform(model_inputs)

                    #Predict Next period
                    real_data = [model_inputs[len(model_inputs) - predictionRequired:len(model_inputs), 0]]
                    real_data = np.array(real_data)
                    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

                    prediction = model.predict(real_data)
                    prediction = scaler.inverse_transform(prediction)
                    stock.predictedPrices.append(prediction)
                    prices.append(stock.prices[i])
                
                while len(stock.predictedPrices) > dataPoints:
                    stock.predictedPrices.pop(0)

                #Get actual prices for stocks used for result
                actualPrices = []
                for i in range(initialData, dataPoints):
                    actualPrices.append(float(stock.prices[i]))
                
                actualPrices = np.reshape(actualPrices, -1)

                predictedPrices = stock.predictedPrices
                predictedPrices = np.reshape(predictedPrices, -1)
                
                plt.style.use('dark_background')   
                plt.plot(actualPrices, color='white')
                plt.plot(predictedPrices, color='green')
                plt.title(str(stock.symbol))
                plt.savefig("./plots/" + str(stock.symbol) + ".png", transparent=True)
                plt.clf()
                stock.newPredictions = True
                
                result = stats.ttest_ind(a=stock.prices, b=stock.predictedPrices, equal_var=True)
                p = result[1]
                if p > 0.05:
                    with open('./plots/' + str(stock.symbol) + '.png', 'rb') as fh:
                        picture = discord.File(fh)
                    await generalChannel.send(file=picture)

if __name__ == '__main__':
    on_ready()

