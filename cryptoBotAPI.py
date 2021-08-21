#Imports
from flask import Flask, request, send_file, Response
from binance.client import Client
from flask_cors.decorator import cross_origin
from tensorflow.python.keras.callbacks import EarlyStopping
import asyncio
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import threading
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
from binance.enums import *
import time as t
import json
import traceback

APIKEY = os.getenv('APIKEY')
APISECRET = os.getenv('APISECRET')

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], False)

#Log into binance api
client = Client(APIKEY, APISECRET)

#Create flask app
app = Flask(__name__)

#Stock Object
class Stock:
    def __init__(self, symbol):
        self.symbol = symbol
        self.prices = []
        self.predictedPrices = []

# @app.route('/test', methods=['POST'])
# @cross_origin()
# def test(): 
#     print(request.data)
#     return send_file('../plots/BTCUSDT.png')

# Price and Predictions
@app.route('/image1', methods=['POST'])
@cross_origin()
def image1(): 
    try:
        stocks = []
        with open("./crypto.txt", "rb") as filehandler:
            stocks = pickle.load(filehandler)
        
        coin = json.loads(request.data)
        coin = coin['coin']
        price = None
        for stock in stocks:
            if stock.symbol == coin.upper() or stock.symbol == coin.upper() + "USDT":
                price = stock.prices[-1]  
                predictedPrices = np.array(stock.predictedPrices).reshape(-1)

                #Last 200 points
                prices = []
                for i in range(len(stock.prices) - 200, len(stock.prices)):
                    prices.append(stock.prices[i])
                
                # #First 200 points
                predicted = []
                if len(predictedPrices) >= 200:
                    for i in range(0, len(predictedPrices) - predictAhead):
                        predicted.append(predictedPrices[i])
                
                plt.style.use('dark_background')   
                plt.plot(prices, color='white', label=f"Actual {stock.symbol} Price")
                plt.plot(predicted, color='green', label=f"Predicted {stock.symbol} Price")
                break
        
        if price == None:
            return Response(status=404)
        
        plt.title(str(coin).upper() + ' Price and Previous Predictions')
        plt.savefig(fname='plot', transparent=True)
        plt.clf()
        return send_file('../plot.png')
    
    except:
        traceback.print_exc()
        return Response(status=404)

# Prediction
@app.route('/image2', methods=['POST'])
@cross_origin()
def image2(): 
    
    try:
        stocks = []
        with open("./crypto.txt", "rb") as filehandler:
            stocks = pickle.load(filehandler)

        coin = json.loads(request.data)
        coin = coin['coin']
        prediction = None
        for stock in stocks:
            
            if stock.symbol == coin.upper() or stock.symbol == coin.upper() + "USDT":
                
                prediction = stock.predictedPrices[-1]
                predictedPrices = np.array(stock.predictedPrices).reshape(-1)

                #Last 60 points
                if len(predictedPrices) >= predictAhead:
                    predicted = []
                    for i in range(len(predictedPrices) - predictAhead, len(predictedPrices)):
                        predicted.append(predictedPrices[i])
                
                else:
                    predicted = predictedPrices

                plt.style.use('dark_background')   
                plt.plot(predicted, color='green', label=f"Predicted {stock.symbol} Price")
                break
        
        if prediction == None:
            return Response(status=404)
        
        plt.title(str(coin).upper() + ' Future Prediction')
        plt.savefig(fname='plot', transparent=True)
        plt.clf()
        return send_file('../plot.png')
    
    except:
        traceback.print_exc()
        return Response(status=404)

#Create stocks object
stocks = []
tickers = client.get_all_tickers()

#Neural Network Settings, predictionRequired must be lower than dataPoints(Restart entire setup procedure if you change anything here, also do not change the predictedPoints)
predictionRequired = 100
predictAhead = 60
predictedPoints = predictAhead + 200

#Number of data points and refresh rate in seconds, dataPoints should stay 500
dataPoints = 500
refreshRate = 300

#Number of coins you want to track
numCoins = 100

#Load model
try:
    model = load_model('model.h5')

except:
    print("No Model!")

#Collect data function
async def collectData():
    await asyncio.sleep(refreshRate)
    while True:
        try:                        
            start = t.time()
            # tickers = client.get_all_tickers()
            #Fill information till there are enough data points
            for stock in stocks:
                #stock.alreadyHave = False
                #stock.marketClosed = False
                ticker = client.get_symbol_ticker(symbol=stock.symbol)
                stock.prices.append(float(ticker['price']))
                while len(stock.prices) > dataPoints:
                    stock.prices.pop(0)
            
            #Write information into file
            #os.remove("crypto.txt")
            with open("crypto.txt", "wb") as filehandler:
                pickle.dump(stocks, filehandler, pickle.HIGHEST_PROTOCOL)

            end = t.time()                              
            newRefresh = round(refreshRate - (end - start))
            
            if newRefresh > 0:
                await asyncio.sleep(newRefresh)
            
        except Exception as e:
            print("Collect Data: " + str(e))

async def predictPrice():
    try: 
        while True: 
            start = t.time()           
            for stock in stocks:
                K.clear_session()
                # tf.compat.v1.reset_default_graph()
                try:
                    
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    prices = np.array(stock.prices).reshape(-1, 1)
                    scaler = scaler.fit(prices)
                    total_dataset = stock.prices

                    model_inputs = np.array(total_dataset[len(total_dataset) - predictionRequired:]).reshape(-1, 1)
                    model_inputs = scaler.transform(model_inputs)

                    #Predict Next period
                    real_data = [model_inputs[len(model_inputs) - predictionRequired:len(model_inputs), 0]]
                    real_data = np.array(real_data)
                    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

                    prediction = model.predict(real_data)
                    prediction = scaler.inverse_transform(prediction)

                except:
                    prediction = 0  
                
                stock.predictedPrices.append(prediction)
                #print(stock.symbol + ": " + str(prediction))
                while len(stock.predictedPrices) > predictedPoints:
                    stock.predictedPrices.pop(0) 
              
            end = t.time()                                                
            newRefresh = round(refreshRate - (end - start))
            
            if newRefresh > 0:
                await asyncio.sleep(newRefresh)
            
            # else:
            #     generalChannel = bot.get_channel(805608327538278423)
            #     test = generalChannel.send("Time Limit Reached for Models")
            #     fut = asyncio.run_coroutine_threadsafe(test, bot.loop)
            #     fut.result()

    except Exception as e:
        print("Predict Price: " + str(e))

async def train():
    while True:
        for stock in stocks:
            K.clear_session()
            if len(stock.prices) == 500:
                
                #Prices used to train and predict next points
                prices = []
                for i in range(400):
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
                    model = load_model('model.h5')

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
                model.save('model.h5')
                
                #Get prices to predict data
                prices = []
                for i in range(0, 400 - predictAhead):
                    prices.append(stock.prices[i])

                predictedPrices = []
                #Get predicted prices for points 400 - 500
                for i in range(400 - predictAhead + 1, 500 - predictAhead):
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
                    predictedPrices.append(prediction)
                    prices.append(stock.prices[i])

                #Get actual prices for stocks used for result
                actualPrices = []
                for i in range(400, 500):
                    actualPrices.append(float(stock.prices[i]))
                
                actualPrices = np.reshape(actualPrices, -1)
                predictedPrices = np.reshape(predictedPrices, -1)
                
                plt.style.use('dark_background')   
                plt.plot(actualPrices, color='white')
                plt.plot(predictedPrices, color='green')
                plt.title(str(stock.symbol))
                plt.savefig("./plots/"+ str(stock.symbol) + ".png", transparent=True)
                plt.clf()

if __name__ == '__main__':

    try:
        with open("./crypto.txt", "rb") as filehandler:
            stocks = pickle.load(filehandler)
            for stock in stocks:
                while len(stock.prices) > dataPoints:
                    stock.prices.pop(0)

    except:          
        num = 0
        for ticker in tickers:
            if ticker['symbol'].find('UP') == -1 and ticker['symbol'].find('DOWN') == -1 and ticker['symbol'].endswith('USDT') == True:
                stocks.append(Stock(ticker['symbol']))
                num = num + 1
            
            if num == numCoins:
                break

        for stock in stocks:
            if len(stock.prices) < dataPoints:
                candles = client.get_klines(symbol=stock.symbol, interval=Client.KLINE_INTERVAL_5MINUTE)
                
                for candle in candles:
                    stock.prices.append(float(candle[3]))

                while len(stock.prices) > dataPoints:
                    stock.prices.pop(0)
                    
        with open("./crypto.txt", "wb") as filehandler:
            pickle.dump(stocks, filehandler, pickle.HIGHEST_PROTOCOL)

    t1 = threading.Thread(target=asyncio.run, args=(collectData(),))
    t1.setDaemon(True)
    t1.start()
    t2 = threading.Thread(target=asyncio.run, args=(predictPrice(),))
    t2.setDaemon(True)
    t2.start()
    t3 = threading.Thread(target=asyncio.run, args=(train(),))
    t3.setDaemon(True)
    t3.start()
    app.run()
