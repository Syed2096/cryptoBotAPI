#Imports
from flask import Flask, request, send_file, Response
from binance.client import Client
from flask_cors.decorator import cross_origin
from numpy.core.fromnumeric import trace
from tensorflow.python.keras.callbacks import EarlyStopping
import asyncio
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
from flask_sqlalchemy import SQLAlchemy

APIKEY = os.getenv('APIKEY')
APISECRET = os.getenv('APISECRET')
URI = os.getenv('CLEARDB_DATABASE_URL')

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], False)

#Log into binance api
client = Client(APIKEY, APISECRET)

#Create flask app
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = URI
app.config['SQLALCHEMY_POOL_RECYCLE'] = 60
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///example.sqlite"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

#File model
class Stock(db.Model):
    symbol = db.Column(db.String(50), unique=True, nullable=False, primary_key=True)
    isStock = db.Column(db.Boolean, nullable=False)
    prices = db.Column(db.String(10000))
    predictedPrices = db.Column(db.String(10000))
    data = db.Column(db.LargeBinary(length=(2**32)-1))

# Price and Predictions
@app.route('/image1', methods=['POST'])
@cross_origin()
def image1(): 
    try:
        coin = json.loads(request.data)
        coin = coin['coin']
        coin = str(coin).upper() + 'USDT'
        stock = Stock.query.filter_by(symbol=str(coin)).first()
        stockPrices = json.loads(str(stock.prices))
        predictedPrices = json.loads(str(stock.predictedPrices))
        # predictedPrices = np.array(stock.predictedPrices).reshape(-1)

        #Last 200 points
        prices = []
        for i in range(len(stockPrices) - predictionRequired, len(stockPrices)):
            prices.append(stockPrices[i])
        
        #First 200 points
        predicted = []
        if len(predictedPrices) >= predictedPoints - 1:
            for i in range(0, len(predictedPrices) - predictAhead):
                predicted.append(predictedPrices[i])
        
        plt.style.use('dark_background')   
        plt.plot(prices, color='white', label=f"Actual {stock.symbol} Price")
        plt.plot(predicted, color='green', label=f"Predicted {stock.symbol} Price")
        plt.title(str(stock.symbol) + ' Price and Predictions')
        plt.savefig(fname='plot1', transparent=True)
        plt.clf()
        return send_file('plot1.png')
        
    except:
        traceback.print_exc()
        return Response(status=404)

# Prediction
@app.route('/image2', methods=['POST'])
@cross_origin()
def image2(): 
    
    try:    
        coin = json.loads(request.data)
        coin = coin['coin']
        coin = str(coin).upper() + 'USDT'
        stock = Stock.query.filter_by(symbol=str(coin)).first()   
        predictedPrices = json.loads(str(stock.predictedPrices))  
        # predictedPrices = stock.predictedPrices

        #Last 60 points
        predicted2 = []
        if len(predictedPrices) >= predictAhead:
            for i in range(len(predictedPrices) - predictAhead, len(predictedPrices)):
                predicted2.append(predictedPrices[i])
        
        else:
            predicted2 = predictedPrices

        plt.style.use('dark_background')   
        plt.plot(predicted2, color='green', label=f"Predicted {stock.symbol} Price")
        plt.title(str(stock.symbol) + ' Future Predictions')
        plt.savefig(fname='plot2', transparent=True)
        return send_file('plot2.png')

    except:
        traceback.print_exc()
        return Response(status=404)

#Neural Network Settings, predictionRequired must be lower than dataPoints(Restart entire setup procedure if you change anything here, also do not change the predictedPoints)
predictionRequired = 200
predictAhead = 60
predictedPoints = predictAhead + 200

#Number of data points and refresh rate in seconds, dataPoints should stay 500
dataPoints = 500
refreshRate = 300

#Number of coins you want to track
numCoins = 10

#Collect data function
async def collectData():
    await asyncio.sleep(refreshRate)
    while True:
        try:                        
            start = t.time()
            stocks = Stock.query.all()
            tickers = client.get_all_tickers()
            #Fill information till there are enough data points
            for stock in stocks:
                if stock.isStock == True:
                    prices = json.loads(str(stock.prices))
                    # prices = stock.prices
                    for ticker in tickers:
                        if ticker['symbol'] == stock.symbol:
                            prices.append(float(ticker['price']))
                            while len(prices) > dataPoints:
                                prices.pop(0)
                            break
                
                    stock.prices = str(json.dumps(prices))
                    db.session.commit()

            end = t.time()                              
            newRefresh = round(refreshRate - (end - start))
            
            if newRefresh > 0:
                await asyncio.sleep(newRefresh)
            
        except:
            print("Collect Data:")
            traceback.print_exc()

async def predictPrice():
    while True: 
        try: 
            start = t.time()  
            stocks = Stock.query.all()
            for stock in stocks:
                K.clear_session()
                # prices = stock.prices
                # predictedPrices = stock.predictedPrices
                if stock.isStock == True:
                    try:   
                        prices = json.loads(str(stock.prices))

                        try:
                            predictedPrices = json.loads(str(stock.predictedPrices))
                    
                        except:
                            predictedPrices = []
                        
                        #Create file from database
                        with open('model.h5', "wb") as filehandler:
                            test = Stock.query.filter_by(symbol=str(stock.symbol) + 'Model.h5').first()
                            filehandler.write(test.data)
                        
                        model = load_model('model.h5')

                        scaler = MinMaxScaler(feature_range=(0, 1))
                        prices = np.array(prices).reshape(-1, 1)
                        scaler = scaler.fit(prices)
                        total_dataset = prices

                        model_inputs = np.array(total_dataset[len(total_dataset) - predictionRequired:]).reshape(-1, 1)
                        model_inputs = scaler.transform(model_inputs)

                        #Predict Next period
                        real_data = [model_inputs[len(model_inputs) - predictionRequired:len(model_inputs), 0]]
                        real_data = np.array(real_data)
                        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

                        prediction = model.predict(real_data)
                        prediction = scaler.inverse_transform(prediction)

                    except:
                        # print("-----------------------------------------------------------------------")
                        # traceback.print_exc()
                        prediction = 0  
                    
                    predictedPrices.append(float(prediction))
                    print(stock.symbol + ": " + str(prediction))
                    while len(predictedPrices) > dataPoints:
                        predictedPrices.pop(0) 
                    
                    stock.predictedPrices = str(json.dumps(predictedPrices))
                    db.session.commit()                 

            end = t.time()                                                
            newRefresh = round(refreshRate - (end - start))
            
            if newRefresh > 0:
                await asyncio.sleep(newRefresh)

        except:
            print("Predict Price:")
            traceback.print_exc()

async def train():
    while True:
        try:
            stocks = Stock.query.all()
            for stock in stocks:
                K.clear_session()
                if stock.isStock == True:
                    prices = json.loads(str(stock.prices))
                # prices = stock.prices
                    if len(prices) == 500:
                        
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
                            
                            #Create file from database
                            with open('model.h5', "wb") as filehandler:
                                test = Stock.query.filter_by(symbol=str(stock.symbol) + 'Model.h5').first()
                                filehandler.write(test.data)

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
                        
                        #Save file to database
                        with open('model.h5', 'rb') as filehandler:
                            try:
                                test = Stock.query.filter_by(symbol=str(stock.symbol) + 'Model.h5').first()
                                db.session.delete(test)
                                db.session.commit()
                            
                            except:
                                db.session.rollback()

                            test = Stock(symbol=str(stock.symbol) + 'Model.h5', data=filehandler.read(), isStock=False)
                            db.session.add(test)
                            db.session.commit()
                            
        except:
            print("Train:")
            traceback.print_exc()

if __name__ == '__main__':
    # try:
    #     db.session.query(Stock).delete()
    #     db.session.commit()
    
    # except:
    #     print("Database Empty!")

    # try:      
    #     num = 0
    #     tickers = client.get_all_tickers()
    #     for ticker in tickers:
    #         if ticker['symbol'].find('UP') == -1 and ticker['symbol'].find('DOWN') == -1 and ticker['symbol'].endswith('USDT') == True:
    #             test = []
    #             test.append(0)
    #             stock = Stock(symbol=ticker['symbol'], isStock=True, predictedPrices=str(json.dumps(test)))
    #             db.session.add(stock)
    #             db.session.commit()          
    #             num = num + 1

    #         if num == numCoins:
    #             break
        
    #     stocks = Stock.query.all()
    #     for stock in stocks:
    #         try:
    #             candles = client.get_klines(symbol=stock.symbol, interval=Client.KLINE_INTERVAL_5MINUTE)
                
    #             prices = []
    #             for candle in candles:
    #                 prices.append(float(candle[3]))
    #                 while len(prices) > dataPoints:
    #                     prices.pop(0)

    #             stock.prices = str(json.dumps(prices))    
    #             db.session.commit() 

    #         except:
    #             print("Invalid Symbol:" + str(stock.symbol))       

        # t1 = threading.Thread(target=asyncio.run, args=(collectData(),))
        # t1.setDaemon(True)
        # t1.start()
        # t2 = threading.Thread(target=asyncio.run, args=(predictPrice(),))
        # t2.setDaemon(True)
        # t2.start()
        # t3 = threading.Thread(target=asyncio.run, args=(train(),))
        # t3.setDaemon(True)
        # t3.start()
        # print("Starting")
        app.run()
    
    # except:
    #     print("Start Up:")
    #     traceback.print_exc()
