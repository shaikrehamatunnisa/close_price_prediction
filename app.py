from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import pickle
import numpy as np
import datetime
from pycaret.regression import *

app = Flask(__name__)

def download_stock_data(stock_name):
    
    # Download stock data from yfinance
    start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    end_date = datetime.datetime.now()

    # Retrieve the data from Yahoo Finance
    tickerData = yf.Ticker(stock_name)
    stock_data = tickerData.history(start=start_date, end=end_date)

    return stock_data

def preprocess_stock_data(stock_data):
    # Preprocess stock data
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    return stock_data

def train_best_model(stock_data):
    # Initialize PyCaret regression setup
    reg = setup(data=stock_data, target='Close', train_size=0.8, session_id=123)

    # Compare and select the best ML model
    best_model = compare_models(sort='R2')
    return best_model

def predict_stock_price(stock_data, best_model):
    # Predict stock prices using the trained ML model
    predictions = predict_model(best_model, data=stock_data)
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    stock_name = request.form['stock_name']

    # Download stock data
    stock_data = download_stock_data(stock_name)

    # Preprocess stock data
    stock_data = preprocess_stock_data(stock_data)

    # Train the best ML model
    best_model = train_best_model(stock_data)

    # Predict stock prices
    predictions = predict_stock_price(stock_data, best_model)

    # Get predicted stock prices for the next 1 week
    predicted_prices = predictions.tail(7)['Close']

    return render_template('result.html', stock_name=stock_name, predicted_prices=predicted_prices)


with open('model.pkl','rb') as file:
    model=pickle.load(file)
    

@app.route('/banknifty', methods=['POST', 'GET'])
def banknifty():
    # return "Hello World"
    pred_value = 0
    if request.method == 'POST':
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low= float(request.form['Low'])

        data = np.array([[Open,High,Low]])
        pred_value = model.predict(data)
        pred_value = np.round(pred_value[0],2)

    return render_template('index1.html', pred_value=pred_value)

with open('AAL_model.pkl','rb') as file:
    model_1=pickle.load(file)
    

@app.route('/AAL', methods=['POST', 'GET'])
def AAL():
    # return "Hello World"
    pred_value = 0
    if request.method == 'POST':
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low= float(request.form['Low'])

        data = np.array([[Open,High,Low]])
        pred_value = model_1.predict(data)
        pred_value = np.round(pred_value[0],2)
    return render_template('index1.html', pred_value=pred_value)

with open('AAPL_model.pkl','rb') as file:
    model_2=pickle.load(file)
    

@app.route('/AAPL', methods=['POST', 'GET'])
def AAPL():
    # return "Hello World"
    pred_value = 0
    if request.method == 'POST':
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low= float(request.form['Low'])

        data = np.array([[Open,High,Low]])
        pred_value = model_2.predict(data)
        pred_value = np.round(pred_value[0],2)

    return render_template('index1.html', pred_value=pred_value)


with open('AMD_model.pkl','rb') as file:
    model_3=pickle.load(file)
    

@app.route('/AMD', methods=['POST', 'GET'])
def AMD():
    # return "Hello World"
    pred_value = 0
    if request.method == 'POST':
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low= float(request.form['Low'])

        data = np.array([[Open,High,Low]])
        pred_value = model_3.predict(data)
        pred_value = np.round(pred_value[0],2)

    return render_template('index1.html', pred_value=pred_value)

with open('AMZN_model.pkl','rb') as file:
    model_4=pickle.load(file)
    

@app.route('/AMZN', methods=['POST', 'GET'])
def AMZN():
    # return "Hello World"
    pred_value = 0
    if request.method == 'POST':
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low= float(request.form['Low'])

        data = np.array([[Open,High,Low]])
        pred_value = model_4.predict(data)
        pred_value = np.round(pred_value[0],2)

    return render_template('index1.html', pred_value=pred_value)

with open('BAC_model.pkl','rb') as file:
    model_5=pickle.load(file)
    

@app.route('/BAC', methods=['POST', 'GET'])
def BAC():
    # return "Hello World"
    pred_value = 0
    if request.method == 'POST':
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low= float(request.form['Low'])

        data = np.array([[Open,High,Low]])
        pred_value = model_5.predict(data)
        pred_value = np.round(pred_value[0],2)

    return render_template('index1.html', pred_value=pred_value)



with open('PLTR_model.pkl','rb') as file:
    model_6=pickle.load(file)
    

@app.route('/PLTR', methods=['POST', 'GET'])
def PLTR():
    # return "Hello World"
    pred_value = 0
    if request.method == 'POST':
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low= float(request.form['Low'])

        data = np.array([[Open,High,Low]])
        pred_value = model_6.predict(data)
        pred_value = np.round(pred_value[0],2)

    return render_template('index1.html', pred_value=pred_value)


with open('SHOP_model.pkl','rb') as file:
    model_7=pickle.load(file)
    

@app.route('/SHOP', methods=['POST', 'GET'])
def SHOP():
    # return "Hello World"
    pred_value = 0
    if request.method == 'POST':
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low= float(request.form['Low'])

        data = np.array([[Open,High,Low]])
        pred_value = model_7.predict(data)
        pred_value = np.round(pred_value[0],2)

    return render_template('index1.html', pred_value=pred_value)


with open('tesla_model.pkl','rb') as file:
    model_8=pickle.load(file)
    

@app.route('/tesla', methods=['POST', 'GET'])
def tesla():
    # return "Hello World"
    pred_value = 0
    if request.method == 'POST':
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low= float(request.form['Low'])

        data = np.array([[Open,High,Low]])
        pred_value = model_8.predict(data)
        pred_value = np.round(pred_value[0],2)

    return render_template('index1.html', pred_value=pred_value)

with open('UBER_model.pkl','rb') as file:
    model_9=pickle.load(file)
    

@app.route('/UBER', methods=['POST', 'GET'])
def UBER():
    # return "Hello World"
    pred_value = 0
    if request.method == 'POST':
        Open = float(request.form['Open'])
        High = float(request.form['High'])
        Low= float(request.form['Low'])

        data = np.array([[Open,High,Low]])
        pred_value = model_9.predict(data)
        pred_value = np.round(pred_value[0],2)

    return render_template('index1.html', pred_value=pred_value)


if __name__ == '__main__':
    app.run(debug=True)
