## install required packages
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools

#SET TRAIN AND TRADE DATES
TRAIN_START_DATE = '2019-01-01'
TRAIN_END_DATE = '2020-07-01'
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2021-05-01'
symbols = [
    'aapl',
    'msft',
    'meta',

]
# df_raw is an instance of an object from YahooDownloader which fetches data within the parameters
# of (start date, end date, tickers(symbols), fetch data gets the data for the object through pandas

df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                                end_date = TRADE_END_DATE,
                                ticker_list = symbols
                         ).fetch_data()
# outputs the first few rows
df_raw.head()
#preprocess data
# adds new fields of technical indicators, Volatility Index, turbulence index which identifies periods of
# high risk

fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature = False)

processed = fe.preprocess_data(df_raw)

#creates a complete data set
list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)
processed_full

#prints number of rows of each
train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
print(len(train))
print(len(trade))

#saves the files to google drive
from google.colab import drive

drive.mount('/content/drive')

train_path = ''
trade_path = ''

with open(train_path, 'w', encoding = 'utf-8-sig') as f:
  train.to_csv(f)

with open(trade_path, 'w', encoding = 'utf-8-sig') as f:
  trade.to_csv(f)