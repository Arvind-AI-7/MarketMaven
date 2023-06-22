import yfinance as yf

import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class DataGathering():
    def __init__(self, company_name):
        self.company_name = company_name

    def get_next_weekday(self, start_date):
        current_date = datetime.datetime.strptime(start_date.strftime('%Y-%m-%d'), '%Y-%m-%d').date()
        one_day = datetime.timedelta(days=1)

        print(current_date)

        if current_date.weekday() < 5:
            current_date += one_day

        while current_date.weekday() >= 5:
            current_date += one_day
        return current_date.strftime('%Y-%m-%d')

    def data_smoothing(self, data, alpha):
        return data.ewm(alpha=alpha).mean()

    def get_data(self,years,split):

        #Train
        today = datetime.date.today() + relativedelta(days=1)
        years_ago = today - relativedelta(years=years)
        data = yf.download(self.company_name, years_ago, today, progress=False)  # 'RELIANCE.NS'
        data.dropna(inplace=True)
        data = self.data_smoothing(data, 0.5)
        x = data[['Open', 'High', 'Low', 'Volume']]
        y = data['Adj Close']

        X_forecast = x.iloc[-1, :]
        current = x.index[-1]
        tomorrow = self.get_next_weekday(current)
        x = x.iloc[:-1, :]
        y = y.iloc[1:]

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=split, shuffle=False, random_state=0)

        return train_x, test_x, train_y, test_y, X_forecast, tomorrow

    def get_graph(self, y, m, w, d):
        today = date.today()
        previous = today - relativedelta(years=y, months=m, weeks=w, days=d)
        data = yf.download(self.company_name, previous, today)  # 'RELIANCE.NS'
        return data[['Adj Close', 'Volume']]