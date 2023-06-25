import yfinance as yf
from finta import TA
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class CLFDataGathering():
    def __init__(self, company_name):
        self.company_name = company_name

    def get_next_weekday(self, start_date):
        current_date = datetime.datetime.strptime(start_date.strftime('%Y-%m-%d'), '%Y-%m-%d').date()
        one_day = datetime.timedelta(days=1)

        if current_date.weekday() < 5:
            current_date += one_day

        while current_date.weekday() >= 5:
            current_date += one_day
        return current_date.strftime('%Y-%m-%d')

    def data_smoothing(self, data, alpha):
        return data.ewm(alpha=alpha).mean()

    def data_preprocess(self, data):
        indi = ['RSI', 'MACD', 'STOCH', 'ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

        for i in indi:
            d = eval('TA.' + i + '(data)')
            if not isinstance(d, pd.DataFrame):
                d = d.to_frame()
            data = data.merge(d, left_index=True, right_index=True)
        data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

        data['NormVol'] = data['Volume'] / data['Volume'].ewm(5).mean()

        data['e5'] = data['Close'] / data['Close'].ewm(5).mean()
        data['e15'] = data['Close'] / data['Close'].ewm(14).mean()
        data['e21'] = data['Close'] / data['Close'].ewm(21).mean()
        data['e50'] = data['Close'] / data['Close'].ewm(50).mean()

        pred = (data.shift(-1)['Close'] >= data['Close'])
        data['Pred'] = pred.astype(int)

        del (data['High'])
        del (data['Low'])
        del (data['Open'])
        del (data['Close'])
        del (data['Adj Close'])
        del (data['Volume'])

        return data

    def get_data(self, years, split):
        #Train
        today = datetime.date.today() + relativedelta(days=1)
        years_ago = today - relativedelta(years=years)
        data = yf.download(self.company_name, years_ago, today, progress=False)  # 'RELIANCE.NS'
        data = self.data_smoothing(data, 0.5)
        data = self.data_preprocess(data)
        data.dropna(inplace=True)

        x = data.loc[:, data.columns != 'Pred']
        y = data['Pred']

        X_forecast = x.iloc[-1, :]
        current = x.index[-1]
        tomorrow = self.get_next_weekday(current)
        x = x.iloc[:-1, :]
        y = y.iloc[1:]

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=split, shuffle=True, random_state=0)

        return train_x, test_x, train_y, test_y, X_forecast, tomorrow