import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class DataGathering():
    def __init__(self, company_name):
        self.company_name = company_name

    def get_data(self):
        today = date.today()
        one_years_ago = today - relativedelta(years=10)
        data = yf.download(self.company_name, one_years_ago, today)  # 'RELIANCE.NS'
        train_size = int(len(data) * 0.8)
        return data[['Adj Close', 'Volume']], train_size

    def get_graph(self, y, m, w, d):
        today = date.today()
        previous = today - relativedelta(years=y, months=m, weeks=w, days=d)
        data = yf.download(self.company_name, previous, today)  # 'RELIANCE.NS'
        return data[['Adj Close', 'Volume']]