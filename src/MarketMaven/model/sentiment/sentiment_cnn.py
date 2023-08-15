import re
import nltk
import tensorflow
from keras.datasets import imdb
from keras_preprocessing import sequence
from keras.models import load_model

from keras import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

nltk.downloader.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import plotly.express as px
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from urllib.request import urlopen, Request

class Sentiment_CNN():
    def __init__(self, tickers, tickers_sector, tickers_dict, tickers_name, number_of_shares, source_url):
        super(Sentiment_CNN, self).__init__()
        self.tickers = tickers
        self.tickers_sector = tickers_sector
        self.tickers_dict = tickers_dict
        self.tickers_name = tickers_name
        self.number_of_shares = number_of_shares
        self.source_url = source_url
        self.vader = SentimentIntensityAnalyzer()

        self.sectors = []
        self.names = []
        for tk in self.tickers:
            self.names.append(self.tickers_name[tk])
            self.sectors.append(self.tickers_sector[tk])

        self.d = {'Company Name': self.names, 'Sector': self.sectors}
        self.df_info = pd.DataFrame(data=self.d, index=self.tickers)

        self.table_news = {}
        for tk in self.tickers:
            url = self.source_url + tk
            req_url = Request(url=url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
            resp = urlopen(req_url)
            html = BeautifulSoup(resp)
            tn = html.find(id='news-table')
            self.table_news[tk] = tn

        self.news = []
        for file_name, news_table in self.table_news.items():
            for tr in news_table.findAll('tr'):
                txt = tr.a.get_text()
                scrape = tr.td.text.split()
                if len(scrape) == 1:
                    time = scrape[0]
                else:
                    date = scrape[0]
                    time = scrape[1]
                tk = file_name.split('_')[0]

                self.news.append([tk, date, time, txt])

        self.columns = ['ticker', 'date', 'time', 'headline']
        self.parsed_and_scored_news = pd.DataFrame(self.news, columns=self.columns)

        self.grouped_df = self.parsed_and_scored_news.groupby('ticker')['headline'].apply(' '.join).reset_index()

        self.grouped_df['unique_words'] = self.grouped_df['headline'].str.lower().apply(
            lambda x: re.sub(r'[^\w\s]', '', x)).str.split()
        self.grouped_df['unique_words'] = self.grouped_df['unique_words'].apply(
            lambda words: [word for word in words if not any(char.isdigit() for char in word)])

        self.grouped_df['unique_words'] = self.grouped_df['unique_words'].apply(set).apply(list)
        self.grouped_df.drop(columns=['headline'], inplace=True)

        self.vocabulary_size = 100000
        (self.X_train, self.y_train), (self.X_test, self.y_test) = imdb.load_data(num_words=self.vocabulary_size)

        self.word2id = imdb.get_word_index()
        self.id2word = {i: word for word, i in self.word2id.items()}
        self.grouped_df['indices'] = self.grouped_df['unique_words'].apply(lambda words: [self.word2id.get(word, 0) for word in words])
        self.grouped_df.drop(columns=['unique_words'], inplace=True)

        self.grouped_df['padded_indices'] = tensorflow.keras.preprocessing.sequence.pad_sequences(self.grouped_df['indices'],
                                                                                             maxlen=500,
                                                                                             padding='pre').tolist()
    def train_sent_cnn(self):
        max_words = 500
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=max_words)
        self.X_test = sequence.pad_sequences(self.X_test, maxlen=max_words)

        embedding_size = 32

        model = Sequential()
        model.add(Embedding(self.vocabulary_size, embedding_size, input_length=max_words))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 256
        num_epochs = 2
        X_valid, y_valid = self.X_train[:batch_size], self.y_train[:batch_size]
        X_train2, y_train2 = self.X_train[batch_size:], self.y_train[batch_size:]
        model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

        scores = model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Test accuracy:', scores[1])

        model.save("src/MarketMaven/pt_h5_pkl/sent_cnn.h5")

        return

    def scale_to_range(self, numbers, new_min, new_max):
        old_min = min(numbers)
        old_max = max(numbers)
        scaled_numbers = [((x - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min for x in numbers]
        return scaled_numbers

    def test_sent_cnn(self):
        model = load_model("src/MarketMaven/pt_h5_pkl/sent_cnn.h5")
        l1 = []
        for i in range(len(self.grouped_df["padded_indices"])):
            l1.append(self.grouped_df["padded_indices"][i])

        l1 = np.array(l1)
        pred = model.predict(l1)

        scaled_numbers = self.scale_to_range(pred, -0.2, 0.2)
        scaled_numbers = [value for [value] in scaled_numbers]

        sector = []
        company = []
        for i in range(len(self.grouped_df)):
            sector.append(self.tickers_sector[self.grouped_df['ticker'][i]])
            company.append(self.tickers_name[self.grouped_df['ticker'][i]])

        data = {
            'ticker': self.grouped_df['ticker'],
            'sector': sector,
            'company': company,
            'value': scaled_numbers
        }

        df = pd.DataFrame(data)

        figure = px.treemap(df, path=[px.Constant("sector"), 'sector', 'ticker'], color='value',
                            hover_data=['company', 'value'],
                            color_continuous_scale=['#FF0000', "#000000", '#00FF00'], color_continuous_midpoint=0)

        figure.update_traces(textposition="middle center")
        figure.update_layout(margin=dict(t=30, l=10, r=10, b=10), font_size=20)

        figure.write_image("src/MarketMaven/screenshots/sentiment_cnn.png")

        return





