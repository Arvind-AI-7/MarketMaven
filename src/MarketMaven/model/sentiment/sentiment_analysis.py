import nltk
nltk.downloader.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import plotly.express as px
import pandas as pd

from bs4 import BeautifulSoup
from urllib.request import urlopen, Request

class Sentiment_Analysis():
    def __init__(self, tickers, tickers_sector, tickers_dict, tickers_name, number_of_shares, source_url):
        super(Sentiment_Analysis, self).__init__()
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

    def sentiment(self):
        table_news = {}
        for tk in self.tickers:
            url = self.source_url + tk
            req_url = Request(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
            resp = urlopen(req_url)
            html = BeautifulSoup(resp)
            tn = html.find(id='news-table')
            table_news[tk] = tn

        news = []
        for file_name, news_table in table_news.items():
            for tr in news_table.findAll('tr'):
                txt = tr.a.get_text()
                scrape = tr.td.text.split()
                if len(scrape) == 1:
                    time = scrape[0]
                else:
                    date = scrape[0]
                    time = scrape[1]
                tk = file_name.split('_')[0]

                news.append([tk, date, time, txt])

        columns = ['Ticker', 'Date', 'Time', 'Headline']
        parsed_data = pd.DataFrame(news, columns=columns)
        sc = parsed_data['Headline'].apply(self.vader.polarity_scores).tolist()
        scores = pd.DataFrame(sc)

        parsed_data = parsed_data.join(scores, rsuffix='_right')
        parsed_data['Date'] = pd.to_datetime(parsed_data.Date).dt.date

        parsed_data = parsed_data.drop(["Headline", "Date", "Time"], axis=1)
        avg_scores = parsed_data.groupby(['Ticker']).mean()

        df = self.df_info.join(avg_scores)
        df = df.rename(columns={"compound": "Sentiment Score", "neg": "Negative", "neu": "Neutral", "pos": "Positive"})
        df = df.reset_index()
        df = df.rename(columns={'index': 'Ticker'})

        figure = px.treemap(df, path=[px.Constant("Sectors"), 'Sector', 'Ticker'], color='Sentiment Score',
                            hover_data=['Company Name', 'Negative', 'Neutral', 'Positive', 'Sentiment Score'],
                            color_continuous_scale=['#FF0000', "#000000", '#00FF00'], color_continuous_midpoint=0)

        figure.update_traces(textposition="middle center")
        figure.update_layout(margin=dict(t=30, l=10, r=10, b=10), font_size=20)

        figure.write_image("src/MarketMaven/screenshots/sentiment.png")

        return

