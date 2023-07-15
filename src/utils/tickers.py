class Tickers:
    def __init__(self):
        super(Tickers, self).__init__()
        self.tickers_dict = {'AXP':1, 'AMGN':1, 'AAPL':1, 'BA':1, 'CAT':1, 'CSCO':1, 'CVX':1, 'GS':1, 'HD':1, 'HON':1,
                             'IBM':1, 'INTC':1, 'JNJ':1, 'KO':1, 'JPM':1, 'MCD':1, 'MMM':1, 'MRK':1, 'MSFT':1, 'NKE':1,
                             'PG':1, 'TRV':1, 'UNH':1, 'CRM':1, 'VZ':1, 'V':1, 'WBA':1, 'WMT':1, 'DIS':1, 'DOW':1}

        self.tickers_name = {"AXP": "American Express Co", "AMGN": "Amgen Inc", "AAPL": "Apple Inc", "BA": "Boeing Co",
                             "CAT": "Caterpillar Inc", "CSCO": "Cisco Systems Inc", "CVX": "Chevron Corp", "GS": "Goldman Sachs Group Inc",
                             "HD": "Home Depot Inc", "HON": "Honeywell International Inc", "IBM": "International Business Machines Corp",
                             "INTC": "Intel Corp", "JNJ": "Johnson & Johnson", "KO": "Coca-Cola Co", "JPM": "JPMorgan Chase & Co",
                             "MCD": "McDonald’s Corp", "MMM": "3M Co", "MRK": "Merck & Co Inc", "MSFT": "Microsoft Corp", "NKE": "Nike Inc",
                             "PG": "Procter & Gamble Co", "TRV": "Travelers Companies Inc", "UNH": "UnitedHealth Group Inc",
                             "CRM": "Salesforce Inc", "VZ": "Verizon Communications Inc", "V": "Visa Inc", "WBA": "Walgreens Boots Alliance Inc",
                             "WMT": "Walmart Inc", "DIS": "Walt Disney Co", "DOW": "Dow Inc"}

        self.tickers_sector = {"AXP": "Financial Services", "AMGN": "Healthcare", "AAPL": "Technology", "BA": "Industrials",
                               "CAT": "Industrials", "CSCO": "Technology", "CVX": "Energy", "GS": "Financial Services", "HD": "Consumer Cyclical",
                               "HON": "Industrials", "IBM": "Technology", "INTC": "Technology", "JNJ": "Healthcare", "KO": "Consumer Defensive",
                               "JPM": "Financial Services", "MCD": "Consumer Cyclical", "MMM": "Industrials", "MRK": "Healthcare",
                               "MSFT": "Technology", "NKE": "Consumer Cyclical", "PG": "Consumer Defensive", "TRV": "Financial Services",
                               "UNH": "Healthcare", "CRM": "Technology", "VZ": "Communication Services", "V": "Financial Services",
                               "WBA": "Healthcare", "WMT": "Consumer Defensive", "DIS": "Communication Services", "DOW": "Basic Materials"}

        self.tickers = self.tickers_dict.keys()
        self.number_of_shares = self.tickers_dict.values()

        self.source_url = 'https://finviz.com/quote.ashx?t='
