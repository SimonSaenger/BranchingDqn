import requests
import pandas as pd
import numpy as np
import os.path
from utils import Config

import warnings
warnings.filterwarnings('ignore')


class Dataset:

    def __init__(self):
        self.config = Config()
        self.Ntest = self.config.Ntest
        self.url = self.config.furl
        self.fname = self.config.fname

        if not os.path.isfile(self.fname):
            r = requests.get(self.url)
            open(self.fname, 'wb').write(r.content)

    def get_train_test(self):
        df0 = pd.read_csv(self.fname, index_col=0, parse_dates=True)
        df0.dropna(axis=0, how='all', inplace=True)
        df0.dropna(axis=1, how='any', inplace=True)

        df_returns = pd.DataFrame()
        for name in df0.columns:
            df_returns[name] = np.log(df0[name]).diff()
        df_returns[self.config.aktie_price] = df0[self.config.aktie]
        df_returns.dropna(axis=0, how='any', inplace=True)

        train_data = df_returns.iloc[:-self.Ntest]
        test_data = df_returns.iloc[-self.Ntest:]

        return train_data, test_data
