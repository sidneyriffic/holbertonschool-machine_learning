#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

plt.close('all')
df = df.drop('Weighted_Price', axis=1)
df = df[::1440]
df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
df = df[df['Datetime'] > '2017']
df.fillna({'Volume_(BTC)': 0, 'Volume_(Currency)': 0}, inplace=True)
df['Close'].ffill(inplace=True)
df.fillna({'Open': df['Close'], 'High': df['Close'], 'Low': df['Close']},
          inplace=True)
df.set_index('Datetime', inplace=True)

df.plot()
plt.show()
