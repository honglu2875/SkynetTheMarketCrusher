from SMC.indicators import *

import pandas as pd
import numpy as np

data_df = pd.read_csv('data/ESU0-test.csv', skipinitialspace = True)

data_df.rename(columns = {'Date':'date', 'Time':'time', 'Open':'open', 'High':'high', \
                          'Low':'low', 'Last':'last', 'Volume':'volume', '# of Trades':'num of trades', \
                          'OHLC Avg':'OHLC avg', 'HLC Avg':'HLC avg', 'HL Avg':'HL avg', 'Bid Volume':'bid vol', \
                          'Ask Volume':'ask vol'}, inplace = True)

input_df = data_df[['date', 'time', 'open', 'high', 'low', 'last', 'volume', 'num of trades', 'ma', 'ema', 'CCI', 'study angle']].copy() # We only take these essential columns and generate our features and indicators on our own

input_df['calculated_ma'] = indicator(input_df['last'].to_numpy(), 'ma', length = 14)

input_df['calculated_ema'] = indicator(input_df['last'].to_numpy(), 'ema', length = 32)
input_df['calculated_CCI'] = indicator(input_df['last'].to_numpy(), 'cci', length = 80, multiplier = .015)
input_df['calculated_study_angle'] = indicator(input_df['calculated_ma'].to_numpy(), 'study angle', length = 10, value_per_point = 1)

print(input_df[:100][['calculated_ma','ma','calculated_ema','ema','calculated_CCI','CCI','calculated_study_angle','study angle']])
