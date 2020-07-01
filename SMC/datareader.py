import numpy as np
import pandas as pd

def datareader(filepath):
    """Read a Sierrachart csv and return a dataframe with 'date', 'time', 'open', 'high', 'low', 'last', 'volume', 'num of trades'
    """
    data_df = pd.read_csv(filepath, skipinitialspace = True)
    column_replacement = {'Date':'date', 'Time':'time', 'Open':'open', 'High':'high', 'Low':'low', 'Last':'last', 'Volume':'volume', '# of Trades':'num of trades'}
    
    #Keep the following line for potential modification
    #'OHLC Avg':'OHLC avg', 'HLC Avg':'HLC avg', 'HL Avg':'HL avg', 'Bid Volume':'bid vol', 'Ask Volume':'ask vol'
    
    data_df.rename(columns = column_replacement, inplace = True)

    return data_df[['date', 'time', 'open', 'high', 'low', 'last', 'volume', 'num of trades']].copy()
