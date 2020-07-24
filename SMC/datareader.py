import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta

def datareader(filepath):
    """Read a Sierrachart csv and return a dataframe with 'date', 'time', 'open', 'high', 'low', 'last', 'volume', 'num of trades'
    """
    data_df = pd.read_csv(filepath, skipinitialspace = True)
    column_replacement = {'Date':'date', 'Time':'time', 'Open':'open', 'High':'high', 'Low':'low', 'Last':'last', 'Volume':'volume', '# of Trades':'num of trades'}

    #Keep the following line for potential modification
    #'OHLC Avg':'OHLC avg', 'HLC Avg':'HLC avg', 'HL Avg':'HL avg', 'Bid Volume':'bid vol', 'Ask Volume':'ask vol'

    data_df.rename(columns = column_replacement, inplace = True)

    data_df['date'] = data_df['date'].apply(lambda x: datetime.strptime(x.strip(), '%Y/%m/%d').date())
    data_df['time'] = data_df['time'].apply(lambda x: datetime.strptime(x.strip().split('.', 1)[0], '%H:%M:%S').time())
    data_df['datetime'] = data_df.apply(lambda x: datetime.combine(x['date'], x['time']), axis=1)


    return data_df[['date', 'time', 'datetime', 'open', 'high', 'low', 'last', 'volume', 'num of trades']].copy()

def datepreprocess(inp1, inp2):
    """
        Preprocessing trading days.
        Input:
            inp1: a list of ordered datetime.
            inp2: a list of ordered datetime
        Return:
            (input_dates, input_date_index, index_mapping)
            input_dates: a list of unique and ordered trading days (Market is closed between 17:00-18:00. After 18:00, data is counted as the next trading day.)
            input_date_index: a list. input_date_index[input_dates[i]] is the index of the last row of data in the trading day input_dates[i]
            index_mapping: a list. Denote j = input_mapping[i]. j is the index of the inp2 entry such that:
                            inp2[j + 1] (the candle closing time) is strictly smaller than inp1[i + 1] (the closing time of the current candle).
                            If there is no such candle in inp2, it's -2 or -1 depending on whether the first entry of inp2 has a smaller datetime.
                            More concretely: if the candle i on inp1 is closed, the input_mapping[i]-th candle on inp2 is the last closed candle that can be observed.
                            (For the last entry, we set input_mapping[len(input_mapping)-1] = input_mapping[len(input_mapping)-2])
    """

    input_dates = []
    input_date_index = []
    date_offset = [True if x.time() > time(17, 0, 0) else False for x in inp1]

    input_mapping = []

    mapping_offset = -1

    for i in range(len(inp1)):

        ###### Processing dates
        if date_offset[i]:
            trading_day = inp1[i].date() + timedelta(1)
        else:
            trading_day = inp1[i].date()

        if len(input_dates) == 0 or input_dates[-1] != trading_day:
            if len(input_dates) != 0: input_date_index.append(i - 1)
            input_dates.append(trading_day)

        ###### Mapping to bigger time frame
        if i == len(inp1) - 1:
            input_mapping.append(input_mapping[i - 1])
            continue

        while mapping_offset < len(inp2) - 1 and inp2[mapping_offset + 1] < inp1[i + 1]:
            mapping_offset += 1
        input_mapping.append(mapping_offset - 1)


    input_date_index.append(len(inp1) - 1)

    return input_dates, input_date_index, input_mapping
