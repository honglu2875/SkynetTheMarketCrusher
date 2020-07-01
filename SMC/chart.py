import numpy as np
import pandas as pd

def trade_to_cropped_pic(start, end, inp, pic_size=84, TICK_SIZE=.25):
    ###### Usage: trade_to_pic(inp) returns a matrix. It can be displayed as a grayscale image.
    ######        Current version turns every candle into a straight line with only the high/low info.
    ######
    ######        inp: a dataframe that records the range bar data.
    ######        BAR_SIZE: a constant representing the size of each range bar.
    ######
    ######        the shape of the output will be (pic_size, pic_size)
    ######        the most recent close price is always printed at [pic_size - 1, int(pic_size / 2)]

    pic = np.zeros((pic_size, pic_size))

    if not isinstance(inp, pd.core.frame.DataFrame):
        raise TypeError('input type is not supported.')

    if end < 0:
        return pic.reshape((pic_size, pic_size, 1))
    
    input_length = end - start + 1
    last_price = inp.loc[end, 'last']

    for i in range(pic_size):

        if i >= input_length: break

        low_offset = int((inp.loc[end - i, 'low'] - last_price) / TICK_SIZE)
        high_offset = int((inp.loc[end - i, 'high'] - last_price) / TICK_SIZE)

        for j in range( max(low_offset, -int(pic_size / 2)), min(high_offset + 1, int(pic_size / 2)) ):
            pic[pic_size - 1 - i][int(pic_size / 2) + j] = 255

    return pic.reshape((pic_size, pic_size, 1))
