import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
from SMC.datareader import *

a = [datetime(2020,1,10,16,30,0), datetime(2020,1,10,18,0,0), datetime(2020,1,11,14,30,0), datetime(2020,1,12,18,20,0)]
b = [datetime(2020,1,10,18,1,1), datetime(2020,1,10,18,30,0), datetime(2020,1,11,14,30,0), datetime(2020,1,11,16,30,0), datetime(2020,1,11,18,0,0), datetime(2020,1,12,18,20,0), datetime(2020,1,12,18,30,0)]

print(a)
print(b)
print(datepreprocess(a,b))

