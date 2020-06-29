import numpy as np 
import pandas as pd 
import math



def valid_length(param, indicator_name):
  if 'length' not in param.keys():
    raise NameError(f'Length must be specified for {indicator_name}.')
    return False
  elif not isinstance(param['length'], int):
    raise TypeError('Length must be an integer.')
    return False
  elif param['length'] <= 0:
    raise ValueError('Length must be a positive integer.')
    return False
  return True


def ema(inp, length):
  assert isinstance(inp, list) or isinstance(inp, np.ndarray)
  c = 2 / (length + 1)
  result = []
  for i in range(len(inp)):
    if i == 0:
      result.append(inp[0])
    elif i < length - 1:
      if result[-1] == 0:
        result.append( 2 / (i + 2) * inp[i] + (1 - 2 / (i + 2)) * inp[i-1] )
      else:
        result.append( 2 / (i + 2) * inp[i] + (1 - 2 / (i + 2)) * result[-1] )
    else:
      if result[-1] == 0:
        result.append( c * inp[i] + (1 - c) * inp[i-1] )
      else:
        result.append( c * inp[i] + (1 - c) * result[-1] )
  return result

def ma(inp, length):
  assert isinstance(inp, list) or isinstance(inp, np.ndarray)
  result = []
  rolling_sum = 0
  for i in range(len(inp)):
    if i < length:
      rolling_sum = rolling_sum + inp[i]
      result.append( rolling_sum / (i + 1) )
    else:
      rolling_sum = rolling_sum + inp[i] - inp[i - length]
      result.append( rolling_sum / length )
  return result

def study_angle(inp, length, value_per_point):
  assert isinstance(inp, list) or isinstance(inp, np.ndarray)
  result = []
  for i in range(len(inp)):
    if i < length:
      result.append(0)
    elif i >= length:
      result.append( math.atan( (inp[i] - inp[i - length]) / (length * value_per_point) ) * 180 / math.pi )
  return result

def cci(inp, length, moving_average = ma, multiplier = .015):
  assert isinstance(inp, list) or isinstance(inp, np.ndarray)
  result = []
  ma_list = moving_average(inp, length)
  for i in range(len(inp)):
    if i < length:
      denom = 0
      for j in range(i + 1):
        denom = denom + abs( ma_list[i] - inp[j] )
      if denom == 0:
        denom = 1e-5 #Avoid division by zero
      result.append( (inp[i] - ma_list[i]) / denom * (i + 1) / multiplier )
    else:
      denom = 0
      for j in range(i - length + 1, i + 1):
        denom = denom + abs( ma_list[i] - inp[j] )
      if denom == 0:
        denom = 1e-5 #Avoid division by zero
      result.append( (inp[i] - ma_list[i]) / denom * length / multiplier )
  return result



def indicator(data, indicator_name, **kwargs):

  ###### Usage: indicator(data, indicator_name, length = ??, other parameters...) returns a list of the result of the specified indicator.
  ######        data needs to be a list, a numpy array, or a pandas Series. The type check is imposed only because it's my personal style. 
  ######        indicator_name supports: 'ma', 'ema', 'study angle', 'cci'.
  ######                                  all indicators require the input of 'length'.
  ######                                  study angle requires an extra input of 'value_per_point'.
  ######                                  cci has optional parameters of 'moving_average' and 'multiplier'. By default we use simple moving average and multiplier = .015

  inp = data
  result = []
  if isinstance(inp, pd.core.series.Series):
    inp = inp.tolist()
  elif isinstance(inp, list) or isinstance(inp, np.ndarray):
    pass
  else:
    raise TypeError(f'The type {type(inp)} of the input data is not supported.')
  ###### remark: I can let it handle DataFrame later. So far there is no need.


  if isinstance(indicator_name, str):
    if indicator_name.lower() == 'ma':
      if valid_length(kwargs, indicator_name):
        result = ma(inp, kwargs['length'])

    elif indicator_name.lower() == 'ema':
      if valid_length(kwargs, indicator_name):
        result = ema(inp, kwargs['length'])

    elif indicator_name.lower() == 'study angle':
      if valid_length(kwargs, indicator_name):
        if 'value_per_point' not in kwargs:
          result = study_angle(inp, kwargs['length'], 1)
        else:
          result = study_angle(inp, kwargs['length'], kwargs['value_per_point'])

    elif indicator_name.lower() == 'cci':
      if valid_length(kwargs, indicator_name):
        if 'moving_average' not in kwargs or kwargs['moving_average'] == 'ma':
          moving_average = ma
        elif kwargs['moving_average'] == 'ema':
          moving_average = ema
        else:
          raise ValueError('The moving average keyword is not recognized.')

        if 'multiplier' not in kwargs:
          multiplier = .015
        elif isinstance(kwargs['multiplier'], int) or isinstance(kwargs['multiplier'], float):
          multiplier = kwargs['multiplier']
        else:
          raise TypeError('multiplier must be an integer or float.')
        result = cci(inp, kwargs['length'], moving_average = moving_average, multiplier = multiplier)

    else:
      raise NameError('The indicator is unspecified or unsupported.')
  else:
    raise TypeError('The indicator name must be a string.')


  if isinstance(data, pd.core.series.Series):
    return pd.Series(result)
  else:
    return result
