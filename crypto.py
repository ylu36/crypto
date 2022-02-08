import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from termcolor import colored as cl
import math
import pandas_ta as pta

short_ema = 14
long_ema = 77
adx_interval = 14
ticker = sys.argv[1]

def ADX(df, interval):
  df['-DM'] = df['low'].shift(1) - df['low']
  df['+DM'] = df['high'] - df['high'].shift(1)
  df['+DM'] = np.where((df['+DM'] > df['-DM']) & (df['+DM']>0), df['+DM'], 0.0)
  df['-DM'] = np.where((df['-DM'] > df['+DM']) & (df['-DM']>0), df['-DM'], 0.0)
  df['TR_TMP1'] = df['high'] - df['low']
  df['TR_TMP2'] = np.abs(df['high'] - df['close'].shift(1))
  df['TR_TMP3'] = np.abs(df['low'] - df['close'].shift(1))
  df['TR'] = df[['TR_TMP1', 'TR_TMP2', 'TR_TMP3']].max(axis=1)
  df['TR'+str(interval)] = df['TR'].rolling(interval).sum()
  df['+DMI'+str(interval)] = df['+DM'].rolling(interval).sum()
  df['-DMI'+str(interval)] = df['-DM'].rolling(interval).sum()
  df['+DI'+str(interval)] = df['+DMI'+str(interval)] /   df['TR'+str(interval)]*100
  df['-DI'+str(interval)] = df['-DMI'+str(interval)] / df['TR'+str(interval)]*100
  df['DI'+str(interval)+'-'] = abs(df['+DI'+str(interval)] - df['-DI'+str(interval)])
  df['DI'+str(interval)] = df['+DI'+str(interval)] + df['-DI'+str(interval)]
  df['DX'] = (df['DI'+str(interval)+'-'] / df['DI'+str(interval)])*100
  df['ADX'+str(interval)] = df['DX'].rolling(interval).mean()
  df['ADX'+str(interval)] = df['ADX'+str(interval)].fillna(df['ADX'+str(interval)].mean())
  del df['TR_TMP1'], df['TR_TMP2'], df['TR_TMP3'], df['TR'], df['TR'+str(interval)]
  del df['+DMI'+str(interval)], df['DI'+str(interval)+'-']
  del df['DI'+str(interval)], df['-DMI'+str(interval)]
  del df['+DI'+str(interval)], df['-DI'+str(interval)]
  del df['DX']
  return df
  
def strategy(df, use_adx=False, use_obv=False, use_rsi=False):
  signal_adx = [np.nan] * len(df['close'])
  signal_rsi = [np.nan] * len(df['close'])
  signal_obv = [np.nan] * len(df['close'])
  signal_total = [0] * len(df['close'])
  buy_price = [np.nan] * len(df['close'])
  sell_price = [np.nan] * len(df['close'])
  if use_adx:
    signal_adx, buy_price1, sell_price1 = adx_strategy(df)
    print_chart(df, buy_price1, sell_price1, 'ADX'+str(adx_interval))
  if use_obv:
    signal_obv, buy_price1, sell_price1 = obv_strategy(df)
    print_chart(df, buy_price1, sell_price1, 'obv')
  if use_rsi:
    signal_rsi, buy_price1, sell_price1 = rsi_strategy(df)
    print_chart(df, buy_price1, sell_price1, 'rsi')
  for i in range(len(df['close'])):
    signal_total[i] = signal_adx[i] or signal_rsi[i] or signal_obv[i]
    if signal_total[i] == 1:
      buy_price[i] = df['close'][i]
    elif signal_total[i] == -1:
      sell_price[i] = df['close'][i]
  print_compound_chart(df, buy_price, sell_price)

def adx_strategy(df):
  print("ADX strategy:")
  df.index = range(len(df['close']))
  signal = [0] * len(df['close'])
  buy_price = [np.nan] * len(df['close'])
  sell_price = [np.nan] * len(df['close'])
  for i in range(len(df['close'])):
    if df['ADX'+str(adx_interval)][i] < 30 or i == 0:
      signal[i] = (0)
    # When the shorter-term MA crosses above the longer-term MA, it's a buy signal
    elif df['ewm_' + str(short_ema)][i-1] < df['ewm_' + str(long_ema)][i-1] and df['ewm_' + str(short_ema)][i] >= df['ewm_' + str(long_ema)][i]:
      buy_price[i] = (df['close'][i])
      print(i, "buy@", buy_price[i])
      signal[i] = (1)
    # When the shorter-term MA crosses below the longer-term MA, it's a sell signal
    elif df['ewm_' + str(short_ema)][i-1] > df['ewm_' + str(long_ema)][i-1] and df['ewm_' + str(short_ema)][i] < df['ewm_' + str(long_ema)][i]:
      sell_price[i] = (df['close'][i])
      print(i, "sell@", sell_price[i])
      signal[i] = (-1)
    else:
      signal[i] = (0)
  print('\n')
  return signal, buy_price, sell_price

def rsi_strategy(df):
  print("RSI strategy:")
  df.index = range(len(df['close']))
  signal = [0] * len(df['close'])
  buy_price = [np.nan] * len(df['close'])
  sell_price = [np.nan] * len(df['close'])
  # leading indicator
  # only count first crossover because second is just returning to normal
  # noise when rsi crosses over just a bit
  for i in range(len(df['close'])):
    if math.isnan(df['rsi'][i]):
      signal[i] = (0)
    elif df['rsi'][i] <= 30 and df['rsi'][i- 1] > 30:
      buy_price[i] = (df['close'][i])
      print(i, "buy@", buy_price[i])
      signal[i] = (1)
    elif df['rsi'][i] >= 70 and df['rsi'][i- 1] < 70:
      sell_price[i] = (df['close'][i])
      print(i, "sell@", sell_price[i])
      signal[i] = (-1)
    else:
      signal[i] = (0)
  print('\n')
  return signal, buy_price, sell_price

def obv_strategy(df):
  print("OBV strategy:")
  df.index = range(len(df['close']))
  signal = [0] * len(df['close'])
  buy_price = [np.nan] * len(df['close'])
  sell_price = [np.nan] * len(df['close'])
  flag = -1
  for i in range(len(df['close'])):
    # If OBV > OBV_EMA Then Buy
    if df['obv'][i] > df['obv_ema'][i] and flag != 1:
      buy_price[i] = (df['close'][i])
      print(i, "buy@", buy_price[i])
      signal[i] = (1)
      flag = 1
    elif df['obv'][i] < df['obv_ema'][i] and flag != 0:
      sell_price[i] = (df['close'][i])
      print(i, "sell@", sell_price[i])
      signal[i] = (-1)
      flag = 0
    else:
      signal[i] = (0)
  print('\n')
  return signal, buy_price, sell_price

def plot_helper(ax1, ax2, buy_price, sell_price, df, show_subplot):
  green = mlines.Line2D([], [], color='green', marker='^', markersize=10, label='buy')
  red = mlines.Line2D([], [], color='red', marker='v', markersize=10, label='sell')
  for i in range(len(buy_price)):
    if buy_price[i] > 0:
      ax1.annotate(buy_price[i], xy= (i, buy_price[i]), xytext =(i, buy_price[i]*1.1), arrowprops = dict(facecolor ='green', shrink = 0.05),)
  for i in range(len(sell_price)):
    if sell_price[i] > 0:
      ax1.annotate(sell_price[i], xy= (i, sell_price[i]), xytext =(i, sell_price[i]*1.1), arrowprops = dict(facecolor ='red', shrink = 0.05),)
  if show_subplot:
    ax2.plot(df['ewm_'+str(short_ema)], color = '#26a69a', label = 'ewm' + str(short_ema), linewidth = 3, alpha = 0.3)
    ax2.plot(df['ewm_'+str(long_ema)], color = '#f44336', label = 'ewm' + str(long_ema), linewidth = 3, alpha = 0.3)
    ax2.legend()
  ax1.set_title(ticker + ' CLOSING PRICE')
  ax1.plot(df['close'], linewidth = 3, color = '#ff9800', alpha = 0.6)
  ax1.legend(handles=[red, green])

def print_compound_chart(df, buy_price, sell_price):
  ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
  ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
  plot_helper(ax1, ax2, buy_price, sell_price, df, True)
  plt.show()

def print_chart(df, buy_price, sell_price, indicator):
  show_subplot = True if indicator == 'ADX'+str(adx_interval) else False
  index = 17 if show_subplot else 11
  ax1 = plt.subplot2grid((index,1), (0,0), rowspan = 5, colspan = 1)
  ax2 = np.nan
  ax3 = plt.subplot2grid((index,1), (6,0), rowspan = 5, colspan = 1)
  if show_subplot:
    ax2 = plt.subplot2grid((index,1), (13,0), rowspan = 5, colspan = 1)
  plot_helper(ax1, ax2, buy_price, sell_price, df, show_subplot)
  ax3.plot(df[indicator], color = '#2196f3', label = indicator, linewidth = 3)
  if indicator == 'ADX' + str(adx_interval):
    ax3.axhline(30, color = 'grey', linewidth = 2, linestyle = '--')
  elif indicator == 'rsi':
    ax3.axhline(30, color = 'grey', linewidth = 2, linestyle = '--')
    ax3.axhline(70, color = 'grey', linewidth = 2, linestyle = '--')
  else:
    ax3.plot(df['obv_ema'], color = '#26a69a', label = 'obv_ema', linewidth = 3)
    ax3.legend()
  ax3.set_title(ticker + ' ' + indicator.upper())
  plt.show()

def create_position(df, signal):
  position = []
  for i in range(len(signal)):
      if signal[i] == -1 or signal[i] == 1:
        print(df['close'][i])
      if signal[i] > 0:
          position[i] = (1)
      else:
          position[i] = (0)
  print(position)
  for i in range(len(df['close'])):
      if signal[i] == 1:
          position[i] = 1
      elif signal[i] == -1:
          position[i] = 0
      else:
          position[i] = position[i-1]
  close_price = df['close']
  plus_di = df['+DM']
  minus_di = df['-DM']
  adx = df['ADX'+str(adx_interval)]
  signal = pd.DataFrame(signal).rename(columns = {0:'signal'}).set_index(df.index)
  position = pd.DataFrame(position).rename(columns = {0:'adx_position'}).set_index(df.index)

  frames = [close_price, plus_di, minus_di, adx, signal, position]
  strategy = pd.concat(frames, join = 'inner', axis = 1)
  return strategy

def main():
  df = pd.read_csv('data_' + ticker + '.csv', sep=',', skiprows=[i for i in range(1,200)])
  df = df.apply(pd.to_numeric, errors='coerce')
  df['change'] = df.apply(lambda row: (row['close'] / row['open'] - 1), axis=1)
  df['ewm_' + str(short_ema)] = df['open'].ewm(span=short_ema, adjust=False).mean()
  df['ewm_' + str(long_ema)] = df['open'].ewm(span=long_ema, adjust=False).mean()
  df = ADX(df, adx_interval)
  df['rsi'] = pta.rsi(df['close'], length = 14)
  df['obv'] = pta.obv(df['close'], df['volume'])
  df['obv_ema'] = df['obv'].ewm(com=30).mean() # note: try diff window
  print(ticker + " price data:")
  print(df)
  print('=======\n')
  strategy(df, use_adx=1, use_obv=0, use_rsi=1)

if __name__ == "__main__":
  main()