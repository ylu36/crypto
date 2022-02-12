from email import message
import sys, os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas_ta as pta
from termcolor import colored as cl
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

short_ema = 14
long_ema = 77
adx_interval = 14
threshold = 0.1 # used for triggering email notification
ticker = sys.argv[1]
CRON = sys.argv[2] if len(sys.argv) > 2 else False
sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))

def email(title, content):
  message = Mail(
    from_email='jameslu0326@outlook.com',
    to_emails='cozyrat.automation@gmail.com',
    subject=title,
    html_content=content)
  try:
    response = sg.send(message)
    print("email is sent with " + str(response.status_code))
  except Exception as e:
    print("error sending email:")
    print(e.message)

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
  
def strategy(df, use_adx=False, use_obv=False, use_rsi=False, use_mfi=False):
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
  if use_mfi:
    signal_mfi, buy_price1, sell_price1 = mfi_strategy(df)
    print_chart(df, buy_price1, sell_price1, 'mfi')
  for i in range(len(df['close'])):
    signal_total[i] = signal_adx[i] or signal_rsi[i] or signal_obv[i] or signal_mfi[i]
    if signal_total[i] == 1:
      buy_price[i] = df['close'][i]
    elif signal_total[i] == -1:
      sell_price[i] = df['close'][i]
  
  # send email
  signal = "HOLD"
  if signal_total[-1] == 1:
    signal = "BUY"
  elif signal_total[-1] == -1:
    signal = "SELL"
  title = "Crypto Monitor Digest for " + ticker + ' on ' + df['timestamp'].iloc[-1]
  content = "<h2>" + ticker + " signal is <strong>" + signal + "</strong></h2>" + \
            "<i>Current price for " + df['timestamp'].iloc[-1] + ":</i>" + "<br>" + \
            "CLOSE: $" + str(df['close'].iloc[-1]) + '<br>' + \
            "HIGH: $" + str(df['high'].iloc[-1]) + '<br>' + \
            "LOW: $" + str(df['low'].iloc[-1]) + '<br>' + \
            "MFI: " + str(df['mfi'].iloc[-1]) + '<br>' + \
            "ADX: " + str(df['ADX'+str(adx_interval)].iloc[-1]) + '<br>' + \
            "RSI: " + str(df['rsi'].iloc[-1]) + '<br>'
  if CRON:
    email(title, content)
  print_compound_chart(df, buy_price, sell_price)

def mfi_strategy(df):
  # print("MFI strategy:")
  df.index = range(len(df['close']))
  signal = [0] * len(df['close'])
  buy_price = [np.nan] * len(df['close'])
  sell_price = [np.nan] * len(df['close'])
  # leading indicator
  # only count first crossover because second is just returning to normal
  # noise when rsi crosses over just a bit
  for i in range(len(df['close'])):
    if math.isnan(df['mfi'][i]):
      signal[i] = (0)
    elif df['mfi'][i-1] <= 20*(1+threshold) and df['mfi'][i] > 20*(1+threshold):
      buy_price[i] = (df['close'][i])
      # print(i, "buy@", buy_price[i])
      signal[i] = (1)
    elif df['mfi'][i-1] >= 80*(1-threshold) and df['mfi'][i] < 80*(1-threshold):
      sell_price[i] = (df['close'][i])
      # print(i, "sell@", sell_price[i])
      signal[i] = (-1)
    else:
      signal[i] = (0)
  print('\n')
  # create_position(df, signal)
  return signal, buy_price, sell_price
 
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
      # print(df['timestamp'][i], "buy@", buy_price[i])
      signal[i] = (1)
    # When the shorter-term MA crosses below the longer-term MA, it's a sell signal
    elif df['ewm_' + str(short_ema)][i-1] > df['ewm_' + str(long_ema)][i-1] and df['ewm_' + str(short_ema)][i] < df['ewm_' + str(long_ema)][i]:
      sell_price[i] = (df['close'][i])
      # print(df['timestamp'][i], "sell@", sell_price[i])
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
    elif df['rsi'][i-1] <= 30 and df['rsi'][i] > 30:
      buy_price[i] = (df['close'][i])
      # print(df['timestamp'][i], "buy@", buy_price[i])
      signal[i] = (1)
    elif df['rsi'][i-1] >= 70 and df['rsi'][i] < 70:
      sell_price[i] = (df['close'][i])
      # print(df['timestamp'][i], "sell@", sell_price[i])
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
      print(df['timestamp'][i], "buy@", buy_price[i])
      signal[i] = (1)
      flag = 1
    elif df['obv'][i] < df['obv_ema'][i] and flag != 0:
      sell_price[i] = (df['close'][i])
      print(df['timestamp'][i], "sell@", sell_price[i])
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
  if CRON:
    return
  ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
  ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
  plot_helper(ax1, ax2, buy_price, sell_price, df, True)
  plt.show()

def print_chart(df, buy_price, sell_price, indicator):
  if CRON:
    return
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
  elif indicator == 'mfi':
    ax3.axhline(20, color = 'grey', linewidth = 2, linestyle = '--')
    ax3.axhline(80, color = 'grey', linewidth = 2, linestyle = '--')
  else:
    ax3.plot(df['obv_ema'], color = '#26a69a', label = 'obv_ema', linewidth = 3)
    ax3.legend()
  ax3.set_title(ticker + ' ' + indicator.upper())
  plt.show()

# loop through all signals:
# if buy, buy $1000 and increase buy counter
# if sell, sell (100/counter) % coins
def create_position(df, signal):
  buy = 0
  sell = 0
  coins = 0
  buy_counter = 0

  for i in range(len(signal)):
    if i > 0 and buy_counter == 0 and signal[i] == -1:
        signal[i] = 0
    if signal[i] == 1:
      buy += 1000
      coins = 1000 / df['close'][i]
      buy_counter += 1
      print("buy@", i, buy, coins)
    elif signal[i] == -1:
      coins = coins * (1-1/buy_counter)
      sell += coins * 1/buy_counter * df['close'][i]
      buy_counter -= 1
      print("sell@", i, sell, coins)
  print(buy, sell, coins)
  return

def main():
  df = pd.read_csv('data_' + ticker + '.csv', sep=',', skiprows=[i for i in range(1,100)])
  temp = df['timestamp']
  df = df.apply(pd.to_numeric, errors='coerce')
  df['timestamp'] = temp
  df['change'] = df.apply(lambda row: (row['close'] / row['open'] - 1), axis=1)
  df['ewm_' + str(short_ema)] = df['open'].ewm(span=short_ema, adjust=False).mean()
  df['ewm_' + str(long_ema)] = df['open'].ewm(span=long_ema, adjust=False).mean()
  df = ADX(df, adx_interval)
  df['rsi'] = pta.rsi(df['close'], length = 14)
  df['mfi'] = pta.mfi(df['high'], df['low'], df['close'], df['volume'])
  df['obv'] = pta.obv(df['close'], df['volume'])
  df['obv_ema'] = df['obv'].ewm(com=30).mean() # note: try diff window
  print(ticker + " price data:")
  print(df.tail(2))
  print('=======\n')
  strategy(df, use_adx=0, use_obv=0, use_rsi=0, use_mfi=1)

if __name__ == "__main__":
  main()