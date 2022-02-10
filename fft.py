# https://medium.com/analytics-vidhya/the-hidden-secrets-of-the-bitcoin-price-201c52d0f11d
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def FTT(df):
  df.index = range(len(df['close']))
  price = []
  time = []
  for i in range(len(df['close'])):
    price.append(df['close'][i])
    time.append(i)
  price = np.array(price, dtype=np.float32)
  time = np.array(time, dtype=int)
  price_dt = price[1:] - price[:-1]
  # filter
  filter_width = 12
  def gaussian_kernel_1d(filter_width):
      #99% of the values
      sigma = (filter_width)/2.33
      norm = 1.0 / (np.sqrt(2*np.pi) * sigma)
      kernel = [norm * np.exp((-1)*(x**2)/(2 * sigma**2)) for x in range(-filter_width, filter_width + 1)]
      return np.float32(kernel / np.sum(kernel))
  f = tf.reshape(gaussian_kernel_1d(filter_width), [-1, 1, 1])
  tf_price = tf.reshape(tf.constant(price, dtype=tf.float32), [1, -1, 1])
  tf_price = tf.reshape(tf.nn.conv1d(tf_price, filters=f, stride=1, padding='VALID'), [-1])
  # padding is necessary to keep same dim
  tf_price = tf.concat([ tf.constant(tf_price[0].numpy(), shape=filter_width), tf_price ], axis=0)
  filt_price = tf.concat([ tf_price,tf.constant(tf_price[-1].numpy(), shape=filter_width) ], axis=0).numpy()
  price_centered = price - filt_price

  price_fouried = tf.signal.fft(price_centered)
  T = 1/24 # sampling interval in days
  N = price_fouried.shape[0]
  frequencies = np.linspace(0, 1 / T, N)
  fig, axes = plt.subplots(2, figsize=(12, 8))
  axes[0].plot(frequencies[:N // 2], tf.abs(price_fouried)[:N // 2] * 1 / N)
  axes[0].set_title('FFT magnitude')
  axes[1].plot(frequencies[:N // 2], tf.math.angle(price_fouried)[:N // 2])
  axes[1].set_title('FFT phase')
  axes[1].set(xlabel='cycles per day', ylabel='amplitude')
  plt.show()
  plt.close()

def main():
  ticker = sys.argv[1]
  df = pd.read_csv('data_' + ticker + '.csv', sep=',', skiprows=[i for i in range(1,100)])
  temp = df['timestamp']
  df = df.apply(pd.to_numeric, errors='coerce')
  df['timestamp'] = temp
  print(ticker + " price data:")
  print(df)
  print('performing fourier transformation...\n')
  FTT(df)

if __name__ == "__main__":
  main()