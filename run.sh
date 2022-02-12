#!/bin/sh
PATH=/usr/local/bin:/usr/local/sbin:~/bin:/usr/bin:/bin:/usr/sbin:/sbin

TICKER=$1
FREQ=$2
CRON=$3
echo "downloading $TICKER price data (every ${FREQ}hr) from Binance..."
node crypto.js $TICKER $FREQ
echo "running models..."
python3 crypto.py $TICKER $CRON
