#!/bin/sh
TICKER=$1
FREQ=$2
CRON=$3
time=`date`
python=/opt/homebrew/bin/python3 # use homebrew python instead of system-default
echo "downloading $TICKER price data (every ${FREQ}hr) from Binance..."
/usr/local/bin/node crypto.js $TICKER $FREQ
echo "running models..."
$python crypto.py $TICKER $CRON
printf "finished on ${time} !\n\n\n"