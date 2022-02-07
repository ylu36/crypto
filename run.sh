TICKER=$1
echo "downloading $TICKER price data from Binance..."
node crypto.js $TICKER
echo "running models..."
python3 crypto.py $TICKER
