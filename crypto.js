let axios = require('axios').default;
const fs = require('fs')

// https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data
const getBinanceData = async(ticker, interval) => {
  interval = interval || 8;
  let res = await axios.get(`https://binance.us/api/v3/klines?symbol=${ticker}USDT&interval=${interval}&limit=1000`);
  let json = {};
  if(res.status != 200) {
    console.error('bad response from Binance api');
    return;
  }
  const message = fs.createWriteStream(`data_${ticker}.csv`);
  message.write(`timestamp,open,high,low,close,volume` + '\n');
  
  for (let i = 0; i <res.data.length; i++){
    let block = res.data[i];
    let time = new Date(block[0]).toString().slice(4, 21);
    message.write(i + ',' + time + ',' + block[1] + ',' + block[2] + ',' +block[3] + ',' + block[4] + ',' + block[5]+ '\n');
  }
  message.close();
}

getBinanceData(process.argv[2].toUpperCase(), process.argv[3]);
