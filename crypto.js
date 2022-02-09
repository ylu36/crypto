let axios = require('axios').default;
const fs = require('fs')

// https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data
const getBinanceData = async(ticker) => {
  let res = await axios.get(`https://binance.com/api/v3/klines?symbol=${ticker}USDT&interval=8h`);
  let json = {};
  if(res.status != 200) {
    console.error('bad response from Binance api');
    return;
  }
  const message = fs.createWriteStream(`data_${ticker}.csv`);
  message.write(`open,high,low,close,volume` + '\n');
  
  for (let i = 0; i <res.data.length; i++){
    let block = res.data[i];
        message.write(i + ',' + block[1] + ',' + block[2] + ',' +block[3] + ',' + block[4] + ',' + block[5]+ '\n');
  }
  message.close();
}

getBinanceData(process.argv[2].toUpperCase());
