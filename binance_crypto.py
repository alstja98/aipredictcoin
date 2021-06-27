# 메인 알고리즘
import ccxt
import time
import pandas as pd

api_key = "LdbYr06exvcLrYmG2zjnxA5xIUCNZcvKgLTfZ6xz0g4eDfyubuNUL8jAccXTTeMl"
secret = "qbva91lDnWGzgD4yL9xhdWTqZHObEh9ZRPky3nfBTXbP0XXwo6FQlzBED1i8SRrf"

binance = ccxt.binance(config={
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})

bought_list = []

# 시세 조회 함수


def get_ohlcv(symbol):
  btc = binance.fetch_ohlcv(
      symbol=symbol,
      timeframe='5m',
      since=None,
      limit=300)

  df = pd.DataFrame(data=btc, columns=[
                    'datetime', 'open', 'high', 'low', 'close', 'volume'])
  df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
  df.set_index('datetime', inplace=True)
  return df

# 현재가 조회 함수


def current_price(symbol):
  price = binance.fetch_ticker(symbol)
  print(price['last'])

# 바로 전 종가 출력


def before_close(symbol):
  df = get_ohlcv(symbol)
  close = df['close'].iloc[-2]
  return close

# 단순 이동평균선 제작 함수


def get_sma(symbol, count):
  df = get_ohlcv(symbol)
  ma = df['close'].rolling(count).mean()
  ma.dropna()
  return ma

# 지수 이동평균선 제작 함수


def get_ema(symbol, count):
  df = get_ohlcv(symbol)
  ma = df['close'].ewm(count).mean()
  ma.dropna()
  return ma

# macd 값 구하기


def get_macd(symbol):
  df = get_ohlcv(symbol)
  macd = df['close'].ewm(span=12, min_periods=11, adjust=False).mean(
  ) - df['close'].ewm(span=26, min_periods=25, adjust=False).mean()
  macd_signal = macd.ewm(span=9, min_periods=8, adjust=False).mean()
  macdhist = macd - macd_signal
  return macdhist

# 롱 포지션 매수 함수


def long_position(symbol, leverage, amount):
  markets = binance.load_markets()
  market = binance.market(symbol)

  resp = binance.fapiPrivate_post_leverage({
      'symbol': market['id'],
      'leverage': leverage
  })

  if symbol in bought_list:
    return False
  else:
    order = binance.create_market_buy_order(
        symbol=symbol,
        amount=amount,
    )
    bought_list.append(symbol)

# 숏 포지션 매도 함수


def short_position(symbol, leverage, amount):
  markets = binance.load_markets()
  market = binance.market(symbol)

  resp = binance.fapiPrivate_post_leverage({
      'symbol': market['id'],
      'leverage': leverage
  })

  if symbol in bought_list:
    return False
  else:
    order = binance.create_market_sell_order(
        symbol=symbol,
        amount=amount,
    )
    bought_list.append(symbol)

# 포지션 정리


def exit_position(symbol, position):
  amount = position['amount']
  if position['type'] == 'long':
    binance.create_market_sell_order(symbol=symbol, amount=amount)
    position['type'] = None
  elif position['type'] == 'short':
    binance.create_market_buy_order(symbol=symbol, amount=amount)
    position['type'] = None


try:
  while True:
    current_price = current_price("BTC/USDT")
    ma5 = get_ema("BTC/USDT", 5).iloc[-1]
    ma10 = get_ema("BTC/USDT", 10).iloc[-1]
    ma12 = get_ema("BTC/USDT", 12).iloc[-1]
    ma20 = get_ema("BTC/USDT", 20).iloc[-1]
    macdhist = get_macd("BTC/USDT")
    bc = before_close("BTC/USDT")

    if (bc > ma5) and (bc > ma10) and (bc > ma20) and (macdhist.iloc[-1] > 0) and (macdhist.iloc[-2] < macdhist.iloc[-1]):
      long_position("BTC/USDT", 1, 0.001)
      if bc < ma12:
        exit_position("BTC/USDT", 0.001)
    elif bc < ma5 and bc < ma10 and bc < ma20 and macdhist.iloc[-1] < 0 and macdhist.iloc[-2] < macdhist.iloc[-1]:
      short_position("BTC/USDT", 1, 0.001)
      if bc > ma12:
        exit_position("BTC/USDT", 0.001)

except Exception as ex:
  print(str(ex))
