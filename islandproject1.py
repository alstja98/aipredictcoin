import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
import pyupbit
import requests
from datetime import datetime

# slack으로 주기적으로 알림받기
def post_message(token, channel, text):
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel, "text": text}
    )
    print(response)


myToken = "xoxb-2213547764481-2201106863555-NAL74UBJh9IayfSLo0h1lVYH"
def hgb_ai_calculate(code, time):
    # 타깃데이터 설정
    eth = pyupbit.get_ohlcv(code, interval=time, count=30000)
    target = eth['close'].shift(-1)
    target = target.dropna()
    target = target.to_numpy()
    target = np.delete(target, -1, 0)

    # 입력데이터 설정
    eth_1 = eth[['open', 'high', 'low', 'close', 'volume']].to_numpy()
    eth_1 = np.delete(eth_1, -1, 0)
    eth_1 = np.delete(eth_1, -1, 0)

    train_input, test_input, train_target, test_target = train_test_split(
        eth_1, target, test_size=0.2, random_state=42)

    # 히스토그램 그레이디언트 모델 적용
    hgb = HistGradientBoostingRegressor(random_state=42)
    scores = cross_validate(hgb, train_input, train_target,
                            return_train_score=True, n_jobs=-1)
    hgb.fit(train_input, train_target)

    # 예측 입력 데이터
    eth_2 = pyupbit.get_ohlcv(code, interval=time, count=2)
    eth_2 = eth_2[['open', 'high', 'low', 'close', 'volume']].to_numpy()
    eth_2 = np.delete(eth_2, -1, 0)

    # 예측하기
    prd_price = hgb.predict(eth_2)
    prd_price = np.round(prd_price)

    # 실제 가격
    eth_3 = pyupbit.get_ohlcv(code, interval=time, count=1)
    cur_price = eth_3[['close']].to_numpy()
    cur_price = np.round(cur_price)
    
    cross_score = np.mean(scores['train_score']), np.mean(scores['test_score'])
    cross_score = np.round(cross_score, 3)
    test_score = hgb.score(test_input, test_target)
    test_score = np.round(test_score, 3)

    strbuf = "종목명 :" + code + "\n" + datetime.now().strftime('[%m/%d %H:%M:%S] ') + "\nAI 예측 가격 :" + str(prd_price) + "\n실제 현재 가격 :" + str(cur_price) + "\n교차검증 확률 :" + str(cross_score) + "\n테스트 확률 :" + str(test_score)
    post_message(myToken,"#ai-predict", strbuf)

try:
    clock = 0
    while clock < 24:
        t_now = datetime.now()
        t_var = t_now.replace(hour=clock, minute=59)
        if t_var == t_now:
            symbol_list = ["KRW-BTC","KRW-ETH","KRW-XRP"]
            for sym in symbol_list:
                hgb_ai_calculate(sym, "minute60")
        else:
            clock = clock + 1
        if clock == 24:
            clock = 0

except Exception as ex:
    print(str(ex) +"오류발생")
