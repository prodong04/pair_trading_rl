import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import datetime
import pickle  # 파라미터 저장을 위한 모듈
from tqdm import tqdm  # 진행률 표시 라이브러리

# 유틸리티 함수 임포트
from utils import find_important_extremes, calculate_empirical_mean_reversion_time, get_reward, get_state

# 데이터 로드 및 전처리
# CSV 파일에서 데이터를 읽어옵니다.
df_avax = pd.read_csv('AVAXUSDC.csv', parse_dates=['Open time'])
df_ssv = pd.read_csv('SSVUSDT.csv', parse_dates=['Open time'])

# 날짜를 인덱스로 설정합니다.
df_avax.set_index('Open time', inplace=True)
df_ssv.set_index('Open time', inplace=True)

# 두 데이터프레임을 공통 날짜 기준으로 병합합니다.
df = pd.merge(df_avax[['Close']], df_ssv[['Close']], left_index=True, right_index=True, how='inner', suffixes=('_avax', '_ssv'))

# 결측치를 제거합니다.
df.dropna(inplace=True)

# 데이터 확인
print("데이터 헤드:")
print(df.head())

# 데이터셋 분할 (학습:테스트 = 7:3)
train_size = int(len(df) * 0.7)  # 전체 데이터의 70%를 학습 데이터로 사용
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

print(f"\n전체 데이터 수: {len(df)}")
print(f"학습 데이터 수: {len(df_train)}")
print(f"테스트 데이터 수: {len(df_test)}")

# 학습 데이터로 스프레드 계산 함수 정의
def calculate_spread_train(a2):
    a1 = 1  # 첫 번째 자산의 계수는 1로 고정
    spread = a1 * df_train['Close_avax'] + a2 * df_train['Close_ssv']
    return spread

# 최적의 a2 값을 찾기 위한 그리드 탐색 (학습 데이터 사용)
a2_values = np.arange(-3.0, 3.0, 0.01)
best_a2 = None
min_reversion_time = np.inf

print("\n[1단계] 학습 데이터로 최적의 계수 a2를 찾는 중...")
for a2 in tqdm(a2_values, desc="최적의 a2 탐색"):
    spread = calculate_spread_train(a2)
    # 경험적 평균 복귀 시간 계산
    reversion_time = calculate_empirical_mean_reversion_time(spread, C=2)
    # 최소 복귀 시간을 업데이트합니다.
    if reversion_time < min_reversion_time:
        min_reversion_time = reversion_time
        best_a2 = a2
        #print(f"새로운 최적의 a2 발견: {best_a2}, 최소 평균 복귀 시간: {min_reversion_time}")

print(f'\n최종 최적의 계수 a2: {best_a2}, 최소 평균 복귀 시간: {min_reversion_time}')

# 최적의 a2로 학습 데이터의 스프레드 계산
spread_train = calculate_spread_train(best_a2)

# 강화학습 파라미터 설정
alpha = 0.1       # 학습률
gamma = 0.99      # 할인율
epsilon = 0.1     # 탐험 확률
num_episodes = 10 # 에피소드 수

state_size = 4    # 상태 길이
k = 3             # 수익률 임계값 (%)
l = state_size    # 상태 길이 (과거 관찰 기간)
c = 0.001         # 거래 비용

# Q-테이블 초기화
Q = {}

# 강화학습 에이전트의 포지션:
# 1: 롱 포지션, 0: 현금 상태, -1: 숏 포지션
position = 0

print("\n[2단계] 강화학습 에이전트 훈련 시작...")
for episode in tqdm(range(num_episodes), desc="에피소드 진행"):
    #print(f"\n에피소드 {episode+1}/{num_episodes}")
    position = 0  # 에피소드 시작 시 포지션 초기화
    for t in range(l, len(spread_train) - 1):
        # 현재 상태를 가져옵니다.
        state = get_state(spread_train, t, l, k)
        mean_price = spread_train.mean()
        spread_price = spread_train.iloc[t]

        # 가능한 행동 결정
        if position == 0:
            possible_actions = [0, 1, -1]  # 현금 상태에서는 유지, 매수, 매도(숏 진입) 가능
        elif position == 1:
            possible_actions = [0, -1]     # 롱 포지션에서는 유지, 매도(롱 청산) 가능
        elif position == -1:
            possible_actions = [0, 1]      # 숏 포지션에서는 유지, 매수(숏 청산) 가능

        # Q-테이블에 상태가 없으면 초기화
        if state not in Q:
            Q[state] = {a: 0 for a in possible_actions}

        # epsilon-greedy 정책에 따라 행동 선택
        if np.random.rand() < epsilon:
            action = np.random.choice(possible_actions)
            #print(f"탐험 선택: 시간 {df_train.index[t]}, 상태 {state}, 행동 {action}")
        else:
            action = max(Q[state], key=Q[state].get)
            #print(f"활용 선택: 시간 {df_train.index[t]}, 상태 {state}, 행동 {action}")

        # 행동에 따른 보상 계산
        reward = get_reward(position, action, spread_price, mean_price, c)
        #print(f"보상 계산: 보상 {reward}")

        # 다음 상태로 전이
        next_position = position
        if position == 0:
            if action == 1:
                next_position = 1   # 롱 포지션 진입
            elif action == -1:
                next_position = -1  # 숏 포지션 진입
        elif position == 1:
            if action == -1:
                next_position = 0   # 롱 포지션 청산
        elif position == -1:
            if action == 1:
                next_position = 0   # 숏 포지션 청산

        next_state = get_state(spread_train, t+1, l, k)
        next_spread_price = spread_train.iloc[t+1]

        # 다음 가능한 행동 결정
        if next_position == 0:
            next_possible_actions = [0, 1, -1]
        elif next_position == 1:
            next_possible_actions = [0, -1]
        elif next_position == -1:
            next_possible_actions = [0, 1]

        # Q-테이블에 다음 상태가 없으면 초기화
        if next_state not in Q:
            Q[next_state] = {a: 0 for a in next_possible_actions}

        # 미래 보상 최대값 계산
        future_rewards = [Q[next_state][a] for a in next_possible_actions]
        max_future_reward = max(future_rewards)

        # Q-러닝 업데이트
        old_value = Q[state][action]
        Q[state][action] = old_value + alpha * (reward + gamma * max_future_reward - old_value)
        #print(f"Q-테이블 업데이트: 상태 {state}, 행동 {action}, Q값 {Q[state][action]}")

        # 포지션 업데이트
        position = next_position

print("\n강화학습 에이전트 훈련 완료!")

# 훈련된 Q-테이블 저장
with open('trained_q_table.pkl', 'wb') as f:
    pickle.dump(Q, f)
print("훈련된 Q-테이블을 'trained_q_table.pkl' 파일로 저장했습니다.")

# 최적의 a2 값 저장
with open('best_a2.pkl', 'wb') as f:
    pickle.dump(best_a2, f)
print(f"최적의 a2 값을 'best_a2.pkl' 파일로 저장했습니다.")

# 저장된 Q-테이블과 최적의 a2 값 로드
with open('trained_q_table.pkl', 'rb') as f:
    Q_loaded = pickle.load(f)
print("\n훈련된 Q-테이블을 로드했습니다.")

with open('best_a2.pkl', 'rb') as f:
    best_a2_loaded = pickle.load(f)
print(f"최적의 a2 값을 로드했습니다: {best_a2_loaded}")

# 테스트 데이터로 스프레드 계산 함수 정의
def calculate_spread_test(a2):
    a1 = 1  # 첫 번째 자산의 계수는 1로 고정
    spread = a1 * df_test['Close_avax'] + a2 * df_test['Close_ssv']
    return spread

# 로드된 최적의 a2로 테스트 데이터의 스프레드 계산
spread_test = calculate_spread_test(best_a2_loaded)

# 거래 시뮬레이션
initial_cash = 10000
cash = initial_cash
position = 0  # 0: 현금, 1: 롱 포지션, -1: 숏 포지션
inventory = 0
portfolio_values = []

# 거래 기록 저장을 위한 리스트
trade_history = []

print("\n[3단계] 테스트 데이터에서 거래 시뮬레이션 시작...")
for t in tqdm(range(l, len(spread_test) - 1), desc="거래 시뮬레이션 진행"):
    state = get_state(spread_test, t, l, k)
    mean_price = spread_test.mean()
    spread_price = spread_test.iloc[t]

    # 가능한 행동 결정
    if position == 0:
        possible_actions = [0, 1, -1]  # 현금 상태에서는 유지, 매수, 매도(숏 진입) 가능
    elif position == 1:
        possible_actions = [0, -1]     # 롱 포지션에서는 유지, 매도(롱 청산) 가능
    elif position == -1:
        possible_actions = [0, 1]      # 숏 포지션에서는 유지, 매수(숏 청산) 가능

    # 행동 선택 (탐험 없이)
    if state in Q_loaded:
        action = max(Q_loaded[state], key=Q_loaded[state].get)
    else:
        action = 0  # 상태가 없으면 유지
    #print(f"시간 {df_test.index[t]}: 상태 {state}, 행동 {action}")

    # 행동에 따른 거래 수행
    if position == 0:
        if action == 1:
            # 롱 포지션 진입 (매수)
            inventory = cash / spread_price
            transaction_cost = c * cash
            cash -= cash + transaction_cost
            position = 1
            trade_history.append((df_test.index[t], 'Long Entry', spread_price))
            #print(f"롱 포지션 진입: 시간 {df_test.index[t]}, 가격 {spread_price}, 남은 현금 {cash}")
        elif action == -1:
            # 숏 포지션 진입 (공매도)
            inventory = cash / spread_price
            transaction_cost = c * cash
            # 공매도로 인해 현금 변화는 없지만, 거래 비용은 발생한다고 가정
            cash -= transaction_cost
            position = -1
            trade_history.append((df_test.index[t], 'Short Entry', spread_price))
            #print(f"숏 포지션 진입: 시간 {df_test.index[t]}, 가격 {spread_price}, 남은 현금 {cash}")
    elif position == 1:
        if action == -1:
            # 롱 포지션 청산 (매도)
            transaction_cost = c * inventory * spread_price
            cash += inventory * spread_price - transaction_cost
            inventory = 0
            position = 0
            trade_history.append((df_test.index[t], 'Long Exit', spread_price))
            #print(f"롱 포지션 청산: 시간 {df_test.index[t]}, 가격 {spread_price}, 남은 현금 {cash}")
    elif position == -1:
        if action == 1:
            # 숏 포지션 청산 (매수)
            transaction_cost = c * inventory * spread_price
            # 숏 포지션 청산 시 비용 발생
            cash -= inventory * spread_price + transaction_cost
            inventory = 0
            position = 0
            trade_history.append((df_test.index[t], 'Short Exit', spread_price))
            #print(f"숏 포지션 청산: 시간 {df_test.index[t]}, 가격 {spread_price}, 남은 현금 {cash}")

    # 현재 포트폴리오 가치 계산
    if position == 1:
        portfolio_value = cash + inventory * spread_price
    elif position == -1:
        # 숏 포지션의 경우, 현재 포트폴리오 가치는 현금 + 숏 포지션의 이익/손실
        portfolio_value = cash + inventory * (mean_price - spread_price)
    else:
        portfolio_value = cash
    portfolio_values.append(portfolio_value)
    #print(f"포트폴리오 가치: {portfolio_value}")

print("\n거래 시뮬레이션 완료!")

# 결과 시각화
plt.figure(figsize=(12,6))
plt.plot(df_test.index[l:len(portfolio_values)+l], portfolio_values)
plt.title('포트폴리오 가치 변화 (테스트 데이터)')
plt.xlabel('날짜')
plt.ylabel('포트폴리오 가치')
plt.grid(True)
plt.show()

# 거래 내역 출력
print("\n거래 내역:")
for trade in trade_history:
    print(f"{trade[0].date()} - {trade[1]} at price {trade[2]:.2f}")
