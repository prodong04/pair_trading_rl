# utils.py

import numpy as np
from scipy.signal import find_peaks

def find_important_extremes(series, C=2):
    s = series.std()
    threshold = C * s

    # 국소 최대값 찾기
    peaks, _ = find_peaks(series)
    # 국소 최소값 찾기
    troughs, _ = find_peaks(-series)

    important_maxima = []
    important_minima = []

    # 중요한 최대값 필터링
    for idx in peaks:
        left_min = series[:idx].min() if idx > 0 else series.iloc[0]
        right_min = series[idx+1:].min() if idx < len(series)-1 else series.iloc[-1]
        if (series.iloc[idx] - left_min >= threshold) and (series.iloc[idx] - right_min >= threshold):
            important_maxima.append(idx)

    # 중요한 최소값 필터링
    for idx in troughs:
        left_max = series[:idx].max() if idx > 0 else series.iloc[0]
        right_max = series[idx+1:].max() if idx < len(series)-1 else series.iloc[-1]
        if (left_max - series.iloc[idx] >= threshold) and (right_max - series.iloc[idx] >= threshold):
            important_minima.append(idx)

    return important_maxima, important_minima

def calculate_empirical_mean_reversion_time(series, C=2):
    important_maxima, important_minima = find_important_extremes(series, C)
    extremes = sorted(important_maxima + important_minima)
    mean = series.mean()
    reversion_times = []

    for idx in extremes:
        current_value = series.iloc[idx]
        for j in range(idx+1, len(series)):
            if (current_value > mean and series.iloc[j] <= mean) or (current_value < mean and series.iloc[j] >= mean):
                reversion_time = j - idx
                reversion_times.append(reversion_time)
                break
    if len(reversion_times) > 0:
        return np.mean(reversion_times)
    else:
        return np.inf  # 복귀 시간이 없을 경우

def get_state(series, t, l=4, k=3):
    state = []
    for i in range(t - l + 1, t + 1):
        if i <= 0:
            state.append(0)  # 초기 기간 패딩
            continue
        Pi = series.iloc[i]
        Pi_prev = series.iloc[i - 1]
        pi = (Pi - Pi_prev) / Pi_prev * 100
        if pi > k:
            di = +2
        elif 0 < pi <= k:
            di = +1
        elif -k < pi <= 0:
            di = -1
        else:  # pi <= -k
            di = -2
        state.append(di)
    return tuple(state)

def get_reward(position, action, spread_price, mean_price, c=0.001):
    # 포지션과 행동에 따른 보상 계산
    # position: 현재 포지션 (1: 롱, 0: 현금, -1: 숏)
    # action: 수행한 행동 (1: 매수, 0: 유지, -1: 매도)
    # 스프레드 가격이 평균보다 낮으면 롱 포지션에 유리, 높으면 숏 포지션에 유리
    if position == 0:
        # 현금 상태에서
        if action == 1:
            # 롱 포지션 진입
            reward = (mean_price - spread_price) - c
        elif action == -1:
            # 숏 포지션 진입
            reward = (spread_price - mean_price) - c
        else:
            reward = 0  # 유지
    elif position == 1:
        # 롱 포지션에서
        if action == -1:
            # 롱 포지션 청산
            reward = (spread_price - mean_price) - c
        else:
            reward = 0  # 유지
    elif position == -1:
        # 숏 포지션에서
        if action == 1:
            # 숏 포지션 청산
            reward = (mean_price - spread_price) - c
        else:
            reward = 0  # 유지
    else:
        reward = 0  # 예외적인 경우

    return reward
