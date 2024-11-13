import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from numba import jit

# Ornstein-Uhlenbeck Spread Calculation
@jit(nopython=True)
def calculate_ou_spread_coefficients(train_data, a2_values, threshold=2):
    best_a2, min_reversion_time = None, np.inf
    for a2 in a2_values:
        spread = train_data[:, 0] + a2 * train_data[:, 1]  # Assuming first column is Close_avax, second is Close_ssv
        reversion_time = calculate_empirical_mean_reversion_time(spread, threshold)
        if reversion_time < min_reversion_time:
            min_reversion_time = reversion_time
            best_a2 = a2
    return best_a2

@jit(nopython=True)
def calculate_empirical_mean_reversion_time(spread, threshold):
    mean_value = np.mean(spread)
    local_extrema = []
    for i in range(1, len(spread) - 1):
        if (spread[i] < spread[i - 1] and spread[i] < spread[i + 1] and abs(spread[i] - mean_value) > threshold):
            local_extrema.append(i)
        elif (spread[i] > spread[i - 1] and spread[i] > spread[i + 1] and abs(spread[i] - mean_value) > threshold):
            local_extrema.append(i)
    if len(local_extrema) < 2:
        return np.inf
    return np.mean(np.diff(local_extrema))

# Load and prepare data
def load_data(file1, file2):
    df_avax = pd.read_csv(file1, parse_dates=['Open time'])
    df_ssv = pd.read_csv(file2, parse_dates=['Open time'])
    df_avax.set_index('Open time', inplace=True)
    df_ssv.set_index('Open time', inplace=True)
    df = pd.merge(df_avax[['Close']], df_ssv[['Close']], left_index=True, right_index=True, how='inner', suffixes=('_avax', '_ssv')).dropna()
    return df

# Custom Mean Reversion Trading Environment using PPO and OU Spread
class MeanReversionEnv(gym.Env):
    def __init__(self, spread, transaction_cost=0.001):
        super(MeanReversionEnv, self).__init__()
        self.spread = spread
        self.transaction_cost = transaction_cost
        self.current_step = 0
        self.position = 0  # 0: no position, 1: long, -1: short
        self.cash = 10000
        self.inventory = 0

        # Define the action and observation space
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: long, 2: short
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.cash = 10000
        self.inventory = 0
        return self._get_observation()

    def step(self, action):
        prev_position = self.position
        spread_price = self.spread[self.current_step]
        mean_price = np.mean(self.spread[:self.current_step + 1]) if self.current_step > 0 else spread_price

        # Calculate reward and perform actions
        reward = 0
        if action == 1:  # Buy (long)
            if self.position == 0:
                self.inventory = self.cash / spread_price
                self.cash -= spread_price * self.inventory * (1 + self.transaction_cost)
                self.position = 1
            elif self.position == -1:
                reward = (self.inventory * (mean_price - spread_price)) - (self.transaction_cost * self.cash)
                self.cash += self.inventory * spread_price * (1 - self.transaction_cost)
                self.inventory = 0
                self.position = 0
        elif action == 2:  # Sell (short)
            if self.position == 0:
                self.inventory = self.cash / spread_price
                self.cash -= self.transaction_cost * self.cash
                self.position = -1
            elif self.position == 1:
                reward = (self.inventory * (spread_price - mean_price)) - (self.transaction_cost * self.cash)
                self.cash += self.inventory * spread_price * (1 - self.transaction_cost)
                self.inventory = 0
                self.position = 0

        # Update cash and inventory based on position
        portfolio_value = self.cash + (self.inventory * spread_price if self.position == 1 else 0)
        self.current_step += 1

        done = self.current_step >= len(self.spread) - 1
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        window = 4
        if self.current_step < window:
            return np.zeros(window)
        return self.spread[self.current_step - window:self.current_step]

# Load data and calculate OU spread
df = load_data('AVAXUSDC.csv', 'SSVUSDT.csv')
df_train, df_test = np.split(df[['Close_avax', 'Close_ssv']].values, [int(0.7 * len(df))])
a2_values = np.linspace(-3.0, 3.0, 601)
best_a2 = calculate_ou_spread_coefficients(df_train, a2_values)
spread_train = df_train[:, 0] + best_a2 * df_train[:, 1]

# Instantiate and train PPO on the custom environment
env = MeanReversionEnv(spread_train)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Test the trained model
obs = env.reset()
for _ in range(len(spread_train) - 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        break

print("Training and testing completed with PPO on OU Process Spread.")
