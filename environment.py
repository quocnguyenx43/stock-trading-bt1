import torch
from gym import spaces
import numpy as np
from random import random

MAX_ACCOUNT_BALANCE = 2_147_483_647
MAX_NUM_SHARES = 2_147_483_647
MAX_SHARE_PRICE = 5_000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20_000
INITIAL_ACCOUNT_BALANCE = 10_000


class StockTradingEnv():

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.LOOKBACK_WINDOW_SIZE = 40

        df['open'] = df['open'] / df['open'].max()
        df['high'] = df['high'] / df['high'].max()
        df['low'] = df['low'] / df['low'].max()
        df['close'] = df['close'] / df['close'].max()

        self.df = df

        # buy x%, sell x%, hold, etc. observation_space is 40 last days and current day
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([7]), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, self.LOOKBACK_WINDOW_SIZE + 1), dtype=np.float16)

    
    def step(self, action):
        current_price = self.df.loc[self.current_step, "close"]

        reward = 0
        done = False
        info = {}

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            prev_cost = self.cost_basis * self.shares_held
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

            if shares_bought > 0:
                self.trades.append({'date': self.df['time'].values[self.current_step],
                                    'step': self.current_step,
                                    'shares': shares_bought,
                                    'total': additional_cost,
                                    'type': 'buy'})
                
            # print('buy', amount, ' - shares: ', shares_bought, ' - price: ', current_price, ' - total: ', additional_cost, end=' - ')

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            sold_total = shares_sold * current_price
            self.balance += sold_total
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

            if shares_sold > 0:
                self.trades.append({'date': self.df['time'].values[self.current_step],
                                    'step': self.current_step,
                                    'shares': shares_sold,
                                    'total': shares_sold * current_price,
                                    'type': 'sell'})
                
            # change reward
            prev_bought_cost = self.cost_basis * shares_sold
            reward = sold_total - prev_bought_cost

            # print('sell', amount, ' - shares: ', shares_sold, ' - price: ', current_price, ' - total: ', sold_total, end=' - ')

        else:
            self.trades.append({'date': self.df['time'].values[self.current_step],
                                'step': self.current_step,
                                'shares': 0,
                                'total': 0,
                                'type': 'hold'})
            

        self.net_worth = self.balance + self.shares_held * current_price
        self.profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0
        

        n_state = self._next_observation()
        done = True if n_state is None or self.net_worth < 0 else False
        reward = 0 if done else reward
        
        return n_state, reward, done, info
    
    
    def _next_observation(self):
        if self.current_step >= len(self.df) - 1:
            return None 
        previous_prices = self.df['close'].values[
            self.current_step - self.LOOKBACK_WINDOW_SIZE: self.current_step]
        today_price = self.df['close'].values[self.current_step]
        # date = self.df['time'].values[self.current_step]

        state = np.concatenate((previous_prices, [today_price]))
                            # self.balance, self.net_worth,
                            # self.current_step, self.shares_held)))
        self.current_step += 1

        return torch.tensor(state)
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE

        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = self.LOOKBACK_WINDOW_SIZE 
        self.trades = []

        return self._next_observation()

    