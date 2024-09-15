import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

class StockEnv(gym.Env):
    def __init__(self, dataframe, **kwargs) -> None:
        self.dataframe = dataframe
        self.terminal = False 

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.data = dataframe.loc[self.day, :]
        self.nb_stock = 1

        self.action_space = spaces.Box(-1, 1, (self.nb_stock,))
        self.observation_space = spaces.Box(0, np.inf, (2 * self.nb_stock + 1,))

        self.state = [self.initial_balance] \
            + [self.data.close] \
            + [0] * self.nb_stock

        self.reward = 0
        self.cost = 0
        # self.turbulence = 0
        self.nb_trades = 0

        self.asset_memory = [self.initial_balance]
        self.reward_mem = []

    def _execute_action(self, actions):
        actions = np.clip(actions, -1, 1) * self.hmax
        actions = (actions.astype(int))

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        # Buy stocks
        for index in buy_index:
            if actions[index] > 0.1:
                if actions[index] >= 0.5:
                    percent = 0.5
                elif actions[index] >= 0.3:
                    percent = 0.3
                else:
                    percent = 0.2
                    
                available_amount = int(self.state[0] // self.state[index + 1] * percent)
                # amount = min(available_amount, actions[index])
                amount = available_amount
                self.state[0] -= self.state[index + 1] * amount * (1 + self.transaction_fee)
                self.state[index + self.nb_stock + 1] += amount
                self.cost += self.state[index + 1] * amount * self.transaction_fee
                self.trades += 1

        # Sell stocks
        for index in sell_index:
            if actions[index] < -0.1:
                if actions[index] <= -0.5:
                    percent = 0.5
                elif actions[index] <= 0.3:
                    percent = 0.3
                else:
                    percent = 0.2

                # amount = min(abs(actions[index]), self.state[index + self.nb_stock + 1])
                amount = int(self.state[index + self.nb_stock + 1] * percent)
                self.state[0] += self.state[index + 1] * amount * (1 - self.transaction_fee)
                self.state[index + self.nb_stock + 1] -= amount
                self.cost += self.state[index + 1] * amount * self.transaction_fee
                self.trades += 1


    def step(self, actions):
        self.terminal = self.day >= len(self.dataframe.index.unique()) - 1
        if self.terminal:
            pass

        else:
            begin_total_asset = self.state[0]+ \
                                sum(np.array(self.state[1:(self.nb_stock  + 1)]) * np.array(self.state[(self.nb_stock  + 1):(self.nb_stock  * 2 + 1)]))

            self._execute_action(actions)

            self.day += 1
            self.data = self.dataframe.loc[self.day,:]
            self.state = [self.state[0]] + \
                         [self.data.close] + \
                         list(self.state[(self.nb_stock  + 1):(self.nb_stock  * 2 + 1)])
            

            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(self.nb_stock  + 1)]) * np.array(self.state[(self.nb_stock  + 1):(self.nb_stock  * 2 + 1)]))
            self.asset_memory.append(end_total_asset)

            # self.turbulence = self.data['turbulence'].values[0]

            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, False, {}
    
    def reset(self, seed=None, options=None,):
        self.asset_memory = [self.initial_balance]
        self.day = 0
        self.data = self.dataframe.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []

        self.state = [self.initial_balance] \
            + [self.data.close] \
            + [0] * self.nb_stock
        
        return self.state, {}
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]