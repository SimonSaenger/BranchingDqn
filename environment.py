import collections
import math

import gym
from gym import spaces
import numpy as np
import torch
from dataset import Dataset
import utils


class Environment(gym.Env):

    def __init__(self, dataframe):
        self.config = utils.Config()
        self.dataframe = dataframe
        self.current_index = 0
        self.INITIAL_CASH = self.config.initial_cash
        self.cash = self.INITIAL_CASH
        self.portfolio_value = 0
        self.stock = 0
        self.states = self.dataframe.loc[:, ~dataframe.columns.isin([f'{self.config.aktie}, {self.config.aktie_price}'])].to_numpy()
        self.rewards = self.dataframe[self.config.aktie].to_numpy()
        self.n = len(self.states)
        self.bins = self.config.bins
        self.low = self.config.low
        self.high = self.config.high
        self.discretized = np.linspace(self.low, self.high, self.bins).astype(int)

        self.action_space = spaces.Box(
            low=self.low,
            high=self.high, shape=(1,),
            dtype=int
        )
        self.observation_space = spaces.Box(
            low=0,
            high=np.nan,
            shape=(self.states.shape[1],),
            dtype=np.float32
        )

    def reset(self):
        self.cash = self.INITIAL_CASH
        self.portfolio_value = 0
        self.stock = 0
        self.current_index = 0
        next_state = self.states[self.current_index]
        next_state = np.array(next_state).reshape(1, -1)
        next_state = torch.tensor(next_state).float()
        return next_state

    def step(self, action: collections.Iterable):

        if self.current_index >= self.n:
            raise Exception("Episode already done")

        action = np.array([self.discretized[aa] for aa in action])
        action = action.ravel()[0]
        current_price = self.dataframe.iloc[self.current_index][self.config.aktie_price]

        past_trend = 0
        if self.current_index > 5:
            past_prices = np.array(self.dataframe.iloc[self.current_index-5:self.current_index][self.config.aktie_price], dtype=float)
            past_trend = sum(np.gradient(past_prices))

        buy_max = int(self.cash / current_price)
        if self.current_index == 0 and self.config.buy_and_hold:
            env_action = buy_max
            self.cash -= env_action * current_price
        elif self.current_index == self.n - 1 and self.config.buy_and_hold:
            env_action = - self.stock
            self.cash += - env_action * current_price
        else:
            if action > 0:
                env_action = np.min([buy_max, action])
                self.cash -= env_action * current_price
            else:
                env_action = - np.min([self.stock, np.abs(action)])
                self.cash += - env_action * current_price

        self.stock += env_action
        self.portfolio_value = self.stock * current_price

        multiplier = math.copysign(1, action) * past_trend
        if past_trend > 0 or past_trend < 0:
            reward = multiplier * env_action * np.exp(self.rewards[self.current_index] + 1e-9) + self.cash + self.portfolio_value
        else:
            reward = env_action * np.exp(self.rewards[self.current_index] + 1e-9) + self.cash + self.portfolio_value

        done = (self.current_index == self.n - 1)
        self.current_index += 1
        if not done:

            if action is None:
                raise Exception("NaNs detected!")

            next_state = self.states[self.current_index]
            next_state = np.array(next_state).reshape(1, -1)
            next_state = torch.tensor(next_state).reshape(1, -1).float()
        else:
            next_state = None

        return next_state, reward, done
