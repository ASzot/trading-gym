from gym import Env, logger
from gym.spaces import Box, Discrete

import pandas as pd
import numpy as np
import os
import os.path as osp
import random



class SeriesEnv(Env):
    def __init__(self):
        base_path = '/home/andy/Documents/quant/data/PRICING/minute/'
        symb = 'AAPL'
        window_length = 30

        self.observation_space = Box(low=0, high=1, shape=(window_length,),
                dtype=np.float32)
        # have three options. Buy, sell or hold
        self.action_space = Discrete(2)

        self.window_length = window_length

        df = pd.read_csv(osp.join(base_path, symb + '.csv'),
                            parse_dates=[0],
                            infer_datetime_format=False,
                            index_col=0)

        def filter_func(x):
            hour = x.hour
            minute = x.minute
            if hour == 9 and minute < 31:
                return True
            elif hour < 9:
                return True
            elif hour > 16:
                return True
            elif hour == 16 and minute > 0:
                return True
            return False

        remove_indices = [filter_func(x) for x in df.index]

        df = df.drop(df.index[remove_indices])

        # We only care about closes.
        self.closes = df['close']

        self.cur_i = 0

        self.reset()

    def render(self, mode='human'):
        raise NotImplemented('No render function')


    def rand_seed(self):
        self.seed(random.randint(0, len(self.closes)))

    def seed(self, s):
        self.cur_i = s % len(self.closes)

    def get_cur_date(self):
        return self.closes.index[self.cur_i]

    def __get_obs(self):
        if (self.cur_i + self.window_length) >= len(self.closes):
            self.cur_i = 0

        obs = self.closes[self.cur_i:self.cur_i + self.window_length]

        days = [dt.day for dt in obs.index]
        start_day = days[-1]
        break_i = None
        for i, day in enumerate(reversed(days)):
            if day != start_day:
                break_i = len(days) - i
                break

        if break_i is not None:
            eod_price = self.closes[break_i - 1]

            use_index = break_i + self.cur_i
            obs = self.closes[use_index : use_index + self.window_length]
            self.cur_i = use_index
        else:
            eod_price = None
            self.cur_i += 1

        pct_diffs = obs.pct_change().fillna(method='bfill')

        return pct_diffs.values, eod_price, obs[-1]


    def step(self, action):
        obs, eod_price, last_price = self.__get_obs()

        # do nothing: 0
        # buy/sell: 1

        reward = 0
        done = False

        if eod_price is not None and self.is_holding:
            price_diff = eod_price - self.bought_price
            reward = price_diff
            done = True
            self.is_holding = False
        else:
            if action == 1:
                if self.is_holding:
                    price_diff = last_price - self.bought_price
                    reward = price_diff
                    done = True
                    self.is_holding = False
                else:
                    self.is_holding = True
                    self.bought_price = last_price

        return obs, reward, done, {}


    def reset(self):
        self.is_holding = False
        self.bought_price = None
        return self.__get_obs()[0]


