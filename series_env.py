from gym import Env, logger
from gym.spaces import Box, Discrete

import pandas as pd
import numpy as np
import os
import os.path as osp
import random


class PnlSnapshot:
    def __init__(self, ):
        self.m_net_position = 0
        self.m_avg_open_price = 0
        self.m_net_investment = 0
        self.m_realized_pnl = 0
        self.m_unrealized_pnl = 0
        self.m_total_pnl = 0

    # buy_or_sell: 1 is buy, 2 is sell
    def update_by_tradefeed(self, buy_or_sell, traded_price, traded_quantity):
        assert (buy_or_sell == 1 or buy_or_sell == 2)

        if buy_or_sell == 2 and (traded_quantity > self.m_net_position):
            # Do nothing to stop us from shorting the stock
            return

        if buy_or_sell == 1 and self.m_net_position >= 1:
            return


        if buy_or_sell == 2 and self.m_net_position <= -1:
            return

        # buy: positive position, sell: negative position
        quantity_with_direction = traded_quantity if buy_or_sell == 1 else (-1) * traded_quantity
        is_still_open = (self.m_net_position * quantity_with_direction) >= 0

        # net investment
        self.m_net_investment = max( self.m_net_investment, abs( self.m_net_position * self.m_avg_open_price  ) )
        # realized pnl
        if not is_still_open:
            # Remember to keep the sign as the net position
            self.m_realized_pnl += ( traded_price - self.m_avg_open_price ) * \
                min(
                    abs(quantity_with_direction),
                    abs(self.m_net_position)
                ) * ( abs(self.m_net_position) / self.m_net_position )
        # total pnl
        self.m_total_pnl = self.m_realized_pnl + self.m_unrealized_pnl
        # avg open price
        if is_still_open:
            self.m_avg_open_price = ( ( self.m_avg_open_price * self.m_net_position ) +
                ( traded_price * quantity_with_direction ) ) / ( self.m_net_position + quantity_with_direction )
        else:
            # Check if it is close-and-open
            if traded_quantity > abs(self.m_net_position):
                self.m_avg_open_price = traded_price
        # net position
        self.m_net_position += quantity_with_direction

    def update_by_marketdata(self, last_price):
        self.m_unrealized_pnl = ( last_price - self.m_avg_open_price ) * self.m_net_position
        self.m_total_pnl = self.m_realized_pnl + self.m_unrealized_pnl






class SeriesEnv(Env):
    def __init__(self):
        base_path = '/home/andy/Documents/quant/data/PRICING/minute/'
        symb = 'AAPL'
        window_length = 32
        self.calc_pct = True

        self.observation_space = Box(low=0, high=1, shape=(window_length,5),
                dtype=np.float32)
        # have three options. Buy, sell or hold
        self.action_space = Discrete(3)

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
        self.closes = df[['open','high', 'low', 'close', 'volume']]
        self.closes = self.closes[self.closes.index > '2018-06-01']

        self.cur_i = 0

        self.rand_seed()



    def render(self, mode='human'):
        raise NotImplemented('No render function')


    def rand_seed(self):
        self.seed(random.randint(0, len(self.closes)))

    def seed(self, s):
        self.cur_i = s % len(self.closes)
        use_date = self.closes.index[self.cur_i].date()

        scan_i = self.cur_i
        while scan_i >= 0:
            if use_date != self.closes.index[scan_i].date():
                break

            scan_i -= 1

        self.cur_i = scan_i + 1

    def get_net_pos(self):
        return self.pnl.m_net_position

    def get_cur_date(self):
        return self.closes.index[self.cur_i]

    def __get_obs(self):
        if (self.cur_i + self.window_length) >= len(self.closes):
            self.cur_i = 0

        obs = self.closes.iloc[self.cur_i:self.cur_i + self.window_length]

        start_date = obs.index[0].date()
        end_date = obs.index[-1].date()

        if start_date != end_date:
            done = True
            self.cur_i = self.cur_i + self.window_length
        else:
            done = False

        self.cur_i += 1

        if self.calc_pct:
            obs = obs.pct_change().fillna(method='bfill')

        # Third index is the close.
        return obs.values, obs.iloc[-1]['close'], done


    def step(self, action):
        obs, last_price, done = self.__get_obs()
        if done:
            self.pnl = None
            return obs, 0.0, done, {}

        reward = 0
        done = False

        self.pnl.update_by_marketdata(last_price)

        if action != 0:
            self.pnl.update_by_tradefeed(action, last_price, 1)

        reward = self.pnl.m_total_pnl

        return obs, reward, done, {}


    def reset(self):
        obs = self.__get_obs()[0]
        self.cur_i -= 1
        self.pnl = PnlSnapshot()

        return obs


