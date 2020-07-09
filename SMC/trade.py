import random
import warnings
from datetime import datetime, date
from .chart import trade_to_cropped_pic
from .datareader import datepreprocess
import numpy as np
import pandas as pd

class TradeEnv:

    def __init__(self, input_df, frame_length, stop=None, target=None,
                 TICK_SIZE=.25, SCALE_FACTOR=1, MAX_DAILY_STOP=10, OUTPUT_LOG=False, FEATURE_LIST='', ALLOW_FLIP=True,
                 SCALE_IN=False, MAX_CONTRACT=1, # MAX_CONTRACT will only be checked when SCALE_IN=True
                 USE_ALT_TIMEFRAME=True, alt_input_df=None, # When ALT_TIMEFRAME is turned on, we include a snapshot of an alternative timeframe in the output
                 COMMISSION=.1): #Commission is in ticks

        self.data = input_df # pandas.DataFrame
        self.alt_data = alt_input_df

        self.frame_length = frame_length
        self.PL = 0
        self.realized_PL = 0
        self.last = 0

        ###### Recording the number of trades and flags whether it entered a new trade (closing is ignored). Mainly used for the design of the reward function.
        self.num_of_trade = 0
        self.entering_trade = False

        self.frame = None # numpy.array of size self.frame_length * self.frame_length * 1
        self.alt_frame = None # numpy.array of size self.frame_length * self.frame_length * 1
        self.stop = stop
        self.target = target

        self.TICK_SIZE = TICK_SIZE
        self.MAX_DAILY_STOP = MAX_DAILY_STOP
        self.OUTPUT_LOG = OUTPUT_LOG
        self.FEATURE_LIST = FEATURE_LIST
        self.SCALE_IN = SCALE_IN
        self.SCALE_FACTOR = SCALE_FACTOR
        self.MAX_CONTRACT = MAX_CONTRACT
        self.ALLOW_FLIP = ALLOW_FLIP
        self.COMMISSION = COMMISSION

        self.USE_ALT_TIMEFRAME = USE_ALT_TIMEFRAME
        if self.USE_ALT_TIMEFRAME:
            if self.alt_data is None:
                raise ValueError('Require alt_input_df.')
            self.dates, self.date_index, self.index_mapping = datepreprocess(self.data, self.alt_data)
        else:
            self.dates, self.date_index, _ = datepreprocess(self.data, self.data)

        self.current_date = None # date object
        self.current_date_index = None # an integer indicating the position of current_date in dates
        self.current_range = None # a tuple (start, end) indicating the range of data under the current trading day
        self.current_step = -1

        self.current_position = 0
        self.current_entry = 0
        self.current_stop = 0
        self.current_target = 0
        self.terminal = True

        self.MIN_FRAME = 10



    def reward_function(self): # override the reward function to try other approaches
        return self.realized_PL + self.current_position * (self.last - self.current_entry) - self.PL

    def get_data(self, selected_date):

        if isinstance(selected_date, int):
            ind = selected_date
            if ind >= len(self.dates) or ind < 0:
                raise ValueError(f'Index of dates out of bounds ({len(self.dates)}).')
        elif isinstance(selected_date, date):
            if selected_date not in self.dates:
                raise ValueError('selected_date is not in the array of dates.')
            ind = self.dates.index(selected_date)

        if ind == 0:
            start = 0
        else:
            start = self.date_index[ind - 1] + 1

        return (start, self.date_index[ind])

    def reset(self, selected_date=None):

        self.PL = 0
        self.realized_PL = 0
        self.current_step = 0
        self.current_position = 0
        self.current_entry = 0
        self.current_stop = 0
        self.current_target = 0

        self.terminal = False
        self.num_of_trade = 0
        self.entering_trade = False

        if selected_date is None:
            self.current_date_index = random.randrange(0, len(self.dates))
            self.current_date = self.dates[self.current_date_index]
        else:
            if isinstance(selected_date, int):
                self.current_date_index = selected_date
                self.current_date = self.dates[selected_date]
            elif isinstance(selected_date, date):
                if selected_date not in self.dates:
                    raise ValueError('Input date is not an available trading date in the data.')
                self.current_date_index = self.dates.index(selected_date)
                self.current_date = selected_date
            else:
                raise TypeError('Unsupported type for the date.')
        self.current_range = self.get_data(self.current_date)

        ###### Select a random date with enough frames
        count = 1
        while selected_date is None and self.current_range[1] - self.current_range[0] + 1 < self.MIN_FRAME:
            self.current_date = self.dates[random.randrange(0, len(self.dates))]
            self.current_range = self.get_data(self.current_date)
            ###### A warning for potential infinite loop
            if count % 10000 == 0: warnings.warn(f'Potentially unable to find a date with more than {self.MIN_FRAME} frames.')
            count += 1

        ###### Set the initial frame and alternative timeframe
        self.frame = trade_to_cropped_pic(self.current_range[0], self.current_range[0], self.data, pic_size=self.frame_length, TICK_SIZE=self.TICK_SIZE * self.SCALE_FACTOR)
        if self.USE_ALT_TIMEFRAME:
            self.alt_frame = trade_to_cropped_pic(0, self.index_mapping[self.current_range[0]], self.alt_data, pic_size=self.frame_length, TICK_SIZE=self.TICK_SIZE * self.SCALE_FACTOR)

        self.last = self.data.loc[self.current_range[0], 'last']

        return self.frame, self.alt_frame, self.data.loc[self.current_range[0], self.FEATURE_LIST].to_list() + [self.current_position, self.PL, self.realized_PL]



    def sell(self):
        """Goes short with one contract.
        """
        if self.current_position < 0:
            if (not self.SCALE_IN) or (self.SCALE_IN and self.MAX_CONTRACT <= abs(self.current_position)):
                if self.OUTPUT_LOG:
                    print(f'{self.current_date}: Currently has a long position. Scaling-in is not allowed or exceeds the maximal number of contracts.')
                return
        if self.current_position > 0:
            if self.ALLOW_FLIP:
                self.flatten()
            else:
                if self.OUTPUT_LOG:
                    print(f'{self.current_date}: Currently has a short position. Flipping is not allowed.')
                return

        self.current_entry = self.data.loc[self.current_range[0] + self.current_step + 1, 'open'] #Enter at the open of the next candle
        self.current_position = self.current_position - 1
        self.realized_PL -= self.COMMISSION
        self.num_of_trade += 1
        self.entering_trade = True

        ###### Only set stop or target when 1. there is a stop/target given, and 2. this is a new entry, not a repeated scale-in
        if self.stop != None and self.current_position == -1:
            self.current_stop = self.current_entry + self.stop * self.TICK_SIZE
        if self.target != None and self.current_position == -1:
            self.current_target = self.current_entry - self.target * self.TICK_SIZE



    def buy(self):
        """Goes long with one contract.
        """
        if self.current_position > 0:
            if (not self.SCALE_IN) or (self.SCALE_IN and self.MAX_CONTRACT <= abs(self.current_position)):
                if self.OUTPUT_LOG:
                    print(f'{self.current_date}: Currently has a long position. Scaling in is not allowed or exceeds the maximal number of contracts.')
                return
        if self.current_position < 0:
            if self.ALLOW_FLIP:
                self.flatten()
            else:
                if self.OUTPUT_LOG:
                    print(f'{self.current_date}: Currently has a short position. Flipping is not allowed.')
                return

        self.current_entry = self.data.loc[self.current_range[0] + self.current_step + 1, 'open'] #Enter at the open of the next candle
        self.current_position = self.current_position + 1
        self.realized_PL -= self.COMMISSION
        self.num_of_trade += 1
        self.entering_trade = True

        ###### Only set stop or target when 1. there is a stop/target given, and 2. this is a new entry, not a repeated scale-in
        if self.stop != None and self.current_position == 1:
            self.current_stop = self.current_entry - self.stop * self.TICK_SIZE
        if self.target != None and self.current_position == 1:
            self.current_target = self.current_entry + self.target * self.TICK_SIZE



    def flatten(self, target_hit = False, stop_hit = False):
        """Flattens the current position
        """
        if target_hit and stop_hit:
            raise ValueError('We assume stop and target do not hit simultaneously.')

        if self.current_position != 0:
            if target_hit:
                assert self.current_position != 0
                self.realized_PL = self.realized_PL + self.current_position * self.target * self.TICK_SIZE
            elif stop_hit:
                assert self.current_position != 0
                self.realized_PL = self.realized_PL + self.current_position * self.stop * self.TICK_SIZE
            else:
                self.realized_PL = self.realized_PL + self.current_position * (self.last - self.current_entry)
            self.realized_PL -= self.COMMISSION

        self.current_position = 0
        self.current_entry = 0
        self.current_stop = 0
        self.current_target = 0



    def step(self, action):

        """Performs an action.
        Arguments:
              action = 0: do nothing
                       1: long
                       2: short
                       3: flatten
        Returns:
              frame: a self.frame_length*self.frame_length*1 numpy array
              alt_frame: None (if self.USE_ALT_TIMEFRAME==False) or a self.frame_length*self.frame_length*1 numpy array
              features: a (len(self.FEATURE_LIST) + 2)*1 numpy array
              reward: an integer
              terminal: a boolean recording whether the game is over (at max drawdown or at market close)
        """

        self.entering_trade = False

        if not self.terminal:
            if action == 0:
                pass
            elif action == 1:
                self.buy()
            elif action == 2:
                self.sell()
            elif action == 3:
                self.flatten()
            else:
                raise ValueError(f'Unsupported action {action}.')

            ###### Update frame
            self.current_step = self.current_step + 1
            self.frame = trade_to_cropped_pic(self.current_range[0], self.current_range[0] + self.current_step, self.data, pic_size=self.frame_length, TICK_SIZE=self.TICK_SIZE * self.SCALE_FACTOR)
            if self.USE_ALT_TIMEFRAME:
                self.alt_frame = trade_to_cropped_pic(0, self.index_mapping[self.current_range[0] + self.current_step], self.alt_data, pic_size=self.frame_length, TICK_SIZE=self.TICK_SIZE * self.SCALE_FACTOR)

            ###### We use range bar chart and we assume the stop and target cannot be filled at the same time (target stop range is larger than the size of each bar)
            high = self.data.loc[self.current_range[0] + self.current_step, 'high']
            low = self.data.loc[self.current_range[0] + self.current_step, 'low']

            if self.current_position > 0:
                if self.target != None and high >= self.current_target:
                    self.flatten(target_hit=True)
                elif self.stop != None and low <= self.current_stop:
                    self.flatten(stop_hit=True)
            elif self.current_position < 0:
                if self.target != None and low <= self.current_target:
                    self.flatten(target_hit=True)
                elif self.stop != None and high >= self.current_stop:
                    self.flatten(stop_hit=True)


        self.last = self.data.loc[self.current_range[0] + self.current_step, 'last']
        reward = self.reward_function()
        self.PL = self.realized_PL + self.current_position * (self.last - self.current_entry)

        ###### If the maximal daily drawdown is hit or it is at the last entry of the trading day, liquidate and mark the self.terminal to be True
        if self.current_step == self.current_range[1] - self.current_range[0] or self.PL <= - self.MAX_DAILY_STOP * self.TICK_SIZE:
            self.flatten()
            self.terminal = True

        ###### returns the frame, the action reward (currently using P&L difference), and a boolean recording whether the game is over (currently meaning that the agent hit the maximal drawdown).
        return (self.frame, self.alt_frame,
                self.data.loc[self.current_range[0] + self.current_step, self.FEATURE_LIST].to_list() + [self.current_position, self.PL, self.realized_PL],
                reward,
                self.terminal)

    def get_PL(self):
        return self.PL

    def get_date(self):
        return self.current_date


class TradeWrapper:

    def __init__(self, input_df, no_op_steps, frame_length=84, history_length=4, stop=None, target=None,
                 TICK_SIZE=.25, SCALE_FACTOR=1, MAX_DAILY_STOP=10, OUTPUT_LOG=False, FEATURE_LIST='', ALLOW_FLIP=True,
                 SCALE_IN=False, MAX_CONTRACT=1,
                 USE_ALT_TIMEFRAME=True, alt_input_df=None,
                 COMMISSION=0, env=TradeEnv):

        self.env = env(input_df, frame_length,
                       stop=stop, target=target, TICK_SIZE=TICK_SIZE, SCALE_FACTOR=SCALE_FACTOR, MAX_DAILY_STOP=MAX_DAILY_STOP, OUTPUT_LOG=OUTPUT_LOG,
                       FEATURE_LIST=FEATURE_LIST, ALLOW_FLIP=ALLOW_FLIP, SCALE_IN=SCALE_IN, MAX_CONTRACT=MAX_CONTRACT,
                       USE_ALT_TIMEFRAME=USE_ALT_TIMEFRAME, alt_input_df=alt_input_df,
                       COMMISSION=COMMISSION)

        self.frame = None
        self.alt_frame = None
        self.no_op_steps = no_op_steps
        self.frame_length = frame_length
        self.history_length = history_length
        self.USE_ALT_TIMEFRAME = USE_ALT_TIMEFRAME
        self.state = None # list of three objects: a numpy array of size 1*84*84*history_length, None (If USE_ALT_TIMEFRAME==False) or a numpy array of size 1*84*84*1, numpy array of size 1*feature_num



    def reset(self, selected_date=None, evaluation=False):
        """Resets the environment
        Arguments:
            evaluation: Set to True when the agent is being evaluated. Takes a random number of no-op steps if True.
        """

        self.frame, self.alt_frame, features = self.env.reset(selected_date=selected_date)

        # For the initial state, we stack the first frame four times
        self.state = [np.repeat(self.frame, self.history_length, axis=2).reshape(1, self.frame_length, self.frame_length, self.history_length),
                      self.alt_frame.reshape(1, self.frame_length, self.frame_length, 1),
                      np.array(features).reshape(1,-1)]

        # If evaluating, take a random number of no-op steps.
        # This adds an element of randomness, so that the each
        # evaluation is slightly different.
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.step(0)



    def step(self, action):
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
            render_mode: None doesn't render anything, 'human' renders the screen in a new window, 'rgb_array' returns an np.array with rgb values
        Returns:
            new_frame: The processed new frame as a result of that action
            alt_frame: The alternative timeframe that one can observe after that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
            life_lost: Whether a life has been lost
            new_frame: The raw new frame as a result of that action
            If render_mode is set to 'rgb_array' this also returns the rendered rgb_array
        """
        self.frame, self.alt_frame, features, reward, terminal = self.env.step(action)

        self.state = [np.append(self.state[0][0, :, :, 1:], self.frame, axis=2).reshape(1, self.frame_length, self.frame_length, self.history_length),
                      self.alt_frame.reshape(1, self.frame_length, self.frame_length, 1),
                      np.array(features).reshape(1,-1)]

        return self.frame, self.alt_frame, features, reward, terminal



    def get_state(self):
        """Returns self.state
        """
        return self.state

    def get_PL(self):
        return self.env.get_PL()

    def get_date(self):
        """Returns the current date of the underlying TradeEnv object
        """
        return self.env.get_date()



    def valid_date(self, selected_date, min_line=10):
        """Check whether the selected_date has more than min_line entries
        """
        data_range = self.env.get_data(selected_date)
        if data_range[1] - data_range[0] + 1 < min_line:
            return False
        return True
