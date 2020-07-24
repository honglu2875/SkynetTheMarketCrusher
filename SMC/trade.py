import random
import warnings
from datetime import datetime, date
from .chart import trade_to_cropped_pic
from .datareader import datepreprocess
import numpy as np
import pandas as pd

class TradeEnv:

    def __init__(self, input_df, frame_length, stop=None, target=None,
                 TICK_SIZE=.25, MAX_DAILY_STOP=10, OUTPUT_LOG=False, FEATURE_LIST=[], ALLOW_FLIP=True,
                 SCALE_IN=False, MAX_CONTRACT=1, # MAX_CONTRACT will only be checked when SCALE_IN=True
                 USE_ALT_TIMEFRAME=True, alt_input_df=None, # When ALT_TIMEFRAME is turned on, we include a snapshot of an alternative timeframe in the output
                 COMMISSION=.1): #Commission is in ticks

        self._data = input_df # pandas.DataFrame
        self._alt_data = alt_input_df

        self._frame_length = frame_length
        self._PL = 0
        self._realized_PL = 0
        self._last = 0

        ###### Recording the number of trades and flags whether it entered a new trade (closing is ignored). Mainly used for the design of the reward function.
        self._num_of_trade = 0
        self._entering_trade = False

        self._frame = None # numpy.array of size self._frame_length * self._frame_length * 1
        self._alt_frame = np.zeros((self._frame_length, self._frame_length, 1)) # numpy.array of size self._frame_length * self._frame_length * 1
        self._stop = stop
        self._target = target

        self._TICK_SIZE = TICK_SIZE
        self._MAX_DAILY_STOP = MAX_DAILY_STOP
        self._OUTPUT_LOG = OUTPUT_LOG
        self._FEATURE_LIST = FEATURE_LIST
        self._SCALE_IN = SCALE_IN
        self._SCALE_FACTOR = 1 # scaling the candle size. 1 by default. Can be modified in each reset.
        self._MAX_CONTRACT = MAX_CONTRACT
        self._ALLOW_FLIP = ALLOW_FLIP
        self._COMMISSION = COMMISSION

        self._USE_ALT_TIMEFRAME = USE_ALT_TIMEFRAME
        if self._USE_ALT_TIMEFRAME:
            if self._alt_data is None:
                raise ValueError('Require alt_input_df.')
            self._dates, self._date_index, self._index_mapping = datepreprocess(self._data['datetime'].to_list(), self._alt_data['datetime'].to_list())
        else:
            self._dates, self._date_index, _ = datepreprocess(self._data['datetime'].to_list(), self._data['datetime'].to_list())

        self._current_date = None # date object
        self._current_date_index = None # an integer indicating the position of current_date in dates
        self._current_range = None # a tuple (start, end) indicating the range of data under the current trading day
        self._current_step = -1

        self._current_position = 0
        self._current_entry = 0
        self._current_stop = 0
        self._current_target = 0
        self._terminal = True

        self._MIN_FRAME = 10



    def reward_function(self): # override the reward function to try other approaches
        return self._realized_PL + self._current_position * (self._last - self._current_entry) - self._PL

    def get_data(self, selected_date):

        if isinstance(selected_date, int):
            ind = selected_date
            if ind >= len(self._dates) or ind < 0:
                raise ValueError(f'Index of dates out of bounds ({len(self._dates)}).')
        elif isinstance(selected_date, date):
            if selected_date not in self._dates:
                raise ValueError('selected_date is not in the array of dates.')
            ind = self._dates.index(selected_date)

        if ind == 0:
            start = 0
        else:
            start = self._date_index[ind - 1] + 1

        return (start, self._date_index[ind])

    def reset(self, selected_date=None, scale=None):

        self._PL = 0
        self._realized_PL = 0
        self._current_step = 0
        self._current_position = 0
        self._current_entry = 0
        self._current_stop = 0
        self._current_target = 0

        self._terminal = False
        self._num_of_trade = 0
        self._entering_trade = False

        if scale is not None: # the modified scale will not reset when a game terminates
            self._SCALE_FACTOR = scale

        if selected_date is None:
            self._current_date_index = random.randrange(0, len(self._dates))
            self._current_date = self._dates[self._current_date_index]
        else:
            if isinstance(selected_date, int):
                self._current_date_index = selected_date
                self._current_date = self._dates[selected_date]
            elif isinstance(selected_date, date):
                if selected_date not in self._dates:
                    raise ValueError('Input date is not an available trading date in the data.')
                self._current_date_index = self._dates.index(selected_date)
                self._current_date = selected_date
            else:
                raise TypeError('Unsupported type for the date.')
        self._current_range = self.get_data(self._current_date)

        ###### Select a random date with enough frames
        count = 1
        while selected_date is None and self._current_range[1] - self._current_range[0] + 1 < self._MIN_FRAME:
            self._current_date = self._dates[random.randrange(0, len(self._dates))]
            self._current_range = self.get_data(self._current_date)
            ###### A warning for potential infinite loop
            if count % 10000 == 0: warnings.warn(f'Potentially unable to find a date with more than {self._MIN_FRAME} frames.')
            count += 1

        ###### Set the initial frame and alternative timeframe
        self._frame = trade_to_cropped_pic(self._current_range[0], self._current_range[0], self._data, pic_size=self._frame_length, TICK_SIZE=self._TICK_SIZE * self._SCALE_FACTOR)
        if self._USE_ALT_TIMEFRAME:
            self._alt_frame = trade_to_cropped_pic(0, self._index_mapping[self._current_range[0]], self._alt_data, pic_size=self._frame_length, TICK_SIZE=self._TICK_SIZE * self._SCALE_FACTOR)

        self._last = self._data.loc[self._current_range[0], 'last']

        return self._frame, self._alt_frame, self._data.loc[self._current_range[0], self._FEATURE_LIST].to_list() + [self._current_position, self._PL, self._realized_PL]



    def sell(self):
        """Goes short with one contract.
        """
        if self._current_position < 0:
            if (not self._SCALE_IN) or (self._SCALE_IN and self._MAX_CONTRACT <= abs(self._current_position)):
                if self._OUTPUT_LOG:
                    print(f'{self._current_date}: Currently has a long position. Scaling-in is not allowed or exceeds the maximal number of contracts.')
                return
        if self._current_position > 0:
            if self._ALLOW_FLIP:
                self.flatten()
            else:
                if self._OUTPUT_LOG:
                    print(f'{self._current_date}: Currently has a short position. Flipping is not allowed.')
                return

        self._current_entry = self._data.loc[self._current_range[0] + self._current_step + 1, 'open'] #Enter at the open of the next candle
        self._current_position = self._current_position - 1
        self._realized_PL -= self._COMMISSION
        self._num_of_trade += 1
        self._entering_trade = True

        ###### Only set stop or target when 1. there is a stop/target given, and 2. this is a new entry, not a repeated scale-in
        if self._stop != None and self._current_position == -1:
            self._current_stop = self._current_entry + self._stop * self._TICK_SIZE
        if self._target != None and self._current_position == -1:
            self._current_target = self._current_entry - self._target * self._TICK_SIZE



    def buy(self):
        """Goes long with one contract.
        """
        if self._current_position > 0:
            if (not self._SCALE_IN) or (self._SCALE_IN and self._MAX_CONTRACT <= abs(self._current_position)):
                if self._OUTPUT_LOG:
                    print(f'{self._current_date}: Currently has a long position. Scaling in is not allowed or exceeds the maximal number of contracts.')
                return
        if self._current_position < 0:
            if self._ALLOW_FLIP:
                self.flatten()
            else:
                if self._OUTPUT_LOG:
                    print(f'{self._current_date}: Currently has a short position. Flipping is not allowed.')
                return

        self._current_entry = self._data.loc[self._current_range[0] + self._current_step + 1, 'open'] #Enter at the open of the next candle
        self._current_position = self._current_position + 1
        self._realized_PL -= self._COMMISSION
        self._num_of_trade += 1
        self._entering_trade = True

        ###### Only set stop or target when 1. there is a stop/target given, and 2. this is a new entry, not a repeated scale-in
        if self._stop != None and self._current_position == 1:
            self._current_stop = self._current_entry - self._stop * self._TICK_SIZE
        if self._target != None and self._current_position == 1:
            self._current_target = self._current_entry + self._target * self._TICK_SIZE



    def flatten(self, target_hit = False, stop_hit = False):
        """Flattens the current position
        """
        if target_hit and stop_hit:
            raise ValueError('We assume stop and target do not hit simultaneously.')

        if self._current_position != 0:
            if target_hit:
                assert self._current_position != 0
                self._realized_PL = self._realized_PL + self._current_position * self._target * self._TICK_SIZE
            elif stop_hit:
                assert self._current_position != 0
                self._realized_PL = self._realized_PL + self._current_position * self._stop * self._TICK_SIZE
            else:
                self._realized_PL = self._realized_PL + self._current_position * (self._last - self._current_entry)
            self._realized_PL -= self._COMMISSION

        self._current_position = 0
        self._current_entry = 0
        self._current_stop = 0
        self._current_target = 0



    def step(self, action):

        """Performs an action.
        Arguments:
              action = 0: do nothing
                       1: long
                       2: short
                       3: flatten
        Returns:
              frame: a self._frame_length*self._frame_length*1 numpy array
              alt_frame: None (if self._USE_ALT_TIMEFRAME==False) or a self._frame_length*self._frame_length*1 numpy array
              features: a (len(self._FEATURE_LIST) + 2)*1 numpy array
              reward: an integer
              terminal: a boolean recording whether the game is over (at max drawdown or at market close)
        """

        self._entering_trade = False

        if not self._terminal:
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
            self._current_step = self._current_step + 1
            self._frame = trade_to_cropped_pic(self._current_range[0], self._current_range[0] + self._current_step, self._data, pic_size=self._frame_length, TICK_SIZE=self._TICK_SIZE * self._SCALE_FACTOR)
            if self._USE_ALT_TIMEFRAME:
                self._alt_frame = trade_to_cropped_pic(0, self._index_mapping[self._current_range[0] + self._current_step], self._alt_data, pic_size=self._frame_length, TICK_SIZE=self._TICK_SIZE * self._SCALE_FACTOR)

            ###### We use range bar chart and we assume the stop and target cannot be filled at the same time (target stop range is larger than the size of each bar)
            high = self._data.loc[self._current_range[0] + self._current_step, 'high']
            low = self._data.loc[self._current_range[0] + self._current_step, 'low']

            if self._current_position > 0:
                if self._target != None and high >= self._current_target:
                    self.flatten(target_hit=True)
                elif self._stop != None and low <= self._current_stop:
                    self.flatten(stop_hit=True)
            elif self._current_position < 0:
                if self._target != None and low <= self._current_target:
                    self.flatten(target_hit=True)
                elif self._stop != None and high >= self._current_stop:
                    self.flatten(stop_hit=True)


        self._last = self._data.loc[self._current_range[0] + self._current_step, 'last']
        reward = self.reward_function()
        self._PL = self._realized_PL + self._current_position * (self._last - self._current_entry)

        ###### If the maximal daily drawdown is hit or it is at the last entry of the trading day, liquidate and mark the self._terminal to be True
        if self._current_step == self._current_range[1] - self._current_range[0] or self._PL <= - self._MAX_DAILY_STOP * self._TICK_SIZE:
            self.flatten()
            self._terminal = True

        ###### returns the frame, the action reward (currently using P&L difference), and a boolean recording whether the game is over (currently meaning that the agent hit the maximal drawdown).
        return (self._frame, self._alt_frame,
                self._data.loc[self._current_range[0] + self._current_step, self._FEATURE_LIST].to_list() + [self._current_position, self._PL, self._realized_PL],
                reward,
                self._terminal)

    def get_PL(self):
        return self._PL

    def get_date(self):
        return self._current_date


class TradeWrapper:

    def __init__(self, input_df, no_op_steps, frame_length=84, history_length=4, stop=None, target=None,
                 TICK_SIZE=.25, MAX_DAILY_STOP=10, OUTPUT_LOG=False, FEATURE_LIST=[], ALLOW_FLIP=True,
                 SCALE_IN=False, MAX_CONTRACT=1,
                 USE_ALT_TIMEFRAME=True, alt_input_df=None,
                 COMMISSION=0, env=TradeEnv):

        self._env = env(input_df, frame_length,
                       stop=stop, target=target, TICK_SIZE=TICK_SIZE, MAX_DAILY_STOP=MAX_DAILY_STOP, OUTPUT_LOG=OUTPUT_LOG,
                       FEATURE_LIST=FEATURE_LIST, ALLOW_FLIP=ALLOW_FLIP, SCALE_IN=SCALE_IN, MAX_CONTRACT=MAX_CONTRACT,
                       USE_ALT_TIMEFRAME=USE_ALT_TIMEFRAME, alt_input_df=alt_input_df,
                       COMMISSION=COMMISSION)

        self._frame = None
        self._alt_frame = None
        self._no_op_steps = no_op_steps
        self._frame_length = frame_length
        self._history_length = history_length
        self._USE_ALT_TIMEFRAME = USE_ALT_TIMEFRAME
        self._state = None # list of three objects: a numpy array of size 1*84*84*history_length, None (If USE_ALT_TIMEFRAME==False) or a numpy array of size 1*84*84*1, numpy array of size 1*feature_num



    def reset(self, selected_date=None, evaluation=False):
        """Resets the environment
        Arguments:
            evaluation: Set to True when the agent is being evaluated. Takes a random number of no-op steps if True.
        """

        self._frame, self._alt_frame, features = self._env.reset(selected_date=selected_date)

        # For the initial state, we stack the first frame four times
        self._state = [np.repeat(self._frame, self._history_length, axis=2).reshape(1, self._frame_length, self._frame_length, self._history_length),
                      self._alt_frame.reshape(1, self._frame_length, self._frame_length, 1),
                      np.array(features).reshape(1,-1)]

        # If evaluating, take a random number of no-op steps.
        # This adds an element of randomness, so that the each
        # evaluation is slightly different.
        if evaluation:
            for _ in range(random.randint(0, self._no_op_steps)):
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
        self._frame, self._alt_frame, features, reward, terminal = self._env.step(action)

        self._state = [np.append(self._state[0][0, :, :, 1:], self._frame, axis=2).reshape(1, self._frame_length, self._frame_length, self._history_length),
                      self._alt_frame.reshape(1, self._frame_length, self._frame_length, 1),
                      np.array(features).reshape(1,-1)]

        return self._frame, self._alt_frame, features, reward, terminal



    def get_state(self):
        """Returns self._state
        """
        return self._state

    def get_PL(self):
        return self._env.get_PL()

    def get_date(self):
        """Returns the current date of the underlying TradeEnv object
        """
        return self._env.get_date()



    def valid_date(self, selected_date, min_line=10):
        """Check whether the selected_date has more than min_line entries
        """
        data_range = self._env.get_data(selected_date)
        if data_range[1] - data_range[0] + 1 < min_line:
            return False
        return True
