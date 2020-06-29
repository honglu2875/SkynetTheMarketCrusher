import random
import warnings
from datetime import datetime

class TradeEnv:
    
    def __init__(self, input_df, dates, date_index, frame_length, stop=None, target=None, TICK_SIZE=.25, MAX_DAILY_STOP=10, OUTPUT_LOG=False, FEATURE_LIST='', SCALE_IN=False, ALLOW_FLIP=True):

      self.data = input_df # pandas.DataFrame
      self.dates = dates # list of datetime objects
      self.date_index = date_index # list of int

      self.frame_length = frame_length
      self.PL = 0
      self.realized_PL = 0
      self.frame = None # numpy.array of size 84*84*1
      self.stop = stop
      self.target = target
      
      self.TICK_SIZE = TICK_SIZE
      self.MAX_DAILY_STOP = MAX_DAILY_STOP
      self.OUTPUT_LOG = OUTPUT_LOG
      self.FEATURE_LIST = FEATURE_LIST
      self.SCALE_IN = SCALE_IN
      self.ALLOW_FLIP = ALLOW_FLIP

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
      self.PL = 0
      self.current_step = 0
      self.current_position = 0
      self.current_entry = 0
      self.current_stop = 0
      self.current_target = 0
      self.terminal = False

      if selected_date == None:
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

      count = 1
      while selected_date == None and self.current_range[1] - self.current_range[0] + 1 < self.MIN_FRAME:
        self.current_date = self.dates[random.randrange(0, len(self.dates))]
        self.current_range = self.get_data(self.current_date)
        ###### A warning to prevent infinite loop
        if count % 10000 == 0: warnings.warn(f'Potentially unable to find a date with more than {self.MIN_FRAME} frames.')
        count += 1

      self.frame = trade_to_cropped_pic(self.current_range[0], self.current_range[0], self.data, pic_size=self.frame_length)
      return self.frame, self.data.loc[self.current_range[0], FEATURE_LIST].to_list() + [self.current_position, self.PL]

        
        
    def sell(self):
      """Goes short with one contract.
      """
      if self.current_position < 0:
        if not self.SCALE_IN:
          if self.OUTPUT_LOG:
            print(f'{self.current_date}: Currently has a long position. Scaling in is not allowed.')
          return
      if self.current_position > 0:
        if self.ALLOW_FLIP:
          self.flatten()
        else:
          if self.OUTPUT_LOG:
            print(f'{self.current_date}: Currently has a short position. Flipping is not allowed.')
          return

      self.current_entry = self.data.loc[self.current_range[0] + self.current_step, 'last']
      self.current_position = self.current_position - 1
      
      ###### Only set stop or target when 1. there is a stop/target given, and 2. this is a new entry, not a repeated scale-in
      if self.stop != None and self.current_position == -1:
        self.current_stop = self.current_entry + self.stop * self.TICK_SIZE
      if self.target != None and self.current_position == -1:
        self.current_target = self.current_entry - self.target * self.TICK_SIZE



    def buy(self):
      """Goes long with one contract.
      """
      if self.current_position > 0:
        if not self.SCALE_IN:
          if self.OUTPUT_LOG:
            print(f'{self.current_date}: Currently has a long position. Scaling in is not allowed.')
          return
      if self.current_position < 0:
        if self.ALLOW_FLIP:
          self.flatten()
        else:
          if self.OUTPUT_LOG:
            print(f'{self.current_date}: Currently has a short position. Flipping is not allowed.')
          return

      self.current_entry = self.data.loc[self.current_range[0] + self.current_step, 'last']
      self.current_position = self.current_position + 1
      
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
          self.realized_PL = self.realized_PL + self.current_position * (self.data.loc[self.current_range[0] + self.current_step, 'last'] - self.current_entry)
        
      self.current_position = 0
      self.current_entry = 0
      self.current_stop = 0
      self.current_target = 0


    
    def step(self, action):
        
      """Performs an action.
      Arguments:
            action = 0: do nothing
                     1: long (stop: 24 tick, target: 24 tick)
                     2: short (stop: 24 tick, target: 24 tick)
                     3: flatten
      Returns:
            frame: an self.frame_length*self.frame_length*1 numpy array
            features: a (len(self.FEATURE_LIST) + 2)*1 numpy array
            reward: an integer
            terminal: a boolean recording whether the game is over (at max drawdown or at market close)
      """
      
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
        self.frame = trade_to_cropped_pic(self.current_range[0], self.current_range[0] + self.current_step, self.data, pic_size=self.frame_length)

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

    
      last = self.data.loc[self.current_range[0] + self.current_step, 'last']
      reward = self.realized_PL + self.current_position * (last - self.current_entry) - self.PL
      self.PL = self.realized_PL + self.current_position * (last - self.current_entry)

      ###### If the maximal daily drawdown is hit or it is at the last entry of the trading day, liquidate and mark the self.terminal to be True
      if self.current_step == self.current_range[1] - self.current_range[0] or self.PL <= - self.MAX_DAILY_STOP * self.TICK_SIZE:
        self.flatten()
        self.terminal = True

      ###### returns the frame, the action reward (currently using P&L difference), and a boolean recording whether the game is over (currently meaning that the agent hit the maximal drawdown).
      return (self.frame, 
              self.data.loc[self.current_range[0] + self.current_step, self.FEATURE_LIST].to_list() + [self.current_position, self.PL],
              reward, 
              self.terminal)

    
    
    def get_date(self):
      return self.current_date


class TradeWrapper:
    
    def __init__(self, input_df, dates, date_index, no_op_steps, frame_length=84, history_length=4, stop=None, target=None, TICK_SIZE=.25, MAX_DAILY_STOP=10, OUTPUT_LOG=False, FEATURE_LIST='', SCALE_IN=False, ALLOW_FLIP=True):
        self.env = TradeEnv(input_df, dates, date_index, frame_length, stop=stop, target=target,
                            TICK_SIZE=TICK_SIZE, MAX_DAILY_STOP=MAX_DAILY_STOP, OUTPUT_LOG=OUTPUT_LOG, FEATURE_LIST=FEATURE_LIST, SCALE_IN=SCALE_IN, ALLOW_FLIP=ALLOW_FLIP)
        self.no_op_steps = no_op_steps
        self.frame_length = frame_length
        self.history_length = history_length
        self.state = None # list of two objects: numpy array of size 1*84*84*history_length, numpy array of size 1*feature_num

    
    
    def reset(self, selected_date=None, evaluation=False):
        """Resets the environment
        Arguments:
            evaluation: Set to True when the agent is being evaluated. Takes a random number of no-op steps if True.
        """

        self.frame, features = self.env.reset(selected_date=selected_date)

        # For the initial state, we stack the first frame four times
        self.state = [np.repeat(self.frame, self.history_length, axis=2).reshape(1, self.frame_length, self.frame_length, self.history_length), np.array(features).reshape(1,-1)]

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
            processed_frame: The processed new frame as a result of that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
            life_lost: Whether a life has been lost
            new_frame: The raw new frame as a result of that action
            If render_mode is set to 'rgb_array' this also returns the rendered rgb_array
        """
        new_frame, features, reward, terminal = self.env.step(action)

        self.state = [np.append(self.state[0][0, :, :, 1:], new_frame, axis=2).reshape(1, self.frame_length, self.frame_length, self.history_length), np.array(features).reshape(1,-1)]

        return new_frame, features, reward, terminal

    
    
    def get_state(self):
        """Returns self.state
        """
        return self.state

    
    
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
