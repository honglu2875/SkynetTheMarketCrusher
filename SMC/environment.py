from bsuite.environments import base
import dm_env
from dm_env import specs
from .trade import TradeEnv
import numpy as np
import pandas as pd
import random



class trade_environment(base.Environment):
    ''' Wrapping the TradeEnv to subclass the base.Environment
        Input:input_df, frame_length=84, history_length=4, stop=None, target=None,
            TICK_SIZE=.25, SCALE_FACTOR=1, MAX_DAILY_STOP=10, OUTPUT_LOG=False, FEATURE_LIST=[], ALLOW_FLIP=True,
            SCALE_IN=False, MAX_CONTRACT=1,
            USE_ALT_TIMEFRAME=True, alt_input_df=None,
            COMMISSION=0, env=TradeEnv
    '''
    def __init__(self, input_df, frame_length=84, history_length=4, stop=None, target=None,
              TICK_SIZE=.25, SCALE_MIN=.25, SCALE_MAX=4, MAX_DAILY_STOP=10, OUTPUT_LOG=False, FEATURE_LIST=[], ALLOW_FLIP=True,
              SCALE_IN=False, MAX_CONTRACT=1,
              USE_ALT_TIMEFRAME=False, alt_input_df=None,
              COMMISSION=0, env=TradeEnv):

        super().__init__()

        self._env = env(input_df, frame_length,
                    stop=stop, target=target, TICK_SIZE=TICK_SIZE, MAX_DAILY_STOP=MAX_DAILY_STOP, OUTPUT_LOG=OUTPUT_LOG,
                    FEATURE_LIST=FEATURE_LIST, ALLOW_FLIP=ALLOW_FLIP, SCALE_IN=SCALE_IN, MAX_CONTRACT=MAX_CONTRACT,
                    USE_ALT_TIMEFRAME=USE_ALT_TIMEFRAME, alt_input_df=alt_input_df,
                    COMMISSION=COMMISSION)
        self._frame = None
        self._alt_frame = None
        self._state = None # list of three objects: a numpy array of size 84*84*history_length, None (If USE_ALT_TIMEFRAME==False) or a numpy array of size 84*84, numpy array of size feature_num

        self._total_days = 0

        self._frame_length = frame_length
        self._history_length = history_length
        self._feature_num = len(FEATURE_LIST) + 3 # features plus position, P&L and realized P&L


        self._USE_ALT_TIMEFRAME = USE_ALT_TIMEFRAME

        self._SCALE_MIN = SCALE_MIN
        self._SCALE_MAX = SCALE_MAX

        self._reset()


    def _reset(self):

        self._frame, self._alt_frame, _features = self._env.reset(scale=random.uniform(self._SCALE_MIN, self._SCALE_MAX))

        # For the initial state, we stack the first frame four times
        self._state = [np.repeat(self._frame, self._history_length, axis=2).astype('uint8'),
                       self._alt_frame.reshape(self._frame_length, self._frame_length).astype('uint8'),
                       np.array(_features).astype('float32')]

        return dm_env.restart(self._get_observation())

    def _step(self, action: int):
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
        self._frame, self._alt_frame, _features, _reward, _terminal = self._env.step(action)

        self._state = [np.append(self._state[0][:, :, 1:], self._frame, axis=2).astype('uint8'),
                      self._alt_frame.reshape(self._frame_length, self._frame_length).astype('uint8'),
                      np.array(_features).astype('float32')]

        if _terminal:
            self._total_days += 1
            return dm_env.termination(reward=_reward, observation=self._get_observation())

        return dm_env.transition(reward=_reward, observation=self._get_observation())



    def _get_observation(self):
        return self._state


    def observation_spec(self):
        return [specs.Array(shape=(self._frame_length, self._frame_length, self._history_length), dtype=np.uint8),
                specs.Array(shape=(self._frame_length, self._frame_length), dtype=np.uint8),
                specs.Array(shape=(self._feature_num, ), dtype=np.float32)] # A list of historical frames, an alternative timeframe and an array of features

    def action_spec(self):
        return specs.DiscreteArray(4, name='action')

    def bsuite_info(self):
        return dict(total_days=self._total_days)

    def get_PL(self):
        return self._env.get_PL()

    def get_date(self):
        """Returns the current date of the underlying TradeEnv object
        """
        return self._env.get_date()
