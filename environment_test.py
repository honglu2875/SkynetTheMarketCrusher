from absl.testing import absltest
from dm_env import test_utils
from SMC.environment import *
from SMC.datareader import datareader
import numpy as np

class TradeInterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

    def make_object_under_test(self):
        input_df = datareader('data/ESU0-test.csv')

        return trade_environment(input_df)

    def make_action_sequence(self):
        valid_actions = [0, 1, 2, 3]
        rng = np.random.RandomState(42)

        for _ in range(100):
            yield rng.choice(valid_actions)


if __name__ == '__main__':
    absltest.main()
