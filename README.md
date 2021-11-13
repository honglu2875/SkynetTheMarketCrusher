# SkynetTheMarketCrusher
A library for nurturing trader behaviors on a neural network in the future market with DQN type reinforcement learning technique.

(update note: This was heavily worked on more than a year ago in 2020 but most of the work was kept private. The README is now cleaned up and updated on Nov 13 2021 because of a renewed interest to revive the hobby project. It's been a year and a lot have changed. As of now, the code is not tested against all the changes of libraries including DeepMind's bsuite, acme, dm_env, etc.)


The purpose of this repo is to transform SierraChart data dump (a .csv file that records future data, E-mini S&P future by default) into a game-like environment that fits DeepMind's bsuite. The documentation here is very simplistic because I do not know who else could be interested. But in case there is any, please contact me and I will make it more user-friendly.

(Note: The training of DQN agent uses this library but is not saved inside this repo.)

The most important files are trade.py and environment.py.

## In trade.py

### TradeEnv: 
#### The core class. Simulate a game-like environment. It will fix a random date with more than 10 candlestick bars, show an intraday historical chart and possibly attach an alternate timeframe (USE_ALT_TIMEFRAME) and trade. 

During trading, an agent can take the following actions (TradeEnv.step(action))
action=0: do nothing
action=1: long
action=2: short
action=3: flatten
Once the step method is invoked, an action will be taken, the next candlestick will be loaded and the chart will be updated.

### TradeWrapper: 
#### Wrap around TradeEnv to make it more suitable for DQN training.
## In environment.py
### trade_environment: 
#### Further wrap around TradeEnv to fit DeepMind's bsuite
