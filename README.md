# SkynetTheMarketCrusher
A library used in a hobby study to train trader behaviors on a neural network in the future market with DQN type reinforcement learning technique (training code not included).

(update note: This was heavily worked on more than a year ago in 2020 but most of the work was kept private. The README is now cleaned up and updated on Nov 13 2021 because of a renewed interest to revive the hobby project. It's been a year and a lot have changed. As of now, the code is not tested against the new versions of libraries including DeepMind's bsuite, acme, dm_env, etc.)

The purpose of this repo is to transform SierraChart data dump (a .csv file that records future data, E-mini S&P future by default) into a game-like environment that fits DeepMind's bsuite.

The most important files are trade.py and environment.py.

## In trade.py

**TradeEnv:**
The core class. Simulate a game-like environment. It will 
- fix a random date with more than 10 candlestick bars, 
- show an intraday historical chart and optionally attach an alternate timeframe (USE_ALT_TIMEFRAME),
- and trade. 

During trading, an agent can take the following actions
- action=0: do nothing
- action=1: long
- action=2: short
- action=3: flatten

Once ```TradeEnv.step(action)``` method is invoked, the corresponding action will be taken. The next candlestick will be loaded and the chart will be updated.

**TradeWrapper:**
Wrap around TradeEnv to make it more suitable for DQN training.
## In environment.py
**trade_environment:**
Further wrap around TradeEnv to fit DeepMind's bsuite


---

### Some thoughts and why the result in 2020 was not splendid:

When I first get to know about neural network and reinforcement learning in 2020, it was a mysterious black box that was able to make everything work. This was part of the reasons why I constructed this library to pass trading data through this magical black box and see if miracle happens. Now that the charm has faded off, I'm able to look at it with a pair of cold eyes and rethink what did not work and what should be improved:

Neural network is one of many ways to construct a function that takes an input and emits an output, and it is flexible enough to be modified (by gradient method for example) to let the output be as close to ground truth as possible on a fixed amount of "training" data. In the kind of mathematical language that I'm used to, I'd say that neural network forms a moduli space isomorphic to R^n inside the space of continuous functions. There is no mystery around that as they are essentially a bunch of complicated piecewise-linear functions (as long as we only use the traditional layers including Dense, Convolution, etc.), and whatever we are doing is in some sense **a form of generalized linear regression**.

DQN has a beautiful-looking backward recursion formula and some fancy experiments started with Atari by Google DeepMind. But imagine the whole game as a game tree/graph (state:vertices, action:edges) with a "Q-value" attached to each vertex satistying the recursion. What DQN does is nothing but starting with a random assignment of the Q-value and update through edges one-by-one to get as close to the ground true "Q-value" as possible (where convergence needs a proof). If the memory (replay buffer) is organized by random play, it's simply a fancy mix of ideas from Depth-first Search (random play into one branch at a time) and Monte-Carlo Search (most of the time choose the best action to become a memory).

The biggest problems with a trading environment are three-fold:
1. It's not deterministic => the future reward is not determined by the past and the action (we are not market makers, sadly). In other words, there might be hidden high probability strategies, but the amount of noise is immense.
2. The data is VERY limited. Despite having randomized intraday play, picture inputs (charts) are limited to what's available in the data. Overfitting on the repeatedly occuring chart is a massive concern (and I did end up overfitting in almost all my experiments)
3. Generalizability of chart knowledge. If the neural network recognizes a pattern, does it still recognize if after translation and scaling? This is basically the robustness issue of the neural network and it needs a lot of engineering.
...(to be cont)

