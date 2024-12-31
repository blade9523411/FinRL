import os
import pandas as pd
#imports modules
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl import config_tickers
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR

check_and_make_directories([TRAINED_MODEL_DIR])

train = pd.read.csv('') # reads into a pandas dataframe

train = train.set_index(train.columns[0]) #sets the first column as the index or date as index
train.index.names = [''] #cleans the column name to nothing

stock_dimension = len(train.tic.unique()) #dimension is set to the number of the unique tickers
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
# calculates state space: 1 represents the scalar value for account balances +
# 2* stock_dimension, which accounts for two pieces of information per stock, holdings and prices
# INDICATORS are the features, so the number of indicators per stock dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension # makes two same lists with values of 0.001 to length
num_stock_shares = [0] * stock_dimension #defines number of shares for each stock


env_kwargs = {
    "hmax": 100, # maximum number of shares traded in 1 transaction
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_train_gym = StockTradingEnv(df = train, **env_kwargs)
#creates instance of StockTradingEnv, **unpacks individually,
#initializes a Reinforcement Learning Environment

env_train, _ = e_train_gym.get_sb_env()
#environment conversion into a format that Stable-Baseline3 can work with
print(type(env_train))

agent = DRLAgent(env = env_train)

# Set the corresponding values to 'True' for the algorithms that you want to use
if_using_a2c = True
if_using_ddpg = False
if_using_ppo = True
if_using_td3 = False
if_using_sac = False

model_a2c = agent.get_model("a2c")
model_ppo = agent.get_model('ppo')

#different RL algorithms
if if_using_a2c:
  # set up logger
  tmp_path = RESULTS_DIR + '/a2c'
  new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_a2c.set_logger(new_logger_a2c)

if if_using_ppo:
  # set up logger
  tmp_path = RESULTS_DIR + '/ppo'
  new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_ppo.set_logger(new_logger_ppo)

trained_a2c = agent.train_model(model=model_a2c,
                             tb_log_name='a2c',
                             total_timesteps=50000) if if_using_a2c else None

trained_ppo = agent.train_model(model=model_ppo,
                             tb_log_name='ppo',
                             total_timesteps=50000) if if_using_ppo else None

trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None