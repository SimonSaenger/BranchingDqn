import numpy as np
import torch 
import random
from argparse import ArgumentParser 
import os 
import pandas as pd 

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d


def arguments(): 
    parser = ArgumentParser()
    parser.add_argument('--env', default='House_Of_Money')
    return parser.parse_args()


def save(agent, hist, args):

    path = './runs/{}/'.format(args.env)
    try: 
        os.makedirs(path)
    except: 
        pass 

    torch.save(agent.q.state_dict(), os.path.join(path, 'model_state_dict'))
    rewards, portfolio_hist, cash_hist, stock_hist, action_hist = hist

    plt.cla()
    plt.plot(rewards, c='r', alpha=0.3)
    plt.plot(gaussian_filter1d(rewards, sigma=5), c='r', label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Cumulative reward')
    plt.savefig(os.path.join(path, 'reward.png'))

    pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index=False)

    plt.cla()
    plt.clf()
    plt.plot(portfolio_hist, c='r', alpha=0.3)
    plt.plot(gaussian_filter1d(portfolio_hist, sigma=5), c='r', label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Value of Portfolio')
    plt.title('Branching DDQN: {}'.format(args.env))
    plt.savefig(os.path.join(path, 'portfolio.png'))

    plt.cla()
    plt.clf()
    plt.plot(cash_hist, c='r', alpha=0.3)
    plt.plot(gaussian_filter1d(cash_hist, sigma=5), c='r', label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('$Cash$')
    plt.title('Branching DDQN: {}'.format(args.env))
    plt.savefig(os.path.join(path, 'cash.png'))

    plt.cla()
    plt.clf()
    plt.plot(action_hist, c='r', alpha=0.3)
    plt.plot(gaussian_filter1d(action_hist, sigma=5), c='r', label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Actions')
    plt.title('Branching DDQN: {}'.format(args.env))
    plt.savefig(os.path.join(path, 'actions.png'))


class Config:

    def __init__(self,
                 epsilon_start=1.,
                 epsilon_final=0.05,
                 epsilon_decay=8000,
                 gamma=0.99,
                 lr=1e-4,
                 target_net_update_freq=1_000,
                 memory_size=100_000,
                 batch_size=128,
                 learning_starts=5_000,
                 max_frames=500_000,
                 bins=20,
                 Ntest=1000,
                 fname="sp500_closefull.csv",
                 furl="https://lazyprogrammer.me/course_files/sp500_closefull.csv",
                 aktie="AAPL",
                 aktie_price="PRICE",
                 spy="SPY",
                 initial_cash=10_000,
                 low=-100,
                 high=100,
                 buy_and_hold=False
                 ):

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon_by_frame = lambda frame: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * frame / self.epsilon_decay)
        self.gamma = gamma
        self.lr = lr
        self.target_net_update_freq = target_net_update_freq
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.max_frames = max_frames
        self.bins = bins
        self.Ntest = Ntest
        self.fname = fname
        self.furl = furl
        self.aktie = aktie
        self.aktie_price = aktie_price
        self.spy = spy
        self.initial_cash = initial_cash
        self.low = low
        self.high = high
        self.buy_and_hold = buy_and_hold


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = [] 
        dones = []

        for b in batch: 
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
