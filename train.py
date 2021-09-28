import utils
import torch.optim as optim
import numpy as np
from agent import Agent
from tqdm import tqdm
from utils import ExperienceReplayMemory
from dataset import Dataset
from environment import Environment


def train():
    args = utils.arguments()
    config = utils.Config()
    bins = config.bins

    dataset = Dataset()
    train_set, _ = dataset.get_train_test()
    env = Environment(train_set)

    memory = ExperienceReplayMemory(config.memory_size)
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0], config)
    adam = optim.Adam(agent.q.parameters(), lr=config.lr)

    state = env.reset()
    episode_reward = 0.
    recap = []

    portfolio_hist = []
    cash_hist = []
    stock_hist = []
    action_hist = []
    action_frame = []

    p_bar = tqdm(total=config.max_frames)
    for frame in range(config.max_frames):

        epsilon = config.epsilon_by_frame(frame)

        if np.random.random() > epsilon:
            action = agent.get_action(state)
        else:
            action = np.random.randint(0, bins, size=env.action_space.shape[0])

        action_frame.append(action)
        next_state, reward, done = env.step(action)
        episode_reward += reward

        if done:

            recap.append(episode_reward)
            portfolio_hist.append(env.portfolio_value)
            cash_hist.append(env.cash)
            stock_hist.append(env.stock)
            action_hist.append(np.mean(action_frame))

            p_bar.set_description('Portfolio: {:.3f}'.format(env.portfolio_value) +
                                  ' Cash: {:.3f}'.format(env.cash) +
                                  ' Stock: {:.3f}'.format(env.stock))
            episode_reward = 0.

            next_state = env.reset()

        memory.push((state.reshape(-1).numpy().tolist(), action, reward, next_state.reshape(-1).numpy().tolist(), 0. if done else 1.))
        state = next_state

        p_bar.update(1)

        if frame > config.learning_starts:
            agent.update_policy(adam, memory, config)

        if frame % 1000 == 0:
            utils.save(agent, (recap, portfolio_hist, cash_hist, stock_hist, action_hist), args)

    p_bar.close()


if __name__ == "__main__":
    train()
