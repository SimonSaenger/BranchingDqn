from tqdm import tqdm
import torch
from agent import Agent
from dataset import Dataset
from environment import Environment
import utils
from utils import Config


def play():
    args = utils.arguments()
    config = Config()
    dataset = Dataset()
    train_set, test_set = dataset.get_train_test()
    environment = Environment(test_set)
    agent = Agent(environment.observation_space.shape[0], environment.action_space.shape[0], config)
    agent.q.load_state_dict(torch.load('./runs/{}/model_state_dict'.format(args.env)))

    for ep in tqdm(range(1)):
        state = environment.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(state)
            state, reward, done = environment.step(action)
            episode_reward += reward

        print('Ep reward: {:.3f}'.format(episode_reward))
        print('Portfolio: {:.3f}'.format(environment.portfolio_value))
        print('Cash: {:.3f}'.format(environment.cash))
        print('Stock: {:.3f}'.format(environment.stock))

    environment.close()


if __name__ == "__main__":
    play()
