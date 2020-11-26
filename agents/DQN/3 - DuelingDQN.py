# Simplifies learning

import numpy as np
import torch
import cpprb
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
LR = 10**-3
GAMMA = 0.99
BATCH_SIZE = 64


class Network(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
        )

        # Estimate value and advantage seperately
        self.value = torch.nn.Linear(64, 1)
        self.advantage = torch.nn.Linear(64, out_size)

    def forward(self, t):
        if type(t) == np.ndarray:
            t = torch.tensor(t, dtype=torch.float32).to(device)

        t = self.fc(t)

        # Combine value and advantage into a Q-value before returning
        advantage = self.advantage(t)
        t = self.value(t) + advantage - advantage.mean(-1, True)

        return t


class DuelingDQN:

    def __init__(self, env):
        self.name = '3 - DuelingDQN'

        # Environment related logic
        self.env = env
        self.observation_size = np.prod(self.env.observation_space.shape)
        self.action_space = self.env.action_space.n

        # Epsilon greedy parameters
        self.eps_min = 0.01
        self.eps_max = 1
        self.eps_dec = 5e-4
        self.eps_cnt = 0

        # Replay memory
        env_dict = {'state': {'shape': self.env.observation_space.shape},
                    'action': {'dtype': np.int64},
                    'reward': {},
                    'state_': {'shape': self.env.observation_space.shape},
                    'done': {'dtype': np.bool}}
        self.memory = cpprb.ReplayBuffer(10**6, env_dict)

        # Neural network
        self.policy = Network(in_size=self.observation_size, out_size=self.action_space).to(device)
        self.optim = torch.optim.Adam(params=self.policy.parameters(), lr=LR)
        self.loss = torch.nn.MSELoss()

    @property
    def eps(self):
        return self.eps_min + (self.eps_max - self.eps_min) * np.exp(-self.eps_cnt * self.eps_dec)

    def choose_action(self, state, eval=False):
        if eval or np.random.random() > self.eps:
            return self.policy(state).argmax().item()
        else:
            return self.env.action_space.sample()

    def step(self, state):
        # Choose action
        action = self.choose_action(state)

        # Update Epsilon
        self.eps_cnt += 1

        # Update environment
        state_, reward, done, _ = self.env.step(action)

        # Add to memory
        self.memory.add(state=state, state_=state_, reward=reward, action=action, done=done)

        # Return useful things
        return state_, reward, done

    def calculate_loss(self):
        batch = self.memory.sample(BATCH_SIZE)
        state = torch.tensor(batch['state']).to(device)
        action = torch.tensor(batch['action']).to(device)
        reward = torch.tensor(batch['reward']).to(device)
        state_ = torch.tensor(batch['state_']).to(device)
        done = torch.tensor(batch['done']).to(device)

        # Calculate the temporal difference
        Qval = self.policy(state).gather(1, action)
        Qval_ = self.policy(state_).max(dim=1, keepdim=True)[0]
        Qval_[done] = 0
        Qgoal = reward + GAMMA * Qval_

        return self.loss(Qval, Qgoal)

    def update_model(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self, num_steps, progress_prefix='', *args, **kwargs):
        # Initialize array to set arr and helper variable
        scores = np.zeros(num_steps)
        idx_left = 0

        # Initialize important variables
        score = 0
        max_reward = 0
        state = self.env.reset()

        # Construct a progress-bar
        progress_bar = tqdm.tqdm(range(num_steps), unit='step')
        progress_bar.set_description(progress_prefix)
        for idx in progress_bar:
            # Make a step
            state, reward, done = self.step(state)
            score += reward

            # Train
            loss = self.calculate_loss()
            self.update_model(loss)

            if done:
                # Update max-reward and progress-bar
                max_reward = max(score, max_reward)
                progress_bar.set_postfix({
                    'reward': score,
                    'epsilon': self.eps,
                    'max_reward': max_reward
                })

                # Set the score of all steps belonging to this episode to
                # the total received reward.
                scores[idx_left: idx] = score
                idx_left = idx

                # Reset basic variables
                score = 0
                state = self.env.reset()
        else:
            # Finish up training to determine last set of arr
            while not done:  # noqa
                state, reward, done = self.step(state)
                score += reward
            scores[idx_left:] = score

        return scores

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.policy.load_state_dict(state_dict)


if __name__ == '__main__':
    import utils.train as train
    train.train_agent(DuelingDQN)