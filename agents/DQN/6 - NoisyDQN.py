import numpy as np
import torch
import cpprb
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
LR = 10**-3
GAMMA = 0.99
BATCH_SIZE = 64


class NoisyLinear(torch.nn.Module):
    """Implements Noisy Linear Layers"""

    def __init__(self, in_features, out_features, std_init=0.5, type='factorized'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self._type = type

        self.mu_w = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.mu_b = torch.nn.Parameter(torch.Tensor(out_features))
        self.sigma_w = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_b = torch.nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('eps_w', torch.Tensor(out_features, in_features))
        self.register_buffer('eps_b', torch.Tensor(out_features))

        if self._type == 'factorized':
            self.f = lambda x: np.sign(x) * np.sqrt(np.abs(x))  # function from paper.

        self.reset_parameters()
        self.reset_noise()

    def reset_noise(self):
        if self._type == 'factorized':
            eps_input = self.f(np.random.normal(size=self.in_features)).astype(np.float32)
            eps_output = self.f(np.random.normal(size=self.out_features)).astype(np.float32)
            self.eps_w = torch.tensor(np.outer(eps_output, eps_input)).to(device)
            self.eps_b = torch.tensor(eps_output).to(device)
        elif self._type == 'independent':
            self.eps_w = torch.tensor(
                np.random.normal(size=(self.out_features, self.in_features)).astype(np.float32)
            ).to(device)
            self.eps_b = torch.tensor(
                np.random.normal(size=self.out_features).astype(np.float32)
            ).to(device)
        else:
            raise ValueError('type must be either `factorized` or `independent`.')

    def reset_parameters(self):
        if self._type == 'factorized':
            mu_range = 1 / np.sqrt(self.in_features)
            sigma_w_val = 0.5 / self.sigma_w.size(1)
            sigma_b_val = 0.5 / self.sigma_b.size(0)
        elif self._type == 'independent':
            mu_range = np.sqrt(3 / self.in_features)
            sigma_w_val = 0.017
            sigma_b_val = 0.017
        else:
            raise ValueError('type must be either `factorized` or `independent`.')

        self.mu_w.data.uniform_(-mu_range, mu_range)
        self.mu_b.data.uniform_(-mu_range, mu_range)
        self.sigma_w.data.fill_(sigma_w_val)
        self.sigma_b.data.fill_(sigma_b_val)

    def forward(self, t):
        weights = (self.mu_w + self.sigma_w * self.eps_w)
        biases = (self.mu_b + self.sigma_b * self.eps_b)
        return torch.nn.functional.linear(t, weights, biases)


class Network(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.noisy1 = NoisyLinear(in_size, 256)
        self.noisy2 = NoisyLinear(256, 64)
        self.noisy3 = NoisyLinear(64, out_size)

        # fc stands for fully-connected
        self.fc = torch.nn.Sequential(
            self.noisy1,
            torch.nn.ReLU(),
            self.noisy2,
            torch.nn.ReLU(),
            self.noisy3
        )

    def forward(self, t):
        if type(t) == np.ndarray:
            t = torch.tensor(t, dtype=torch.float32).to(device)

        return self.fc(t)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
        self.noisy3.reset_noise()


class NoisyDQN:

    def __init__(self, env):
        self.name = '6 - NoisyDQN'

        # Environment related logic
        self.env = env
        self.observation_size = np.prod(self.env.observation_space.shape)
        self.action_space = self.env.action_space.n

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

    def choose_action(self, state):
        return self.policy(state).argmax().item()

    def step(self, state):
        # Choose action
        action = self.choose_action(state)

        # Update environment
        state_, reward, done, _ = self.env.step(action)

        # Reset network noise
        self.policy.reset_noise()

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
        scores = np.zeros(num_steps)
        progress_bar = tqdm.tqdm(range(num_steps), unit='step')
        progress_bar.set_description(progress_prefix)
        idx_left = 0
        score = 0
        state = self.env.reset()
        for idx in progress_bar:
            # Make a step
            state, reward, done = self.step(state)
            score += reward

            # Train
            loss = self.calculate_loss()
            self.update_model(loss)

            if done:
                progress_bar.set_postfix({
                    'reward': score
                })

                scores[idx_left : idx] = score
                idx_left = idx
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
    train.train_agent(NoisyDQN)