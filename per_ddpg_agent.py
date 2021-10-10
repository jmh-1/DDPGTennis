import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
import sum_tree

BUFFER_SIZE = int(2 ** 16)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
UPDATE_EVERY = 4        # how often to update the network
UPDATE_ITERATIONS = 4
EPSILON_START = .1
EPSILON_END = .01
EPSILON_DECAY = 1e-5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.a = .5
        self.b = .5
        self.a_init = self.a
        self.b_init = self.b

        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = PorportionalPriorityReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.episodes = 1

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            self.memory.add(s, a, r, ns, d)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for i in range(UPDATE_ITERATIONS):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            sample = self.noise.sample()
            noise = 10*(sample * ((-1)**self.episodes)) / np.sqrt(self.episodes)
            action += noise
        self.epsilon = max(self.epsilon - EPSILON_DECAY, EPSILON_END)
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()
        self.episodes += 1
        self.a = max(self.a - BATCH_SIZE*self.a_init/2000, 0)
        self.b = min(self.b - BATCH_SIZE*self.b_init/2000, 1)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indexes, probs = experiences

        weights = ((1 / len(self.memory)) * (1 / probs))**self.b
        weights /= torch.max(weights)
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        error = (Q_targets - Q_expected).detach()
        for i in range(error.shape[0]):
            self.memory.update_priority(indexes[i].item(), error[i].item(), self.a)
        Q_targets *= weights
        Q_expected *= weights
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
#         torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -(weights * self.critic_local(states, actions_pred)).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma *  np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class PorportionalPriorityReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            a (float): value used to determine how much priority is used
        """
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = sum_tree.SumTree(buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
#         self.a = a
        self.count = 0
        # max priority should be higher than real priority so each element is 
        # likely to get picked at least once
        self.max_priority = 300
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.insert(self.max_priority, e)
        if self.count < self.buffer_size:
            self.count += 1

    def update_priority(self, idx, err, a):
        val = (.1 + abs(err))**a
        self.memory.update(idx, val)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = []
        indexes = []
        probs = []
        for i in range(self.batch_size):
            try:
                sampleVal = np.random.uniform(high=self.memory.total)
                idx = self.memory.find_val_idx(sampleVal)
                exp = self.memory.data[idx]
                experiences.append(exp)
                indexes.append(idx)
                probs.append(self.memory.get_val(idx) / self.memory.total)
#                 set_trace()
            except:
                e = sys.exc_info()[0]
                print(self.memory.total)
                print(e)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        indexes_d = torch.from_numpy(np.vstack(indexes).astype(np.uint8)).to(device)
        probs_d = torch.from_numpy(np.vstack(probs).astype(np.float)).float().to(device)
        return (states, actions, rewards, next_states, dones, indexes_d, probs_d)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.count;
