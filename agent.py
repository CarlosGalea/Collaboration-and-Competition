#Import Packages
from collections import namedtuple, deque
from model import Actor, Critic

import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.       # noise weight decay

N_LEARN_UPDATES = 10
N_LEARN_TIMESTEPS = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """
    Interacts with and learns from the environment
    """

    def __init__(self, state_size, action_size, random_seed):
        """
        Initialize parameters and build an Agent model
        
        state_size: Environment's state size
        action_size: Agents' action size
        seed: Random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.seed = random.seed(random_seed)
        self.i_learn = 0

        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise Process
        self.noise = OUNoise(2 * action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        print("GPU: {}".format(torch.cuda.is_available()))

    def step(self, states, actions, actions_player2, rewards, next_states, next_states_player2, dones, timestep):
        """
        Save experience in replay memory, and use random sample from buffer to learn
        """
        
        # Save Memory
        for state, action, action_player2, reward, next_state, next_state_player2, done \
                in zip(states, actions, actions_player2, rewards, next_states, next_states_player2, dones):
            self.memory.add(state, action, action_player2, reward, next_state, next_state_player2, done)
        
        if timestep % N_LEARN_TIMESTEPS != 0:
            return
        
        #IF enough samples in memory
        if len(self.memory) > BATCH_SIZE:
            for i in range(N_LEARN_UPDATES):
                #Load sample of tuples from memory
                experiences = self.memory.sample()

                #Learn from a randomly selected sample
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True, noise=1.0):
        """
        Returns action for given state as per current policy
        """
        
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            
        self.actor_local.train()
        
        if add_noise:
            action += noise * self.noise.sample().reshape((-1, 2))
            
        #Return action
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch of experience tuples
        
        experiences: tuple of s, a, a_2, r, s', s'_2, done
        gamma: discount factor
        """
        
        states, actions, actions_player2, rewards, next_states, next_states_player2, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions (also for the other player) and Q values from target models
        actions_next = self.actor_target(next_states)
        actions_next_player2 = self.actor_target(next_states_player2)        
        Q_targets_next = self.critic_target(next_states, actions_next, actions_next_player2)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Current expected Q-values
        Q_expected = self.critic_local(states, actions, actions_player2)
        
        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred, actions_player2).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters
        
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau: Interpolation parameter
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """
    Ornstein-Uhlenbeck process
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        Initialize parameters and noise process
        """
        
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu)
        """
        
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Update internal state and return it as a noise sample
        """
        
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples
    """

    def __init__(self, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object
        
        buffer_size: Maximum size of buffer
        batch_size: Size of each training batch
        seed: Random seed
        """
        
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "action_player2", "reward", "next_state", "next_state_player2", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, action_player2, reward, next_state, next_state_player2, done):
        """
        Add a new experience to memory
        """
        
        e = self.experience(state, action, action_player2, reward, next_state, next_state_player2, done)
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        """
        
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        actions_player2 = torch.from_numpy(np.vstack([e.action_player2 for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        next_states_player2 = torch.from_numpy(np.vstack([e.next_state_player2 for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, actions_player2, rewards, next_states, next_states_player2, dones

    def __len__(self):
        """
        Return the current size of internal memory
        """
        
        return len(self.memory)