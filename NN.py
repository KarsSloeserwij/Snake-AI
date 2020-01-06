import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
  def __init__(self, learning_rate, input_dim, layer1_dim, layer2_dim, num_actions):
    super(QNetwork, self).__init__()
    self.input_dim = input_dim
    self.learning_rate = learning_rate
    self.num_actions = num_actions

    self.layer1_dim = layer1_dim
    self.layer2_dim = layer2_dim

    self.linear_layer1 = nn.Linear(*self.input_dim, self.layer1_dim)
    self.linear_layer2 = nn.Linear(self.layer1_dim, self.layer2_dim)
    self.linear_layer3 = nn.Linear(self.layer2_dim, self.num_actions)

    self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
    self.loss = nn.MSELoss()
    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, observation):
    state = T.Tensor(observation).to(self.device)
    x = F.relu(self.linear_layer1(state))
    x = F.relu(self.linear_layer2(x))
    actions = self.linear_layer3(x)
    return actions


class Agent(object):
    def __init__(self, gamma, epsilon, learning_rate, input_dim, batch_size,
                n_actions, max_mem_size=1000000, eps_end=0.01, eps_dec=0.999):
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.mem_counter = 0
        self.action_space = [i for i in range(n_actions)]

        self.Qeval = QNetwork(learning_rate=learning_rate, input_dim=input_dim,
                            layer1_dim=32, layer2_dim=16, num_actions=n_actions)

        #memory
        self.mem_size = max_mem_size
        self.state_memory = np.zeros((self.mem_size, *input_dim))
        self.new_state_memory = np.zeros((self.mem_size, *input_dim))
        self.action_memory = np.zeros((self.mem_size, self.n_actions),
                                      dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, new_state, terminal):
        index = self.mem_counter % self.mem_size
        #print(index)
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - terminal
        self.new_state_memory[index] = new_state
        self.mem_counter += 1

    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.Qeval.forward(observation)
            action = T.argmax(actions).item()

        return action


    def learn(self):
        if self.mem_counter > self.batch_size:
            self.Qeval.optimizer.zero_grad()

            max_mem = self.mem_counter if self.mem_counter < self.mem_size else self.mem_size
            batch = np.random.choice(max_mem, self.batch_size)

            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.bool)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            terminal_batch = self.terminal_memory[batch]
            new_state_batch = self.new_state_memory[batch]

            reward_batch = T.Tensor(reward_batch).to(self.Qeval.device)
            terminal_batch = T.Tensor(terminal_batch).to(self.Qeval.device)

            q_eval = self.Qeval.forward(state_batch).to(self.Qeval.device)
            q_target = q_eval.clone()
            q_next = self.Qeval.forward(new_state_batch).to(self.Qeval.device)

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            target_update = reward_batch + \
                                    self.gamma*T.max(q_next, dim=1)[0]*terminal_batch

            for i in range(len(batch_index)):
                temp = q_target[batch_index[i], action_indices[i]]
                q_target[batch_index[i], action_indices[i]] = target_update[i]

            self.epsilon = self.epsilon*self.eps_dec if self.epsilon > self.eps_end else self.eps_end
            loss = self.Qeval.loss(q_target, q_eval).to(self.Qeval.device)
            loss.backward()
            self.Qeval.optimizer.step()
