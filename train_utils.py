## Update notes ##
# W&B and Sweep module implementation for hyperparameter tuning. 17/03/22
from tokenize import Double
"""from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2"""

import math
import random
import numpy as np
from collections import namedtuple, deque



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os
import wandb
#from Sweep_training import hyperparameters
#import Sweep_training
os.environ['KMP_DUPLICATE_LIB_OK']='True'
## Example W%B commands 
wandb_USAGE_FLAG = True
device = 'cpu'

#hyperparameters = Sweep_training.hyperparameters
hyperparameters = dict(
    BATCH_SIZE = 50,
    GAMMA = 0.9738312485226788,
    EPS_START = 0.9514174619459358,
    EPS_END = 0.2531408887479644,
    EPS_DECAY = 3900,
    dropout = 0.5,
    channels_one = 20,
    channels_two = 51,
    learning_rate = 0.0008426492730994772,
    episodes = 10,
    Replay_memory = 1000,
    FL_cycles=3,
    num_workers=10)


Configs=hyperparameters
if wandb_USAGE_FLAG == True:
    #wandb.init(project="FL_RAT_env_OOP_v01",config=hyperparameter_defaults,name="{} workers {} FL cycle {} ep".format(config['num_workers'],
    #                                            #config['FL_cycles'],config['num_episodes']), entity="herdol")
    
    wandb.init(project="FL_RAT_env_OOP_v01",config=hyperparameters,name="Initial trials", entity="herdol")
    config = wandb.config

random_seeds = np.random.randint(500, size = 5)

# Replay Memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#env = gym.make('gym_dataOffload:dataCache-v0')
class ReplayMemory(object):
    def __init__(self, max_size = config.Replay_memory):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def wipe_memory(self,max_size=config.Replay_memory):
        self.buffer = deque(maxlen=max_size)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]


class DQN(nn.Module):

    def __init__(self, learning_rate=config.learning_rate, state_size=47, 
                 action_size=11, hidden_size=config.channels_two, hidden_size_1=config.channels_two, hidden_size_2=config.channels_one, batch_size=config.BATCH_SIZE, gamma=config.GAMMA):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        #self.fc2 = nn.Linear(hidden_size, hidden_size_1)
        self.fc3 = nn.Linear(hidden_size_1, hidden_size_2)
        #self.fc4 = nn.Linear(hidden_size_2, hidden_size_2)
        self.output = nn.Linear(hidden_size_2, action_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #x = torch.from_numpy(x,dtype=Double)
        global device
        x = torch.tensor(x).float()
        x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.fc2(x)
        #x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        #x = self.fc4(x)
        #x = F.relu(x)
        output=self.output(x)
        return output


# Utilities
BATCH_SIZE = config.BATCH_SIZE
GAMMA = config.GAMMA
EPS_START =config.EPS_START 
EPS_END = config.EPS_END
EPS_DECAY =config.EPS_DECAY 


## Get number of actions from gym action space
def select_action(state,policy_net,n_actions,device,Training,steps_done):
    #global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if Training==1:
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return torch.argmax(policy_net(state))
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
         with torch.no_grad():
            return torch.argmax(policy_net(state)) # was policy_net


episode_durations = []

def optimize_model(memory,policy_net,target_net,optimizer):
    
    if len(memory.buffer) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in torch.tensor(np.array(batch.next_state))
                                                if s is not None])
    state_batch = torch.tensor(batch.state).to(device)
    action_batch = torch.tensor(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.view(BATCH_SIZE,1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.view(BATCH_SIZE,47)).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def Heuristic_select_action(state,env):
    view_buffer = state[30:45] # Extract buffer info from state
    
    destinations = view_buffer[1:-1:3] # Extract destinations from buffer
    deadlines= view_buffer[2::3]
    #dest_line=np.argmin(deadlines) # Choose the least remaining time in deadlines
    #dest = destinations[dest_line] * 4 # denormalize veh index
    destinations=destinations * 4
    dest = np.random.choice(destinations,size=1) # Choose random action from destinations
    LTE_available = state[45] 
    mmWave_available = state[46]
    Distances=env.vehicle_positions(scale = False)
    Distances=Distances-500 # distance to the center
    Dist=[]
    for i in range(5):
        x,y=Distances[i][0],Distances[i][1]
        Dist.append(np.sqrt(np.sum((x - y) ** 2, axis=0)))
    # If mmWave available, select it. Because it has more datarate.
    chosen_veh=999
    if mmWave_available == 1:
        for veh in destinations:
            if Dist[int(veh)]<=200:
                action = veh*2 + 1
                chosen_veh=veh
        if chosen_veh>5 and LTE_available == 1: # If all vehicles are out of mmWave coverage # Not clever conditions
            action = dest *2
        elif chosen_veh>5 and LTE_available == 0: 
            action = 10
    elif mmWave_available == 0 and LTE_available == 1 : # the mmWave max range is 200m
        action = dest *2
    else:
        action = 10
        # action = np.random.randint(1,10)
    return int(action)