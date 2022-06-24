'''
The main function to be called

Info and update notes will be added. 

'''
from re import X
from tokenize import Double
from turtle import distance

import gym
import math
import random
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
#from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os
import wandb
from collections import OrderedDict, defaultdict
from copy import deepcopy
import time
## global parameters ## 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
wandb_USAGE_FLAG = True
device = 'cpu'
global hyperparameters
hyperparameters = dict(
    BATCH_SIZE = 200,
    GAMMA = 0.9738312485226788,
    EPS_START = 0.9514174619459358,
    EPS_END = 0.2531408887479644,
    EPS_DECAY = 3900,
    dropout = 0.5,
    channels_one = 20,
    channels_two = 51,
    learning_rate = 0.0008426492730994772,
    episodes = 80,
    Replay_memory = 8837,
    FL_cycles=3,
    num_workers=5)

Configs=hyperparameters
if wandb_USAGE_FLAG == True:
    #wandb.init(project="FL_RAT_env_OOP_v01",config=hyperparameter_defaults,name="{} workers {} FL cycle {} ep".format(config['num_workers'],
    #                                            #config['FL_cycles'],config['num_episodes']), entity="herdol")
    
    wandb.init(project="FL_RAT_env_OOP_sweep_v01",config=hyperparameters,name="Sweep params", entity="herdol")
    config = wandb.config

from train_utils import ReplayMemory, select_action, optimize_model, DQN
from federated_learning import Initial_Broadcast, Register_update, FedAvg, Broadcast
from utilities import logger
  
def train(model,widx,Fl_cycle,algorithm,target):
    """
    Requires model to train and index of widx for logging
    """
    #model.buffer_empty()
    memory=ReplayMemory()
    total_reward=0
    Logger= logger(wandb_USAGE_FLAG)
    for ep in range(config.episodes):
        s = env.reset()
        obs = env.reset()
        if algorithm== 'fedavg' or 'single':
            Training=1
        reward_list=[]
        t= np.random.randint(0,115)
        T=t
        steps=0
        sim_start = time.time()
        while t<T+10:
            start_time = env.time
            state=obs
            action = select_action(state,model,n_actions,device,Training,steps)     
            next_state, reward, dones, info= env.step(action.item())            
            reward = torch.tensor([reward], device=device)
            reward_list.append(reward[0])          
            total_reward+=reward
            
            memory.add([state, action, next_state, total_reward]) # Total reward or current reward?
            obs=next_state
            if algorithm== 'fedavg' or 'single':
                optimize_model(memory,policy_net=model,target_net=target, optimizer=optimizer)
                
            job_history=env.history
            end_time = env.time           
            time_elapsed = end_time - start_time
            t += max(time_elapsed, 0)
            steps+=1
        train_time=time.time()-sim_start
        metrics= Logger.log(job_history,episode=ep+Fl_cycle*config.episodes,alg=algorithm,widx=widx,sim_timer= train_time)
            
            
if __name__ == "__main__":
    np.random.seed(1)
    print("{} Device selected".format(device))
    num_workers=config.num_workers
    env=gym.make('gym_dataCachingCoding1:dataCachingCoding-v0')
    n_actions = env.action_space.n
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    Initial_model = DQN(n_actions).to(device)
    optimizer = optim.RMSprop(policy_net.parameters(),lr=config.learning_rate)
    memory = ReplayMemory(config.Replay_memory)
    Initial_policy_net=deepcopy(policy_net.state_dict())
    Initial_model.load_state_dict({name : 
                    Initial_policy_net[name] for name in Initial_policy_net})
    model = DQN()
    algorithm='fedavg'
    model_dict=defaultdict(list)
    ### Train
    Training = 1
    # Fed broadcast
    model_dict=Initial_Broadcast(policy_net,Initial_policy_net,num_workers,model_dict)
    Begin_federated_sim=time.time()

    ## Downloading model for worker widx ##

    policy_net.load_state_dict({name : 
            model_dict['models'][0][name] for name in model_dict['models'][0]})
    target_net.load_state_dict({name : 
            model_dict['targets'][0][name] for name in model_dict['targets'][0]})
    model= policy_net
    train(model,0,Fl_cycle=0,algorithm=algorithm,target=target_net)



    End_of_sim=time.time()
    print('Federated learning training is done in {:.1f} sec'.format((End_of_sim-Begin_federated_sim)))
    