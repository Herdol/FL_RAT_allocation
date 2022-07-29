'''
The main function to be called

Info and update notes will be added. 
# Update 23/06/22 - 27/06/22
Weighted aggregation for FL.
# Update 27/06/22
Meta learning for different QoS metrics are added.
'''
from asyncio import Task
from re import X
from tokenize import Double
from turtle import distance

import gym

import numpy as np
import torch
import torch.optim as optim
import os
import wandb
from collections import OrderedDict, defaultdict
from copy import deepcopy
import time
import cProfile
## global parameters ## 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
wandb_USAGE_FLAG = True
device = 'cpu'

hyperparameters = dict(
    BATCH_SIZE = 40,
    GAMMA = 0.9738,
    EPS_START = 0.9514,
    EPS_END = 0.2531,
    EPS_DECAY = 3900,
    dropout = 0.5,
    channels_one = 20,
    channels_two = 51,
    learning_rate = 0.0005,
    episodes = 20,
    adapt_phase = 10,
    Replay_memory = 2000,
    FL_cycles=5,
    num_workers=4,
    Beta=0.2)

Configs=hyperparameters
if wandb_USAGE_FLAG == True:
    ## Change this section for your own account ## 
    wandb.init(project="FL_RAT_env_OOP_v02",config=hyperparameters,name="FedMeta", entity="herdol")
    config = wandb.config

from train_utils import Heuristic_select_action, ReplayMemory, select_action, optimize_model, DQN
from federated_learning import Initial_Broadcast, Register_update, FedAvg, Broadcast, FedRL
from utilities import logger
  
def train(model,widx,Fl_cycle,algorithm,target,ML=True):
    """
    Requires model to train, and index of widx for logging
    """
    #model.buffer_empty()
    Beta = config.Beta
    memory=ReplayMemory()
    Logger= logger(wandb_USAGE_FLAG)
    task=0
    episodes = config.episodes
    env.reset_history()
    if algorithm == 'single' or algorithm == 'reptile':
        episodes = config.episodes * config.FL_cycles
    for ep in range(episodes):
        obs = env.reset()
        total_reward=0
        
        if algorithm == 'fedavg' or algorithm == 'reptile':   
            reptile_weights_before = deepcopy(model.state_dict())
            # Not sure If optimizer should be done in reptile as well.
            #optimizer_before= deepcopy(optimizer.state_dict())

        if algorithm== 'fedavg' or algorithm == 'single' or algorithm == 'reptile':
            Training=1
                 
        if algorithm == 'fedavg_val' or algorithm == 'single_val' or algorithm == 'reptile_val' or algorithm == 'heuristic':
            task = 4
            Training=1
        
        reward_list=[]
        metric_list=[]
        t= np.random.randint(0,115)
        T=t
        steps=0
        sim_start = time.time()
        
        while t<T+10:
            start_time = env.time
            state=obs
            if algorithm== 'heuristic':
                action = Heuristic_select_action(state,env)
                next_state, reward, dones, info= env.step(action)   
            else:
                action = select_action(state,model,n_actions,device,Training,steps)
                env.meta_task(task)       
                next_state, reward, dones, info= env.step(action.item())# Task should be given ni step ...           
            reward = torch.tensor([reward], device=device)
            reward_list.append(reward[0])          
            total_reward+=reward
            
            memory.add([state, action, next_state, total_reward]) # Total reward or current reward?
            obs=next_state
            if algorithm!= 'heuristic':
                optimize_model(memory,policy_net=model,target_net=target, optimizer=optimizer)
                
            job_history=env.history
            end_time = env.time           
            time_elapsed = end_time - start_time
            t += max(time_elapsed, 0)
            steps+=1
        train_time=time.time()-sim_start
      
        
        if algorithm == 'fedavg' or algorithm == 'reptile':
            task += 1
            reptile_weights_after = deepcopy(model.state_dict())
            #optimizer_after= deepcopy(optimizer.state_dict())
            if task %5 == 4 :
                # meta learning (Reptile)
                model.load_state_dict({name : 
                        reptile_weights_before[name] + (reptile_weights_after[name] - reptile_weights_before[name]) * Beta 
                        for name in reptile_weights_before})
                '''optimizer.load_state_dict({name : 
                        optimizer_before[name] + (optimizer_after[name] - optimizer_before[name]) * Beta 
                        for name in optimizer_before})'''
                # Reset to first task
                task=0
        metrics= Logger.log(job_history,episode=ep+Fl_cycle*config.episodes,alg=algorithm,widx=widx,sim_timer= train_time, task=task) 
        metric_list.append(metrics)
        # Metrics can be used to print Logger class based variables. No need to use.  
    return metric_list
        
            
            
if __name__ == "__main__":
    np.random.seed(61)
    print("{} Device selected".format(device))
    num_workers=config.num_workers
    # Environment
    env=gym.make('gym_dataCachingCoding1:dataCachingCoding-v0')
    # Model related definitions
    n_actions = env.action_space.n
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    Initial_model = DQN(n_actions).to(device)
    memory = ReplayMemory(config.Replay_memory)
    Initial_policy_net=deepcopy(policy_net.state_dict())
    Initial_model.load_state_dict({name : 
                    Initial_policy_net[name] for name in Initial_policy_net})
    #model = DQN()
    #env=gym.make('gym_dataCachingCoding1:dataCachingCoding-v0')
    algorithm='fedavg'
    
    model_dict=defaultdict(list)
    ### Train
    Training = 1
    # Fed broadcast
    
    PATH_model='./models/global_model.pt'
    torch.save(policy_net, PATH_model)
    optimizer = optim.RMSprop(policy_net.parameters(),lr=config.learning_rate)

    model_dict=Initial_Broadcast(policy_net,Initial_policy_net,num_workers,model_dict,optimizer)
    Initial_checkpoint = { 
        'epoch': 0,
        'model': policy_net.state_dict(),
        'optimizer': optimizer.state_dict()}
    checkpoint=[]
    for widx in range(num_workers): 
        checkpoint.append({ 
            'model{}'.format(widx): policy_net.state_dict(),
            'optimizer{}'.format(widx): optimizer.state_dict()})
    checkpoint.append({'epoch': 0})
    torch.save(checkpoint, './models/checkpoint.pth')
    torch.save(Initial_checkpoint,'./models/initial_checkpoint.pth')
    Begin_federated_sim=time.time()
    for fl in range(config.FL_cycles):


        for widx in range(num_workers):
            ## Creating new env to wipe history of environment.
            env=gym.make('gym_dataCachingCoding1:dataCachingCoding-v0')
            memory = ReplayMemory(config.Replay_memory)
            print('New environment has been created.')
            ## Downloading model for worker widx ##
            checkpoint_dict=torch.load('./models/checkpoint.pth')
            ## Since each worker will train the same model no need to keep all of them for memory efficiency.
            policy_net.load_state_dict(checkpoint_dict[widx]['model{}'.format(widx)])
            optimizer.load_state_dict(checkpoint_dict[widx]['optimizer{}'.format(widx)])
            policy_net.train()
            print('Globel model {} for worker {} is loaded'.format(checkpoint_dict[num_workers]['epoch'],widx)) # Last index of checkpoint is epoch register.
            # Profiling training
            #cProfile.run('train(policy_net,widx,Fl_cycle=fl,algorithm=algorithm,target=target_net)', './Profiles/Profiling_worker_{}.dat'.format(widx))
            
            # Train worker (Time for training is increasing with each episode passed)
            metrics = train(policy_net,widx,Fl_cycle=fl,algorithm=algorithm,target=target_net)
            # Upload model
            model_dict['models'][widx]=Register_update(model_dict,widx,policy_net)    
            model_dict['targets'][widx]=deepcopy(policy_net.state_dict())
            model_dict['optimizer'][widx]=deepcopy(optimizer.state_dict())
            model_dict['performance'][widx] = metrics
            
            
            #wandb.watch(policy_net,log="all",log_freq= 1)
            #wandb.watch(Initial_model,log="all",log_freq= 1)
            obs = env.reset()
        
        
        #### Aggregation scheme will be chosen in this section ####
        # Aggregate all models and broadcast  
        ## FedAvg - Simply averaging all updates for global model
        Global_model = FedAvg(model_dict['models'],num_workers) # Calculated global model weights

        ## FednRL - Takes "n" most succesful models for aggregation
        #Global_model = FedRL(model_dict,num_workers,n=3)

        policy_net.load_state_dict(Global_model)
        checkpoint_dict=[]
        for widx in range(num_workers): 
            checkpoint_dict.append({ 
                'model{}'.format(widx): policy_net.state_dict(),
                'optimizer{}'.format(widx): optimizer.state_dict()})
        checkpoint_dict.append({'epoch': fl+1})
        torch.save(checkpoint_dict, './models/checkpoint.pth')
        model_dict = Broadcast(Global_model,model_dict,num_workers)
    End_of_sim=time.time()
    print('Federated learning training is done in {:.1f} sec'.format((End_of_sim-Begin_federated_sim)))
    ### Validate
    memory = ReplayMemory(config.Replay_memory)
    algorithm = 'fedavg_val'
    widx=0
    fl=0
    env=gym.make('gym_dataCachingCoding1:dataCachingCoding-v0')
    #config.episodes=10

    # Create Checkpoint to try this model "config.adapt_phase" times
    Fed_checkpoint = { 
        'epoch': 0,
        'model': policy_net.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(Fed_checkpoint, './models/Trained_Global.pth')
    
    for i in range(config.adapt_phase):
        Global_model=torch.load('./models/Trained_Global.pth')
        policy_net.load_state_dict(Global_model['model'])
        target_net.load_state_dict(Global_model['model'])
        optimizer.load_state_dict(Global_model['optimizer'])
        model= policy_net
        model.train()
        metrics = train(model,widx=i,Fl_cycle=fl,algorithm=algorithm,target=target_net)
    
    algorithm = 'single'
    env=gym.make('gym_dataCachingCoding1:dataCachingCoding-v0')
    ### Train
    memory = ReplayMemory(config.Replay_memory)
    Initial_checkpoint=torch.load('./models/Initial_checkpoint.pth')
    ## Since each worker will train the same model no need to keep all of them for memory efficiency.
    policy_net.load_state_dict(Initial_checkpoint['model'])
    target_net.load_state_dict(Initial_checkpoint['model'])
    optimizer.load_state_dict(Initial_checkpoint['optimizer'])
    widx=0
    fl=0
    model= policy_net
    model.train()
    metrics = train(model,widx,Fl_cycle=fl,algorithm=algorithm,target=target_net)
    
    ### Validate
    memory = ReplayMemory(config.Replay_memory)
    algorithm = 'single_val'
    env=gym.make('gym_dataCachingCoding1:dataCachingCoding-v0')

    
    Single_checkpoint = { 
        'epoch': 0,
        'model': policy_net.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(Single_checkpoint, './models/Trained_Single.pth')
    
    for i in range(config.adapt_phase):
        Single_model=torch.load('./models/Trained_Single.pth')
        policy_net.load_state_dict(Single_model['model'])
        target_net.load_state_dict(Single_model['model'])
        optimizer.load_state_dict(Single_model['optimizer'])
        model= policy_net
        model.train()
        
        metrics = train(model,widx=i,Fl_cycle=fl,algorithm=algorithm,target=target_net)

    ### Reptile algorithm 
    
    algorithm = 'reptile'
    env=gym.make('gym_dataCachingCoding1:dataCachingCoding-v0')
    ### Train
    memory = ReplayMemory(config.Replay_memory)
    Initial_checkpoint=torch.load('./models/Initial_checkpoint.pth')
    ## Since each worker will train the same model no need to keep all of them for memory efficiency.
    policy_net.load_state_dict(Initial_checkpoint['model'])
    target_net.load_state_dict(Initial_checkpoint['model'])
    optimizer.load_state_dict(Initial_checkpoint['optimizer'])
    widx=0
    fl=0
    model= policy_net
    model.train()
    metrics = train(model,widx,Fl_cycle=fl,algorithm=algorithm,target=target_net)
    
    ### Validate
    memory = ReplayMemory(config.Replay_memory)
    algorithm = 'reptile_val'
    env=gym.make('gym_dataCachingCoding1:dataCachingCoding-v0')
    Reptile_checkpoint = { 
        'epoch': 0,
        'model': policy_net.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(Reptile_checkpoint, './models/Trained_Reptile.pth')

    for i in range(config.adapt_phase):
        Reptile_model=torch.load('./models/Trained_Reptile.pth')
        policy_net.load_state_dict(Reptile_model['model'])
        target_net.load_state_dict(Reptile_model['model'])
        optimizer.load_state_dict(Reptile_model['optimizer'])
        model= policy_net
        model.train()
        metrics = train(model,widx=i,Fl_cycle=fl,algorithm=algorithm,target=target_net)

    
    algorithm = 'heuristic'
    memory = ReplayMemory(config.Replay_memory)
    env=gym.make('gym_dataCachingCoding1:dataCachingCoding-v0')
    # Model, widx, FL_cycle and target won't be usedf in Heuristic.
    for widx in range(config.adapt_phase):
        metrics = train(model,widx,Fl_cycle=fl,algorithm=algorithm,target=target_net)
