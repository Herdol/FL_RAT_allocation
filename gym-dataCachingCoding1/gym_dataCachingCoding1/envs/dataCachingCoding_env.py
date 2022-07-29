from urllib.parse import scheme_chars
import numpy as np
import scipy.io
import scipy.special
import scipy.stats
from gym_dataCachingCoding1.envs.simulation_entities import *
import gym
from gym import error, spaces, utils

class Environment_Map(gym.Env):

    def __init__(self, vehicle_number=5):
        self.history = {'completed_jobs':0, 'completed_jobs_size':[], 'exceed_deadline':0, 'exceed_deadline_size':[], 'latency':0.0,'throughput':0.0}
        self.time = 0.0
        self.grid = (1000.0,1000.0)
        self.max_speed = 8.0
        self.max_coord = 1000.0
        self.start_points = {0:{'pos':np.array([500.0,0.0,0.0]),'vel':np.array([0.0,8.0,0.0])},
                             1:{'pos':np.array([0.0,500.0,0.0]),'vel':np.array([8.0,0.0,0.0])},
                             2:{'pos':np.array([500.0,1000.0,0.0]),'vel':np.array([0.0,-8.0,0.0])},
                             3:{'pos':np.array([1000.0,500.0,0.0]),'vel':np.array([-8.0,0.0,0.0])}}
        self.centre = np.array([500.0,500.0])
        self.vehicle_number = vehicle_number
        self.vehicles = {}
        self.bs_height = 10.0
        self.action_space= spaces.Discrete(11)
        low = np.array([0]*((self.vehicle_number*4)+(self.vehicle_number*2)+(5*3)+(2)))
        high = np.array(([1]*(self.vehicle_number*4))+([4,7]*self.vehicle_number)+([16,4,40]*5)+[1,1])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.user_usage = {i:0 for i in range(self.vehicle_number)} #might want to play around with the initialisation of this
        self.update_alpha = 0.05 #this is for the lambda function later in update_user_usage
        self.job_overview = {'completed_jobs':0, 'completed_jobs_size':[], 'exceed_deadline':0, 'exceed_deadline_size':[], 'latency':0.0,'throughput':0.0}
        # flag to control fair usage reward. if flag = False, we do not introduce the reward of user usage
        self.USER_USAGE_FLAG = False
        self.LATENCY_USAGE_FLAG = True
        self.THROUGHPUT_USAGE_FLAG = True
        # Meta learning task
        self.task = 0
        # same as reset
        self.generate_bs()
        self.gen_vehicles()

    def reset(self, USER_USAGE=False):
        self.step_count = 0 
        self.generate_bs()
        self.gen_vehicles()
        self.reward_arguments_list = []
        self.USER_USAGE_FLAG = USER_USAGE
        self.actions_todo = 2
        self.sim_time=10
        #self.time = 0.0
        self.time = np.random.randint(0,125-self.sim_time) # 10 seconds simulation
        
        # self.history = {'completed_jobs':0, 'completed_jobs_size':0.0, 'lost_jobs':0, 'exceed_deadline':0}
        # self.lost_history = {'lost_job_size':0.0, 'exceed_deadline_job_size':0.0}
        return self.state()
    
    def reset_history(self):
        self.history = {'completed_jobs':0, 'completed_jobs_size':[], 'exceed_deadline':0, 'exceed_deadline_size':[], 'latency':0.0, 'throughput':0.0}
        #self.lost_history = {'lost_job_size':0.0, 'exceed_deadline_job_size':0.0}
        self.job_overview = {'completed_jobs':0, 'completed_jobs_size':[], 'exceed_deadline':0, 'exceed_deadline_size':[], 'latency':0.0, 'throughput':0.0}
    
    def generate_bs(self):
        self.bs = {'init': BaseStation(),
                      'pos': np.append(self.centre.copy(),self.bs_height)}
        #generate 4 more micro base stations with mmWaves, plus the central BS
        self.bs_micro = {'number': 1,
                            'pos':np.append(self.centre.copy(),self.bs_height)}
        #np.array([[1000.0,500,self.bs_height], [500.0,1000.0,self.bs_height], [500.0,500.0,self.bs_height],
        #                                    [0.0,500.0,self.bs_height], [500.0,0.0,self.bs_height]])}

    def gen_vehicles(self):
        """
        Need to add in some offset between the vehicles otherwise they are always going
        to moving at the same speed
        """
        # offsets = [0.0, 100.0, 200.0, 300.0, 400.0] #less random
        for i in range(self.vehicle_number):
            offset = np.random.randint(0,1000)
            # offset = offsets[i]
            self.gen_pos(i, offset = float(offset))

    def gen_pos(self,i,offset = 0.0):
        """
        This generates a random position for the vehicle
        """
        n = np.random.choice([0,1,2,3])
        self.vehicles[i] = {'veh':Vehicle()}
        #this next bit is because otherwise numpy points towards the memory location rather than the values
        extra_info = { 'pos': self.start_points[n]['pos'].copy(),
                       'vel': self.start_points[n]['vel'].copy() }
        self.vehicles[i].update(extra_info)
        if n>=2:
            scaling = -1.0
        else:
            scaling = 1.0
        self.vehicles[i]['pos'][1-(n%2)] += scaling*offset

    def update_vehicles(self, time_elapsed = 0.0006):
        """
        Update the positions of the vehicles, the time elapsed term allows for transitions of any time
        """
        for i in range(self.vehicle_number):
            self.vehicles[i]['pos'] = self.vehicles[i]['pos'] + (self.vehicles[i]['vel'] * time_elapsed)
            if self.vehicles[i]['pos'][0] > 1000.0 or self.vehicles[i]['pos'][0] < 0.0:
                self.gen_pos(i)
            if self.vehicles[i]['pos'][1] > 1000.0 or self.vehicles[i]['pos'][1] < 0.0:
                self.gen_pos(i)

    def update_bs(self, time_elapsed = 0.001):
        return self.bs['init'].update(self.calculate_distances(),time_elapsed = time_elapsed)

    def calculate_distances(self):
        """
        This calculates all the distances between the bs and the vehicles so that the datarates can be inferred.
        LTE: one central base station
        mmWaves: 5 base stations, on in the centre - calculate the nearest distance among 5 BS
        |------BS------|
        |              |
        BS   BS(main)  BS
        |              |              |
        |------BS------|
        """
        distance_lte = np.array([np.linalg.norm(self.bs['pos'] - self.vehicles[i]['pos']) for i in range(self.vehicle_number)])
        distance_mmWaves = np.array([np.linalg.norm(self.bs_micro['pos'] - self.vehicles[i]['pos']) for i in range(self.vehicle_number)])

        """
        distance_mmWaves = []
        for i in range(self.vehicle_number):
            v_pos = self.vehicles[i]['pos']
            dist_to_microBS = np.array([np.linalg.norm((self.bs_micro['pos']- v_pos)[i]) for i in range(self.bs_micro['number'])])
            nearest_distance = np.min(dist_to_microBS)
            distance_mmWaves.append(nearest_distance)
        """
        distances = {'lte': distance_lte,
                        'mmWaves': np.array(distance_mmWaves)}
        return distances

    def vehicle_positions(self, scale = True):
        scaling = self.max_coord
        if not scale:
            scaling = 1.0
        return np.array([self.vehicles[i]['pos'] for i in range(self.vehicle_number)])/scaling

    def vehicle_velocities(self, scale = True):
        scaling = self.max_speed
        if not scale:
            scaling = 1.0
        return np.array([self.vehicles[i]['vel'] for i in range(self.vehicle_number)])/scaling

    def state(self):
        """
        State output, which is the vehicle positions, and buffer state for now
        Add in datarate too. #TODO
        """
        return np.concatenate([self.vehicle_positions()[:,0:2].flatten(),
                                self.vehicle_velocities()[:,0:2].flatten(),
                                self.bs['init'].state(self.calculate_distances())])

    def step(self, action, meta_task=0):
        """
        Perform a step in the environment.
        action: the action the agent has selected
        actions_todo: if there are two rats that need to be set, the agent may want to select multiple actions at
        one time step
        """
        meta_task = self.task
        self.step_count += 1
        self.actions_todo = self.actions_todo - 1
        r = 0.0
        info = {'episode':0} #this is because of stable baselines
        done = False
        if action != self.action_space.n -1: #if action is equal to do nothing action, skip
            index, RAT = np.divmod(action,2)# some operation to specify different RATs
            # add job
            self.bs['init'].add_job(RAT,index)
            # if add_successful: # get more reward by choosing a job close to deadline
            #     # cur_time_left = self.bs['init'].RATs[RAT]['job'][3] - self.bs['init'].RATs[RAT]['job'][2]
            #     # r += (self.assignment_reward(add_successful) * self.bs['init'].RATs[RAT]['job'][3] / cur_time_left)
            #     r += self.assignment_reward(add_successful)
            #     r += self.lte_assignment_reward(action)
            #     r += self.mmWaves_assignment_reward(action)
            # else:
            #     r += self.assignment_reward(add_successful)
        #else:
        #    r += -0.5 #for doing nothing
        r += self.RAT_usage_reward()
        # add part to allow better integration with baselines
        while np.where(self.bs['init'].RAT_status() == 1.0)[0].shape[0] == 0 or self.actions_todo == 0:
            self.update_env()
            self.time += 0.001 # get time
            self.actions_todo = np.where(self.bs['init'].RAT_status() == 1.0)[0].shape[0] #number of RATs to try and update
        """
        add in reward associated with succesful completion here
        """
        # get job history
        self.get_job_history()
        # get reward
        r += self.reward(reward_task=meta_task)
        if self.step_count > 5000:
            done = True
            self.step_count = 0
        return self.state(), r, done, info

    def update_env(self, time_elapsed = 0.001):
        # vehicle moving
        self.update_vehicles(time_elapsed = time_elapsed)
        # job and and buffer update 
        completed_jobs, exceed_deadline, completed_size, exceed_ddl_size, latency, throughput = self.update_bs(time_elapsed = time_elapsed)
        # self.update_user_usage(data_tx)
        for dict_key, add_to_value in zip(sorted(self.job_overview.keys()), [completed_jobs, completed_size, exceed_deadline, exceed_ddl_size]): #, latency, throughput removed
            self.job_overview[dict_key] += add_to_value
        if latency != 0:
            self.job_overview['latency']=latency
        if throughput != 0:
            self.job_overview['throughput']=throughput
        return self.state() #re-calculate state without getting new reward

    def get_job_history(self):
        for i in self.job_overview.keys():
            self.history[i] += self.job_overview[i]
            '''if i == 'latency':
                self.history[i] = self.job_overview['latency']
            elif i == 'throughput':
                self.history[i] = self.job_overview['throughput']'''

    def meta_task(self,meta_task):
        """
        Args:
            meta_task (task): This function will change how reward is calulated. 
            The function created because of error in step function: 
            "step() takes 2 positional arguments but 3 were given"
        """
        self.task = meta_task        

    def reward(self,reward_task=None):
        """Reward function defines the tasks that motivates RL agents. 

        Args:
            reward_task (_type_, reward_task): Defaults to None. Env.step() triggers reward will be derived by which task.

        Returns:
            _type_: _description_
        """
        #Tasks can define different rewards for meta learning under this function.
        reward=0
        scheme =2
        if scheme == 1:
            if reward_task==0:
                reward = self.usage_reward() + self.completed_reward() + self.deadline_reward() + self.latency_reward() + self.throughput_reward()
            elif reward_task == 1:
                reward = self.usage_reward() + self.completed_reward() + self.deadline_reward()
            elif reward_task == 2:
                reward = self.usage_reward() + self.latency_reward() 
            elif reward_task == 3: 
                reward = self.usage_reward() + self.throughput_reward()
            elif reward_task == 4:
                reward = self.usage_reward() + self.prop_fairness_reward()
        elif scheme == 2:
            if reward_task==0:
                reward = self.usage_reward() + self.completed_reward() + self.deadline_reward()
            elif reward_task == 1:
                reward = self.usage_reward() + self.prop_fairness_reward()
            elif reward_task == 2:
                reward = self.usage_reward() + self.latency_reward() 
            elif reward_task == 3: 
                reward = self.usage_reward() + self.throughput_reward()
            elif reward_task == 4:
                reward = self.usage_reward() + self.completed_reward() + self.deadline_reward() + self.latency_reward() + self.throughput_reward()
        
        
        # reset job_overview
        self.job_overview = {'completed_jobs':0, 'completed_jobs_size':[], 'exceed_deadline':0, 'exceed_deadline_size':[], 'latency':0.0, 'throughput':0.0}
        return reward

    def assignment_reward(self, success):
        """
        If the agent succesfully selects a job, it gets a reward of +1 else, it gets a reward
        of -10
        """
        if success:
            return 1.0
        else:
            return 0.0
        #     return -10.0

    def update_user_usage(self, stats):
        """
        When updating the user data rates, we do an incremental update to an averaged value. In this manner, we do not need to keep the full history of the
        user data rates. You will need to play around with the alpha parameter though
        Some concern with this function if I'm being honest. I'm concerned that due to the disparity between the two radio access technologies data rates,
        the updates will be significantly larger when anyone is using the mmwave technology. Might be an idea to change this to time access to the medium? Atleast
        then it is independent of data rate.
        """
        update_usage = lambda old, new, alpha: old + (alpha  * (new - old))
        for i in self.user_usage.keys():
            self.user_usage[i] = update_usage(self.user_usage[i], stats[i], self.update_alpha)

    def usage_reward(self):
        """
        As email, this converts the current usage into a probability distribution then measures the similarity using an entropy meausre
        might not import scipy properly
        """
        entropy = 0.0
        # user usage reward can be included or removed from the reward function
        if self.USER_USAGE_FLAG:
            prob = scipy.special.softmax(list(self.user_usage.values())) #convert from raw values to probability distribution
            entropy = scipy.stats.entropy(prob) #convert to entropy
        return entropy

    def completed_reward(self):
        """
        You'll want to change the co-effeciencts probably
        """
        # return float(self.job_overview['completed_jobs']) * 1.0 + self.job_overview['completed_job_rate'] * 0.05
        return float(self.job_overview['completed_jobs']) * 10.0

    def latency_reward(self):
        """
        Co-effecienct of latency is defined as 0.1. It indicates the total completing time divided by 1000
        """
        Completed=self.job_overview['completed_jobs']
        if Completed == 0:
            Completed =1 # Not a good solution to prevent dividing by 0
        latency = 0
        if self.LATENCY_USAGE_FLAG:
            latency =100*(1- float((self.job_overview['latency'])/Completed))
        return latency
    
    def throughput_reward(self):
        """
        Co-effecienct of throughput is defined as 0.05. It indicates the MBytes/s divided by 20
        """
        throughput = 0
        Completed=self.job_overview['completed_jobs']
        Lost = self.job_overview['exceed_deadline']
        if (Completed + Lost) == 0:
            Lost =1 # Not a good solution
        if self.THROUGHPUT_USAGE_FLAG:
            throughput = float((self.job_overview['throughput'])/(Completed+Lost)) * 1
        return throughput

    def prop_fairness_reward(self):
        """
        Co-effecienct of throughput is defined as 0.05. It indicates the MBytes/s divided by 20
        """
        throughput = 0
        Completed=self.job_overview['completed_jobs']
        Lost = self.job_overview['exceed_deadline']
        if (Completed + Lost) == 0:
            Lost =1 # Not a good solution
        if self.THROUGHPUT_USAGE_FLAG:
            throughput = float((self.job_overview['throughput'])/(Completed+Lost)) * 3
        return np.float32(np.log(throughput))
    # def lost_reward(self):
    #     """
    #     You'll want to change the co-effeciencts probably
    #     """
    #     return float(self.job_overview['lost_jobs']) * -50.0

    def deadline_reward(self):
        """
        Deadline reward: the more time a job has been waiting, the more penalizaiton it would get.
        Exponential reward, see Buffer_Object: waiting_time_reward
        Previous reward function:
        # return self.job_overview['exceed_deadline'] * 0
        """
        # #check the buffer
        # r = 0
        # for ii in range(len(self.bs['init'].buffer.buffer)):
        #     r += self.bs['init'].buffer.buffer[ii][4] #add up the waiting reward for every job in the buffer
        # # #check current RATs
        # # for jj in self.bs['init'].RATs.keys():
        # #     if not self.bs['init'].RATs[jj]['free']:
        # #         r += self.bs['init'].RATs[jj]['job'][4]
        # return r
        return float(self.job_overview['exceed_deadline']) * -100.0 # was -100.0

    def RAT_usage_reward(self):
        """
        This function gives the RAT availability reward.
        """
        r = 0.0
        for i in self.bs['init'].RATs.keys():
            r -= float(self.bs['init'].RATs[i]['free'])
        return r/100

    def lte_assignment_reward(self, action):
        # if the job assigned to the lte RAT is for the vehicle out of mmWaves range, get extra reward
        index, RAT = np.divmod(action,2)
        r = 0.0
        if RAT == 0:
            job = self.bs['init'].buffer.buffer[index].copy()
            target_v = job[1]
            v_dist = self.calculate_distances()['mmWaves']
            if v_dist[target_v] > self.bs['init'].RAT_info[1]['max_range']:
                r += 2.0
        return r

    def mmWaves_assignment_reward(self, action):
        # if the job assigned to the mmWaves RAT is for the vehicle out of mmWaves range, get negative reward
        index, RAT = np.divmod(action,2)
        r = 0.0
        if RAT == 1:
            job = self.bs['init'].buffer.buffer[index].copy()
            target_v = job[1]
            v_dist = self.calculate_distances()['mmWaves']
            if v_dist[target_v] > self.bs['init'].RAT_info[1]['max_range']:
                r -= 2.0
        return r

    def return_jobs(self):
        """
        Return the finnshed jobs table and the lost jobs when an episode ends
        """
        return self.history
