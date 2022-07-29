import numpy as np
import math
import scipy.io
#import matplotlib.pyplot as plot
#from matplotlib.animation import FuncAnimation

class Buffer_Object(object):

    def __init__(self, size = 4, destinations = 5):
        """
        This initialises a data buffer, from which the sender will select between
        size: integar, the length of the buffer
        destinations: integar, the number of possible recipients.
        """
        self.size = size
        self.destinations = destinations
        #self.data_packets = np.array([np.linspace(20,20,8),np.linspace(100,100,8)]) # Was np.linspace(200,200,8)
        self.sim_length='short'
        if self.sim_length == 'short':
            self.data_packets = np.array([np.linspace(1,1,8),np.linspace(10,10,8)]) 
            self.data_deadline = np.array([np.linspace(0.1,0.1,8),np.linspace(1,1,8)])  #0.2 is not tested in terms of variable format.
        elif self.sim_length == 'long':
            self.data_packets = np.array([np.linspace(20,20,8),np.linspace(100,100,8)]) 
            self.data_deadline = np.array([np.linspace(2,2,8),np.linspace(20,20,8)])  #set deadlines corresponding to job size
         
        self.data_unit_size = 1 # size of one data unit
        self.data_packets_num = self.data_packets / self.data_unit_size
	    #same as above but includes a 0 size item, to indicate no job to send
        self.data = np.concatenate([np.array([0]),self.data_packets_num.flatten()])
        self.fill_buffer(first_run = True)
        # record lost jobs due to 'stays too long in the buffer'
        #?
        self.lost_table = []
        # a parameter used in calculating deadline reward
        self.ddl_reward_par = 8.0

    def gen_item(self):
        """
        This generates jobs and deadlines for adding to the buffer
        p is the probability of the choice of different size packets
        """
        row = np.random.choice([0,1], p = [0.5,0.5])
        column = np.random.choice(list(range(8)))
        return self.data_packets_num[row,column], self.data_deadline[row,column]

    # def gen_deadline(self, data_size):
    #     """
    #     This generates deadlines given job size.
    #     """
    #     if data_size % 1000 == 0:
    #         ddl = 15.0 + data_size/1000 - 1
    #     elif data_size % 10 == 0:
    #         ddl = 5.0 + data_size/10 - 1
    #     return ddl

    def fill_buffer(self, first_run = False):
        """
        this fills the buffer
        The items are appended with the following values
        [size, dest, time_since_request, deadline]
        Todo: figure out when to do this
        """
        if first_run:
            self.buffer = []
        for i in range(self.size - len(self.buffer)):
            dest = np.random.choice(list(range(self.destinations)))
            size, deadline = self.gen_item()
            self.buffer.append([size, dest, deadline, size])
            # self.buffer.append([size,dest,0,deadline])

    def view_buffer(self):
        """
        This function allows for easier representation of the state to the agent.
        the np.vectorize thing, is to allow a function to be applied to a numpy array.
        This effectively scales our 16 different jobs sizes, plus job size 0, to a value between 0 and 16
        Potential change, change the values to be binary??
        """
        cp_buffer = np.array(self.buffer.copy()) #make a copy of the buffer so I don't corrupt it
        cp_buffer = cp_buffer[:,0:3]
        # v_scale = np.vectorize(lambda value: np.where(self.data == value)[0][0]) #vectorized function
        #scale every item in the buffer to [0,1] - designed for multiple job sizes
        cp_buffer[:,0] = cp_buffer[:,0] / np.max(self.data)
        cp_buffer[:,1] = cp_buffer[:,1] / (self.destinations-1)
        cp_buffer[:,2] = cp_buffer[:,2] / np.max(self.data_deadline.flatten())
        # cp_buffer[:,3] = cp_buffer[:,3] / np.max(self.data_deadline.flatten())
        # cp_buffer[:,4] = cp_buffer[:,4] / v_scale(cp_buffer[:,4]) / 16.0
        return cp_buffer.flatten()

    def update(self, job_finished, job_from, time_elapsed = 0.001 ):
        """
        This function increments the time waited value, and removes jobs that have exceeded this deadline or have
        been assigned. It also refills the buffer
        -------
        Later
        Action: Update job in RAT
        Time elapse
        Deadline: remove
        Refill buffer
        -------
        Update from Hakan
        Latency info
        """
        exceed_deadline = 0
        finished_jobs = 0
        latency=0
        throughput =0 
        #?
        finished_job_size = []
        exceed_deadline_size = []
        to_remove = []
        # update jobs in the buffer after transmission
        for ii in range(len(job_from)):
            idx = job_from[ii]
            self.buffer[idx][0] -= 1
        for i in range(len(self.buffer)): #iterate through buffer
            self.buffer[i][2] -= time_elapsed #increment time waited
            if self.buffer[i][0] <= 0:
                # Completed job
                to_remove.append(i)
                finished_jobs += 1
                finished_job_size.append(self.buffer[i][3])
                ## Latency calculation ##
                # self.buffer[i][3] is total job size divided 5 means total deadline
                # self.buffer[i][2] is remaining deadline
                latency = ((self.buffer[i][3]/10)-self.buffer[i][2])/(self.buffer[i][3]/10)
                throughput= (self.buffer[i][3]/((self.buffer[i][3]/10)-self.buffer[i][2]))
            elif self.buffer[i][2] <= 0:
                # Lost job
                exceed_deadline += 1 #track that it's due to be removed
                to_remove.append(i) #add it to a list to allow for removal at end of function 
                exceed_deadline_size.append(self.buffer[i][3])  # record size of jobs being removed
                throughput= (self.buffer[i][3]/((self.buffer[i][3]/10)-self.buffer[i][2]))
        for i in to_remove[::-1]: #run backwards through the list of jobs to be removed. Backwards to avoid indexing error.
            self.remove_item(i) #call removal function
        self.fill_buffer() #refill the buffer
        return finished_jobs, exceed_deadline, finished_job_size, exceed_deadline_size, latency, throughput #report the number of jobs that exceeded deadline

    def remove_item(self,i):
        """
        remove item from list, if it has been placed into one of the RAT
        """
        del self.buffer[i]

    def to_be_removed(self, i):
        """
        This function is used when a job has been succesfully assigned. It is represented by the size being set to 0
        """
        self.buffer[i][0]=0

    # def waiting_time_reward(self, waiting_time):
    #     """
    #     This is the penalization for job waited in the buffer (also in the RATs).
    #     Assume wating time is x, y is a constant for contronling the value
    #     r = -e^(x - y)
    #     y could be tuned
    #     """
    #     if waiting_time == 0:
    #         reward = 0
    #     else:
    #         reward = -0.5 * waiting_time
    #         # reward = -1.1**(waiting_time - self.ddl_reward_par)
    #         # reward =  -math.exp(waiting_time - self.ddl_reward_par)
    #     return reward


class BaseStation(object):

    def __init__(self, vehicles = 5, I2V = True):
        if I2V:
            self.buffer = Buffer_Object(size = 5)
        self.vehicles = vehicles
        self.RATs = {
                0: {'job': [], 'free' : True, 'from': 999},
                1: {'job': [], 'free' : True, 'from': 999},
                }
        self.load_RAT_spec()
        self.time = 0.0
        # record finished jobs:[size]; lost jobs: [size]
        self.finished_table = []
        self.lost_table_r = [] # lost jobs because of out of the range of RAT
        self.lost_table_d = [] # lost jobs due to deadline
        # cur_RATs: label = 0: start status; label = 1: finished; label = 2: lost
        self.cur_RATs = {
                0: {'job': []},
                1: {'job': []},
                }

    def load_RAT_spec(self):
        """
        Load the matlab lists which indicate data rate and distance into a dicitionary
        for easy access later
        """
        linkBudget_lte = scipy.io.loadmat('lte.mat')
        lte = linkBudget_lte['lte']
        linkBudget_mmWaves = scipy.io.loadmat('mmWaves.mat')
        mmWaves = linkBudget_mmWaves['mmWaves']
        # data rate: in Mb
        self.RAT_info = {
                0:{'name':'LTE','dist': lte[0][0][0][0],
                    'datarate':lte[0][0][1][0]/ 1e6,'res':0,
                    'max_range': np.max(lte[0][0][0][0])},
                1:{'name':'mmWave','dist': mmWaves[0][0][0][0],
                    'datarate':mmWaves[0][0][1][0]/ 1e6,'res':1,
                    'max_range': np.max(mmWaves[0][0][0][0])}
                }
        self.RAT_info[0]['data_index'] = np.concatenate([np.array([0]),np.unique(self.RAT_info[0]['datarate'])])
        self.RAT_info[1]['data_index'] = np.concatenate([np.array([0]),np.unique(self.RAT_info[1]['datarate'])])

    def add_job(self, RAT, index):
        """
        Append job to RAT specified.
        RAT: 0 or 1, either lte or mmwave
        index: index of item in buffer
        """
        row=index # In case of choosing an action that is not in buffer destination ( in RL training) This is never needed in heurustic method  
        for idx in range(5):
            if self.buffer.buffer[idx][1]== index:
                row=idx
        ########### ERROR was here #############
        item = self.buffer.buffer[row].copy() # instead of veh number it adds job to the row number ## was index
        item[0] = 1 * self.buffer.data_unit_size # add one unit to RAT every time
        success = False
        if self.RATs[RAT]['free']:
            self.RATs[RAT]['job'] = item
            self.RATs[RAT]['from'] = index # where this unit comes from 
            self.RATs[RAT]['free'] = False
            success = True
            # add the job size for record
            #?
            self.cur_RATs[RAT]['job'] = self.RATs[RAT]['job'][0]
        return success

    def update(self, distances, time_elapsed = 0.001):
        """
        This updates the base station entity. It does a number of things including update a time variable.
        It updates the jobs progress in transmission (which includes checking for inability to send and
        also checking if jobs in the buffer have exceeded the available time)
        """
        self.time += time_elapsed
        job_finished, job_from = self.update_jobs(distances, time_elapsed = time_elapsed)
        finished_jobs, exceed_deadlines, finished_size, exceed_ddl_size, latency,throughput = self.buffer.update(
                job_finished, job_from, time_elapsed = time_elapsed)
        return finished_jobs, exceed_deadlines, finished_size, exceed_ddl_size, latency, throughput

    def update_jobs(self, distances ,time_elapsed = 0.001):
        """
        Transmit some of the data and then return the amount of data that has been transmitted
        arguments:
            distances - dictionary, an np.array of all of the distances to the vehicles, for LTE and mmWaves RATs.
            time_elapsed - float, a real number. Time that has elapsed.
        operation:
            goes through the items in the RATs and calculates the amount of data that has been sent.
        Things to consider:
            One problem is it assumes the data rate now has been the data rate since time elapsed. This should be changed.
        """
        data_tx = {i:0 for i in range(self.vehicles)}
        idx = 0
        RATs=[1,0] #mmWave priority
        #for i in self.RATs.keys():
        for i in RATs:   
            if not self.RATs[i]['free']: #if the RAT isn't free there is a job
                size_before = self.RATs[i]['job'][0] #the size of the job
                dest = self.RATs[i]['job'][1] #the destination of the job
                distance = np.round(distances[list(distances.keys())[idx]][dest],self.RAT_info[i]['res'])#job rounded w.r to RAT
                if distance > self.RAT_info[i]['max_range']: #if out of range
                    data_rate = 0 #there is no service, so the data rate is 0
                    self.RATs[i]['job'] = [] #therefore, we drop the job
                    self.RATs[i]['free'] = True #and change the status of the RAT
                else: #in range
                    data_rate = self.RAT_info[i]['datarate'][np.where(self.RAT_info[i]['dist']==distance)[0][0]] #calculate data rate at user position
                    self.RATs[i]['job'][0] -= data_rate * time_elapsed #now update the job size
                    data_tx[dest] += size_before - np.max([0,self.RATs[i]['job'][0]])
                    self.RATs[i]['free'] = False # Hakan added 
            idx += 1           
        job_finished, job_from = self.check_job_finished()#check if jobs are finished
        # exceed_deadlines = self.check_deadlines() #check if the time has exceeded avaialble
        return job_finished, job_from

    def check_job_finished(self):
        """
        Checks if the jobs have finished, which is defined by if their remaining units are equal to 0
        Returns the number of jobs that have been completed
        """
        job_finished = []
        job_from = []
        idx = 0
        for i in self.RATs.keys():
            if not self.RATs[i]['free']:
                if self.RATs[i]['job'][0]<=0.0:
                    self.RATs[i]['job'] = []
                    self.RATs[i]['free'] = True
                    job_finished.append(i)# Was idx 
                    job_from.append(self.RATs[i]['from'])
                    idx += 1
        return job_finished, job_from

    def check_deadlines(self):
        """
        Checks if the jobs have exceeded their deadline.
        If they have they are removed.
        """
        exceed_deadlines = 0
        for i in self.RATs.keys():
            if not self.RATs[i]['free']:
                if self.RATs[i]['job'][2] >= self.RATs[i]['job'][3]:
                    self.RATs[i]['job'] = []
                    self.RATs[i]['free'] = True
                    exceed_deadlines += 1
                    # record lost jobs
                    self.lost_table_d.append(self.cur_RATs[i]['job'])
                    self.cur_RATs[i]['job'] = self.RATs[i]['job']
        return exceed_deadlines

    def data_rate_vehs(self, distances):
        """
        this gives the status of connected vehicles links, i.e., the data rates of RATs for all vehicles
        this is part of the BS status
        Input: distance of connected vehicles
        """
        datarate_veh = np.zeros((self.vehicles,len(self.RAT_info.keys())))
        idx = 0
        for ii in distances.keys():
            cur_distances = distances[ii]
            for vehicle, dist in enumerate(cur_distances):
                i = list(self.RAT_info.keys())[idx]
                dist = np.round(dist, self.RAT_info[i]['res'])#job rounded w.r to RAT
                if dist > self.RAT_info[i]['max_range']: #if out of range
                    data_rate = 0 #there is no service, so the data rate is 0
                else: #in range
                    data_rate = self.RAT_info[i]['datarate'][np.where(self.RAT_info[i]['dist']==dist)[0]][0] #calculate data rate at user position
                datarate_veh[vehicle, i] = np.where(self.RAT_info[i]['data_index'] == data_rate)[0][0]
            idx += 1
        return datarate_veh

    def view_datarate_vehs(self, datarate_veh):
        """
        Scale datarate_veh to [0,1]
        """
        datarate_veh_c = np.array(datarate_veh.copy())
        datarate_veh_c[:,0] = datarate_veh_c[:,0] / (len(self.RAT_info[0]['data_index'])-1)
        datarate_veh_c[:,1] = datarate_veh_c[:,1] / (len(self.RAT_info[1]['data_index'])-1)
        return datarate_veh_c.flatten()
    
    def RAT_status(self):
        return np.array([self.RATs[i]['free'] for i in self.RATs.keys()]).astype(float)

    def state(self, distances):
        """
        this gives the BS status, including the status of links of connected vehicles (datarate_veh)
        RATs (RAT_status) and the buffer
        Input: distances of connected vehicles
        """
        return np.concatenate([self.view_datarate_vehs(self.data_rate_vehs(distances)), self.buffer.view_buffer(), self.RAT_status()])


class Vehicle(object):

    def __init__(self, V2I = False):
        if V2I:
            self.buffer = Buffer_Object()
