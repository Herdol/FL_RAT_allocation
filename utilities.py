from re import X
import numpy as np
import os
import wandb

os.environ['KMP_DUPLICATE_LIB_OK']='True'
## Example W%B commands 
class logger:
    def __init__(self, wandb_flag):
        self.wandb_flag=wandb_flag

    def log(self,job_history,episode,alg,widx=None, sim_timer=None):
        completed_jobs = job_history['completed_jobs']
        completed_jobs_size = np.sum(job_history['completed_jobs_size'])
        exceed_deadline = job_history['exceed_deadline']
        exceed_deadline_size = np.sum(job_history['exceed_deadline_size'])
        latency = job_history['latency']/(completed_jobs)
        throughput = job_history['throughput']/(completed_jobs+exceed_deadline)
        episode = episode
        alg = alg
        widx = widx
        caching_rate=completed_jobs/(completed_jobs+exceed_deadline)
        caching_bytes=completed_jobs_size/(completed_jobs_size+exceed_deadline_size)
        if alg == 'fedavg':
            metrics = {'Worker {} caching rate'.format(widx): caching_rate,
                        'Worker {} caching rate(bytes)'.format(widx): caching_bytes,
                        'Worker {} latency'.format(widx):latency,'Worker {} throughput'.format(widx):throughput,'Episode':episode }
        elif alg == 'fedavg_val':
            metrics = {'Fed adaptation caching rate': caching_rate,
                        'Fed adaptation caching rate(bytes)': caching_bytes,
                        'Fed adaptation latency':latency,'Fed adaptation throughput':throughput,'Episode':episode}
        elif alg == 'single':
            metrics = {'Single agent caching rate': caching_rate,
                        'Single agent caching rate(bytes)': caching_bytes,
                        'Single agent latency':latency,'Single agent throughput':throughput,'Episode':episode}
        elif alg == 'single_val':
            metrics = {'Single agent adaptation caching rate': caching_rate,
                        'Single agent adaptation caching rate(bytes)': caching_bytes,
                        'Single agent adaptation latency':latency,'Single agent adaptation throughput':throughput,'Episode':episode} 
        else:
            metrics = {'Heuristic caching rate': caching_rate,
                        'Heuristic caching rate(bytes)': caching_bytes,
                        'Heuristic latency':latency,'Heuristic throughput':throughput,'Episode':episode}
        
        
        if self.wandb_flag == True:
            wandb.log(metrics)
            print('Episode {} for worker {} in {} algorithm is logged (Caching rate: {:.2f}%, Bytes: {:.2f}%) in {:.1f} sec'.format(episode,widx,alg,100*caching_rate,100*caching_bytes,sim_timer))
        else:
            print('Episode {} for worker {} in {} algorithm is done (Caching rate: {:.2f}%, Bytes: {:.2f}%) in {:.1f} sec'.format(episode,widx,alg,100*caching_rate,100*caching_bytes,sim_timer))

        return metrics

        
    
