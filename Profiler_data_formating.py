import cProfile
import pstats
from pstats import SortKey

#import cProfile

#cProfile.run('.\Pytorch_v07_FL_GPU.py', 'restats1.dat')
for widx in range(5):
   with open("output_FL_time_worker_{}.txt".format(widx),"w") as f:
      p=pstats.Stats("Profiling_worker_{}.dat".format(widx),stream=f)
      p.sort_stats("time").print_stats()

   with open("output_FL_calls_worker_{}.txt".format(widx),"w") as f:
      p=pstats.Stats("Profiling_worker_{}.dat".format(widx),stream=f)
      p.sort_stats("calls").print_stats()
                                    