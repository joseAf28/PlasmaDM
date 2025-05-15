import numpy as np
import scipy as sp
import os
from pathos.multiprocessing import ProcessPool

import Simulator as sim_system 


class Optimize():
    
    def __init__(self, simulator_object, func_uncertainty_dict, func_exp_dict, print_flag=True):
        
        self.simulator = simulator_object
        self.func_uncertainty_dict = func_uncertainty_dict
        self.func_exp_dict = func_exp_dict
        self.gamma_exp_data = self.simulator.gamma_exp_data_arr
        
        self.functional_calls = 0
        self.print_flag = print_flag
        


