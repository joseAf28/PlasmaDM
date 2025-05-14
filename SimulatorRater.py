import numpy as np
import scipy as sp
import h5py

import rates as rates_module


class SimulatorRates():
    
    def __init__(self, reactions_file, const_dict, exp_data_file, output_parser):
        
        self.const_dict = const_dict
        self.exp_data_file = exp_data_file
        self.output_parser = output_parser
    
    
    def prepare_experimental_data(self, transformations_exp=None):
        n_rows = None
                
        data_dict = {}
        with h5py.File(self.exp_data_file, "r") as file:
            keys_hdf5 = list(file.keys())
            
            for key in keys_hdf5:
                data_dict[key] = file[key][:]
                
                if n_rows is None:
                    n_rows = len(data_dict[key])
        file.close()
        
        ### rewrite the final data
        exp_vec_arr = np.zeros(n_rows, dtype=object)
        for i in range(n_rows):
            dict_aux = {}
            for key in keys_hdf5:
                if key == 'File_Data' or key == 'gamma_exp':
                    continue
                dict_aux[key] = data_dict[key][i]
            
            for key in transformations_exp.keys():
                dict_aux[key] = transformations_exp[key](self.const_dict, dict_aux)  
            
            exp_vec_arr[i] = dict_aux
            
        gamma_exp_vec = data_dict["gamma_exp"]        
        return exp_vec_arr, gamma_exp_vec
    
    
    def preload_rates_functions(self):
        different_rates_list = list(set([reaction['rate'] for reaction in self.output_parser['reactions_list']]))
        self.available_rates_functions = dict()
        
        for name in different_rates_list:
            if hasattr(rates_module, name):
                self.available_rates_functions[name] = getattr(rates_module, name)
            else:
                raise ValueError(f"{name} is not defined on the rates module!")
    
    
    def compute_rates_simulation(self, exp_vec_arr, reactions_list):
        rates_calculation_arr = np.zeros((len(exp_vec_arr), len(reactions_list)), dtype=float)
        
        for i in range(len(exp_vec_arr)):
            for j in range(len(reactions_list)):
                
                model_dict = reactions_list[j]['model_dict']
                gas_specie = reactions_list[j]['gas_specie']
                model_dict['gas_specie'] = gas_specie
                
                rate_name = reactions_list[j]['rate']
                rate_function = self.available_rates_functions[rate_name]
                rate_value = rate_function(self.const_dict, exp_vec_arr[i], model_dict)
                
                rates_calculation_arr[i, j] = rate_value
        
        return rates_calculation_arr