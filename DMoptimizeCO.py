import numpy as np
import scipy as sp
import time
from abc import ABC, abstractmethod
import DMsimulatorCO as DMsim
import logging


class Optimize(ABC):
    
    def __init__(self, const_dict, steric_tuple, energy_dict, file_input_data, loss_func, input_dim, 
                init_conditions=[0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05], 
                max_time=7, print_flag=True, max_buffer_size=10_000):
        
        
        self.system = DMsim.SurfaceKineticsSimulator(const_dict, steric_tuple, file_input_data, init_conditions=init_conditions)
        self.input_data_dict, self.recProbExp_vec = self.system.prepare_data()
        self.const_dict = const_dict
        
        self.energy_dict_base = energy_dict.copy()
        
        self.loss_func = loss_func
        self.max_time = max_time
        
        self.print_flag = print_flag
        self.nb_calls = 0
        self.buffer = np.empty((max_buffer_size, input_dim + 1))
    
    
    @abstractmethod
    def modify_energy_dict(self, params, counter):
        #### Modify the energy dict based on parameters and counter
        #### This method should be implemented in any class
        pass
    
    
    def functional_loss(self, params, flag_return_recProb=False):
        ##* fix loss function declaration
        
        loss_vec = np.zeros(len(self.input_data_dict))
        
        if flag_return_recProb:
            gammas_tensor = []
        
        for i in range(len(self.input_data_dict)):
            
            energy_new_dict = self.modify_energy_dict(params, i)
            _, recProb_aux, _, sucess = self.system.solve_system(self.input_data_dict[i], energy_new_dict, solver="fixed_point", max_time=self.max_time)
            
            if sucess == True:
                value_vec = (np.sum(recProb_aux) - self.recProbExp_vec[i])/self.recProbExp_vec[i]
                loss_vec[i] = self.loss_func(value_vec)       
            else: 
                logging.warning(f"Simulation failed for index {i}. Assigning high loss.")
                loss_vec[i] = 1e6
            
            if flag_return_recProb:
                gammas_tensor.append(recProb_aux)
                
        value = np.mean(loss_vec)
        
        self.nb_calls += 1
        self.buffer[self.nb_calls % len(self.buffer), :] = [value] + params.tolist()
        
        if self.print_flag and self.nb_calls % 5 == 0:
            print("Loss: ", value, "Params: ", params)
        
        if flag_return_recProb:
            return value, np.array(gammas_tensor)
        else:
            return value
    
    
    
    def hybrid_search(self, config):
        
        bounds = config["bounds"]
        nb_de_calls = config["nb_de_calls"]
        de_maxiter = config["de_maxiter"]
        local_attempts = config["local_attempts"]
        epsilon_local = config["epsilon_local"]
        top_k = config["top_k"]
        
        
        candidates = []
        for _ in range(nb_de_calls):
            result = sp.optimize.differential_evolution(self.functional_loss, bounds, polish=True, disp=True, maxiter=de_maxiter)
        
            candidates.append(result.x)
            
            if self.print_flag:
                print("Global Search Result: ", result)
                print()
        
        candidates_losses = [self.functional_loss(candidate) for candidate in candidates]
        best_candidate = candidates[np.argmin(candidates_losses)]
        
        if self.print_flag:
            print("Candidates: ", candidates)
            print("Candidates Losses: ", candidates_losses)
            
            print("Best candidate Global Search: ", best_candidate)
            print()
        
        bounds_array = np.array(bounds)
    
        k_local = top_k if top_k < nb_de_calls else nb_de_calls
        
        ### chose the k best candidates
        idx_best_candidates = np.argsort(candidates_losses)[:k_local]
        best_candidates = np.array([candidates[i] for i in idx_best_candidates])
        best_candidates_losses = np.array([candidates_losses[i] for i in idx_best_candidates])
        
        
        best_local = best_candidates[np.argmin(best_candidates_losses)]
        best_local_loss = np.min(best_candidates_losses)
        
        
        
        if self.print_flag:
            print("best_local_loss: ", best_candidates_losses)
            print("best_local: ", best_candidates)
        
        for best_candidate, best_candidate_loss in zip(best_candidates, best_candidates_losses):
        
            for attempt in range(local_attempts):
                
                perturbation = np.random.uniform(-epsilon_local, epsilon_local, len(best_candidate))
                x0 = best_local + perturbation
                
                ### check bounds
                x0 = np.maximum(x0, bounds_array[:, 0]) 
                x0 = np.minimum(x0, bounds_array[:, 1])
                
                # local_result = sp.optimize.minimize(self.functional_loss, x0, method="Nelder-Mead", bounds=bounds)
                local_result = sp.optimize.minimize(self.functional_loss, x0, method="L-BFGS-B", bounds=bounds)
                
                if self.print_flag:
                    print("Local Search Attempt: ", attempt, "Loss: ", local_result.fun, "Params: ", local_result.x)
                    print()
                
                if local_result.fun < best_local_loss:
                    best_local_loss = local_result.fun
                    best_local = local_result.x
        
        ### trim buffer
        self.buffer = self.buffer[:self.nb_calls, :]
        
        return best_local, best_local_loss
    