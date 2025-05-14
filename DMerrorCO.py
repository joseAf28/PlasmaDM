import numpy as np
import scipy as sp
import time
import DMsimulatorCO as DMsim
import logging
from abc import ABC, abstractmethod
from tqdm import tqdm
from scipy.stats import norm, qmc
from joblib import Parallel, delayed


class ErrorPropagation(ABC):
    
    def __init__(self, const_dict, steric_dict, energy_dict, file_input_data, 
                init_conditions=[0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05], 
                max_time=7, print_flag=True):
        
        self.system = DMsim.SurfaceKineticsSimulator(const_dict, steric_dict, file_input_data, init_conditions=init_conditions)
        self.input_data_dict, self.recProbExp_vec = self.system.prepare_data()
        self.const_dict = const_dict
        
        self.energy_dict_base = energy_dict.copy()
        
        self.max_time = max_time
        self.print_flag = print_flag
        self.nb_calls = 0
    
    
    
    def modify_energy_dict(self, counter):
        #### Modify the energy dict based on parameters and counter
        #### This method should be implemented in any class
        pass
    
    
    
    def func_input_error_propagation_numeric(self, func, std_input_ratios_dict, counter):
        ### too slow in the case of many parameters
        variables = len(std_input_ratios_dict)
        exp_samples = self.input_data_dict[counter]
        
        mean_input_dict = exp_samples.copy()
        std_input_dict = {key: std_input_ratios_dict[key] * exp_samples[key] for key in std_input_ratios_dict.keys()}
        
        keys_list = list(std_input_ratios_dict.keys())
        
        mean_input_array = np.array([mean_input_dict[key] for key in keys_list])
        std_input_array = np.array([std_input_dict[key] for key in keys_list])
        
        def pdf_function(*x):
            prod_func = 1.0
            for i in range(variables):
                prod_func *= sp.stats.norm.pdf(x[i], loc=mean_input_array[i], scale=std_input_array[i])
            
            return prod_func
        
        def integrand(*x):
            
            pdf_value = pdf_function(*x)
            
            sample_mod_dict = {key: x[i] for i, key in enumerate(keys_list)}
            exp_samples.update(sample_mod_dict)
            
            energy_dict_new = self.modify_energy_dict(counter)
            
            _, recProb_aux, _, sucess = self.system.solve_system(exp_samples, energy_dict_new, solver="fixed_point", max_time=self.max_time)
            
            if sucess == True:
                prob_value = np.sum(recProb_aux)
            else: 
                logging.warning(f"Simulation failed for index {counter}. Assigning high loss.")
                prob_value = 0.0
                
            return pdf_value * func(prob_value)
            
        
        bounds = [[mean_input_dict[key] - 4 * std_input_dict[key], mean_input_dict[key] + 4 * std_input_dict[key]] for key in keys_list]
        
        # Perform the integration
        start_time = time.time()
        integral_value, error = sp.integrate.nquad(integrand, bounds)
        end_time = time.time()
        
        logging.info(f"Integration time: {end_time - start_time:.2f} seconds")
        
        return integral_value, error
    
    
    
    def func_input_error_propagation_MC(self, std_input_ratios_dict, counter, N=10_000, use_qmc=False, qmc_pow2=12, n_jobs=8):
        
        exp_sample = self.input_data_dict[counter]
        energy_dict = self.modify_energy_dict(counter)
        
        keys = list(std_input_ratios_dict.keys())
        mean_arr = np.array([exp_sample[k] for k in keys])
        std_arr = np.array([(exp_sample[k] * std_input_ratios_dict[k] if exp_sample[k] > 0 else std_input_ratios_dict[k]) for k in keys])
        
        if use_qmc:
            sampler = qmc.Sobol(d=len(keys), scramble=True)
            u = sampler.random_base2(m=qmc_pow2)
            samples = norm.ppf(u, loc=mean_arr, scale=std_arr)
        else:
            samples = np.random.randn(N, len(keys)) * std_arr + mean_arr
        
        
        def run_sim(theta):
            params = exp_sample.copy()
            for i, k in enumerate(keys):
                params[k] = theta[i]
            
            _, recProb, _, success = self.system.solve_system(params, energy_dict, solver="fixed_point", max_time=self.max_time)
            if not success:
                logging.warning(f"Simulation failed at counter={counter}, theta={theta}")
                val = 0.0
            else:
                val = np.sum(recProb)
            return val
        
        results = Parallel(n_jobs=n_jobs)(delayed(run_sim)(theta) for theta in samples)
        results = np.asarray(results)
        
        mean_est = float(np.mean(results))
        std_est = float(np.std(results, ddof=1))
        
        logging.info(
            f"Monte Carlo ({'QMC' if use_qmc else 'MC'}) N={len(results)} | mean={mean_est:.5g} | std={std_est:.5g}"
        )
        
        return mean_est, std_est, results
    
    
    def func_parameter_error_propagation_numeric(self, func, std_energy_dict, counter, flag_dist_desorption="logNorm"):
        ### too slow in the case of many parameters
        variables = len(std_energy_dict)
        
        exp_samples = self.input_data_dict[counter]
        
        energy_dict_new = self.modify_energy_dict(counter)
        energy_dict_new_copy = energy_dict_new.copy()
        
        mean_energy_dict = {key: energy_dict_new_copy[key] for key in std_energy_dict.keys()}
        std_energy_dict = {key: mean_energy_dict[key] * std_energy_dict[key] if mean_energy_dict[key] > 0.0 \
            else std_energy_dict[key] for key in std_energy_dict.keys()}
        
        keys_list = list(std_energy_dict.keys())
        
        mean_energy_array = np.array([mean_energy_dict[key] for key in keys_list])
        std_energy_array = np.array([std_energy_dict[key] for key in keys_list])
        
        def pdf_function(*x):
            prod_func = 1.0
            for i in range(variables):
                prod_func *= sp.stats.norm.pdf(x[i], loc=mean_energy_array[i], scale=std_energy_array[i])
            return prod_func
        
        
        def integrand(*x):
            pdf_value = pdf_function(*x)
            
            energy_mod_dict = {key: x[i] for i, key in enumerate(keys_list)}
            energy_dict_new_copy.update(energy_mod_dict)
            
            _, recProb_aux, _, sucess = self.system.solve_system(exp_samples, energy_dict_new_copy, solver="fixed_point", max_time=self.max_time)
            
            if sucess == True:
                prob_value = np.sum(recProb_aux)
            else: 
                logging.warning(f"Simulation failed for index {counter}. Assigning high loss.")
                prob_value = 0.0
                
            return pdf_value * func(prob_value)
            
        
        bounds = [[mean_energy_dict[key] - 10 * std_energy_dict[key], mean_energy_dict[key] + 10 * std_energy_dict[key]] for key in keys_list]
        
        bounds = np.array(bounds)
        bounds[bounds < 0] = 0
        
        # Perform the integration
        start_time = time.time()
        integral_value, error = sp.integrate.nquad(integrand, bounds)
        end_time = time.time()
        
        logging.info(f"Integration time: {end_time - start_time:.2f} seconds")
        
        return integral_value, error
    
    
    
    
    
    def func_parameter_error_propagation_MC(self, std_energy_dict, counter, N=10_000, use_qmc=False, qmc_pow2=12, n_jobs=8):
        
        exp_samples = self.input_data_dict[counter]
        base_dict = self.modify_energy_dict(counter)
        
        keys = list(std_energy_dict.keys())
        mean_arr = np.array([base_dict[k] for k in keys])
        std_arr = np.array([
            (base_dict[k] * std_energy_dict[k] if base_dict[k] > 0 else std_energy_dict[k])
            for k in keys
        ])
        
        if use_qmc:
            sampler = qmc.Sobol(d=len(keys), scramble=True)
            u = sampler.random_base2(m=qmc_pow2)
            samples = norm.ppf(u, loc=mean_arr, scale=std_arr)
        else:
            samples = np.random.randn(N, len(keys)) * std_arr + mean_arr
        
        
        def run_sim(theta):
            params = base_dict.copy()
            for i, k in enumerate(keys):
                params[k] = theta[i]

            _, recProb, _, success = self.system.solve_system(exp_samples, params, solver="fixed_point", max_time=self.max_time)
            if not success:
                logging.warning(f"Simulation failed at counter={counter}, theta={theta}")
                val = 0.0
            else:
                val = np.sum(recProb)

            return val

        results = Parallel(n_jobs=n_jobs)(delayed(run_sim)(theta) for theta in samples)
        results = np.asarray(results)

        mean_est = float(np.mean(results))
        std_est = float(np.std(results, ddof=1))

        logging.info(
            f"Monte Carlo ({'QMC' if use_qmc else 'MC'}) N={len(results)} | mean={mean_est:.5g} | std={std_est:.5g}"
        )

        return mean_est, std_est, samples, results
    
    
    
    # def func_parameter_error_propagation_MC(self, std_energy_dict, counter, N=10_000, n_jobs=8, flag_dist_desorption="logNorm"):
        
    #     exp_samples = self.input_data_dict[counter]
    #     base_dict = self.modify_energy_dict(counter)
        
    #     keys = list(std_energy_dict.keys())
        
    #     keys_gaussian = [key for key in keys if key != 'E_de_O']
    #     keys_de = [key for key in keys if key == 'E_de_O']
        
    #     mean_arr = np.array([base_dict[k] for k in keys_gaussian])
    #     std_arr = np.array([
    #         (base_dict[k] * std_energy_dict[k] if base_dict[k] > 0 else std_energy_dict[k])
    #         for k in keys_gaussian
    #     ])
        
    #     if len(keys_gaussian) > 0:
    #         samples_gaussian = np.random.randn(N, len(keys_gaussian)) * std_arr + mean_arr
        
    #     if len(keys_de) > 0 and flag_dist_desorption == "logNorm":
            
    #         # sigma_log = 0.2
    #         # mu_log = np.exp(np.log(base_dict['E_de_O']) + sigma_log**2)
    #         # rv = sp.stats.lognorm(s=sigma_log, scale=mu_log)
            
    #         # samples_de = rv.rvs(size=N)
    #         # samples_de[samples_de < 0] = 0.0
            
    #         # vec_uniform = [30, 35, 40, 45, 50, 55, 60, 65]
    #         vec_uniform = [30, 31, 32]
            
    #         samples_de = np.random.choice(vec_uniform, size=N)
            
            
            
        
    #     elif len(keys_de) > 0 and flag_dist_desorption == "gaussian":

    #         samples_de = np.random.randn(N) * std_energy_dict['E_de_O'] + base_dict['E_de_O'] * std_energy_dict['E_de_O']
    #         samples_de[samples_de < 0] = 0.0
        
    #     else:
    #         pass

        
    #     if len(keys_gaussian) > 0 and len(keys_de) > 0:
    #         samples = np.concatenate((samples_gaussian, samples_de.reshape(-1, 1)), axis=1)
    #         keys = keys_gaussian + keys_de
    #         samples[samples < 0] = 0.0
            
    #     elif len(keys_gaussian) > 0 and len(keys_de) == 0:
    #         samples = samples_gaussian
            
    #     elif len(keys_gaussian) == 0 and len(keys_de) > 0:
    #         samples = samples_de.reshape(-1, 1)
    #         keys = keys_de
            
    #     else:
    #         samples = np.random.randn(N, len(keys)) * std_arr + mean_arrs
        
    #     def run_sim(theta):
    #         params = base_dict.copy()
    #         for i, k in enumerate(keys):
    #             params[k] = theta[i]
            
    #         _, recProb, _, success = self.system.solve_system(exp_samples, params, solver="fixed_point", max_time=self.max_time)
    #         if not success:
    #             logging.warning(f"Simulation failed at counter={counter}, theta={theta}")
    #             val = 0.0
    #         else:
    #             val = np.sum(recProb)

    #         return val

    #     results = Parallel(n_jobs=n_jobs)(delayed(run_sim)(theta) for theta in samples)
    #     results = np.asarray(results)

    #     mean_est = float(np.mean(results))
    #     std_est = float(np.std(results, ddof=1))

    #     logging.info(
    #         f"Monte Carlo N={len(results)} | mean={mean_est:.5g} | std={std_est:.5g}"
    #     )

    #     return mean_est, std_est, samples, results
    
    
