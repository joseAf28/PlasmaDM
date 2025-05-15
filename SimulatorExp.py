import numpy as np
import scipy as sp
import sympy as sy
import time

import SimulatorParser as sim_par
import SimulatorRater as sim_rater


class Simulator():
    
    def __init__(self, reactions_file, const_dict, exp_data_file, initial_state_dict,
                flag_debugging=True, timeSpace=np.linspace(0, 1_00, 5_00), transformations_exp=None):
        
        self.const_dict = const_dict
        self.timeSpace = timeSpace
        
        ###* call the Parser and build the symbolic system of differential equations
        parser = sim_par.SimulatorParser(reactions_file, const_dict)
        self.output_parser = parser.create_physical_model(flag_debugging=flag_debugging)
        
        ###* define the initial state
        try:
            self.initial_state = [initial_state_dict[ele] for ele in self.output_parser['species_model']]
        except:
            raise ValueError(f"Invalid initial state dictionary. The model species are: {self.output_parser['species_model']} ")
        
        ###* compute the rates of the reactions included in list for all the experimental data 
        ###! and prepare the experimental data 
        self.sim_rates = sim_rater.SimulatorRates(reactions_file, const_dict, exp_data_file, self.output_parser)
        
        self.exp_data_arr, self.gamma_exp_data_arr = self.sim_rates.prepare_experimental_data(transformations_exp=transformations_exp)
        self.sim_rates.preload_rates_functions()
        
        
    def compute_rates(self, exp_data_arr, reaction_list):
        rates_calculation_arr = self.sim_rates.compute_rates_simulation(exp_data_arr, reaction_list)
                
        return rates_calculation_arr
    
    
    def system_ode(self, X, t, rates_vec):
        
        return self.output_parser['model'](X, rates_vec)
    
    
    def solve_ode(self, rates_vec, method="simple"):
        
        if method == "simple":
            timeSpace = np.linspace(0, 10.0, 1_00)
            solution = sp.integrate.odeint(
                func=lambda X, t: self.system_ode(X, t, rates_vec),
                y0=self.initial_state,
                t=timeSpace
            )
            
        elif method == "stiff":
            sol = sp.integrate.solve_ivp(
                fun=lambda t, X: self.system_ode(X, t, rates_vec),
                t_span=(self.timeSpace[0], self.timeSpace[-1]),
                y0=self.initial_state,
                method="BDF",
                t_eval=self.timeSpace,
                atol=1e-5, rtol=1e-5
            )
            solution = sol.y.T
            
        else:
            raise ValueError("Invalid method - choose between 'simple' and 'stiff'")
        
        sucess = self.solution_check(solution, rates_dict_tuple)
        return solution[-1], sucess
    
    
    def solve_fixed_point(self, rates_vec, max_time=15):
        
        ###* First inital 'educated' guess
        short_time = np.linspace(self.timeSpace[0], self.timeSpace[min(30, len(self.timeSpace)-1)], 30)
        
        start_time = time.time()
        events = None
        if max_time is not None:
            def timeout_event(t, X):
                elapsed = time.time() - start_time
                return max_time - elapsed
            
            timeout_event.terminal = True
            timeout_event.direction = -1
            events = [timeout_event]
            
        sol_short = sp.integrate.solve_ivp(
            fun=lambda t, X: self.system_ode(X, t, rates_vec),
            t_span=(short_time[0], short_time[-1]),
            y0=self.initial_state,
            method="Radau",
            t_eval=short_time,
            atol=1e-5, rtol=1e-5,
            events=events
        )
        
        refined_guess = sol_short.y.T[-1]
        
        ###* Attempt to find the fixed point using the refined guess
        try:
            sol = sp.optimize.root(self.system_ode, refined_guess, args=(0, rates_vec), method="hybr")
            success = self.solution_check(np.atleast_2d(sol.x), rates_vec)
        except Exception as e:
            print("Fixed point solver failed with message: ", e)
            success = False
            sol = self.initial_state
        
        return sol.x, success
    
    
    def solution_check(self, sol, rates_vec):
        vec_aux = self.system_ode(sol[-1], 0, rates_vec)
        absolute_error = np.sum(np.abs(vec_aux))
        
        if absolute_error > 1e-4:
            return False
        else:
            return True
    
    
    def compute_gammas(self, steady_solutions_arr, exp_data_arr, rates_calculation_arr, S_specie = 'S', O2_specie='O2'):
        
        steady_dict = {
            specie: steady_solutions_arr[:,idx] 
            for idx, specie in enumerate(self.output_parser['species_model'])
        }
        
        gamma_reactions = []
        for idx, r in enumerate(self.output_parser['reactions_list']):
            if not r['gamma']:
                continue
            left = r['left']
            fams = [ele.split('_')[1] for ele in left.keys()]
            species = [ele.split('_')[0] for ele in left.keys()]
            
            if r['gas_specie'] == O2_specie or (list(set(fams))[0] == 'F' and O2_specie in species):
                factor = 1.0
            else:
                factor = 2.0
            
            flag_S = (S_specie in fams)
            
            gamma_reactions.append({
                'idx':    idx,
                'id':     r['id'],
                'left':  left,
                'factor': factor,
                'flag_S': flag_S
            })
        
        n_exp = exp_data_arr.shape[0]
        gammas = np.empty(n_exp, dtype=object)
        
        for i in range(n_exp):
            rates = rates_calculation_arr[i]
            fluxO = exp_data_arr[i]['fluxO']
            one_gamma = {}
            
            for rxn in gamma_reactions:
                ##* product of steady-state terms^nu
                terms = (
                    steady_dict[specie][i] ** nu for specie, nu in rxn['left'].items()
                )
                ss_product = np.prod(list(terms))
                prefactor = self.const_dict['S0'] if rxn['flag_S'] else self.const_dict['F0']
                value = rxn['factor'] * rates[rxn['idx']] * ss_product * prefactor / fluxO
                
                one_gamma[f"r_{rxn['id']}"] = value
            
            gammas[i] = one_gamma
        return gammas
    
    
    def solve_system(self, exp_data_arr, rates_calculation_arr, solver="fixed_point", max_time=15):
        
        frac_steady_sol_arr = np.zeros((len(exp_data_arr), len(self.output_parser['species_model'])), dtype=float)
        for counter in range(self.exp_data_arr.shape[0]):
            
            rates_vec = rates_calculation_arr[counter]
            
            if solver == "fixed_point":
                sol, success = self.solve_fixed_point(rates_vec, max_time=max_time)
            elif solver == "odeint":
                sol, success = self.solve_ode(rates_vec, method="odeint")
            else:
                raise ValueError("Invalid solver - choose between 'fixed_point' and 'odeint'")
            
            if success:
                frac_steady_sol_arr[counter] = sol
            else:
                frac_steady_sol_arr[counter] = sol
                print(f"In {exp_data_arr[counter]} conditions, the solver failed to converge")
                
        ###* compute gammas
        gammas_result_arr = self.compute_gammas(frac_steady_sol_arr, exp_data_arr, rates_calculation_arr)
        
        return frac_steady_sol_arr, gammas_result_arr
    
    
    
    # def solve_system_uncertainty(self, exp_data_arr, rates_calculation_mod_arr, func_uncertainty_dict=None, solver="fixed_point", max_time=15):
        
    #     if func_uncertainty_dict is not None:
    #         exp_data_mod_arr = func_uncertainty_dict(exp_data_arr)
    #     else:
    #         exp_data_mod_arr = exp_data_arr
        
        
    #     frac_steady_sol_arr = np.zeros((len(exp_data_mod_arr), len(self.output_parser['species_model'])), dtype=float)
    #     for counter in range(exp_data_mod_arr.shape[0]):
            
    #         rates_vec = rates_calculation_mod_arr[counter]
            
    #         if solver == "fixed_point":
    #             sol, success = self.solve_fixed_point(rates_vec, max_time=max_time)
    #         elif solver == "odeint":
    #             sol, success = self.solve_ode(rates_vec, method="odeint")
    #         else:
    #             raise ValueError("Invalid solver - choose between 'fixed_point' and 'odeint'")
            
    #         if success:
    #             frac_steady_sol_arr[counter] = sol
    #         else:
    #             frac_steady_sol_arr[counter] = sol
    #             print(f"In {exp_data_mod_arr[counter]} conditions, the solver failed to converge")
                
    #     ###* compute gammas
    #     gammas_result_arr = self.compute_gammas(frac_steady_sol_arr, exp_data_mod_arr, rates_calculation_mod_arr)
        
    #     return frac_steady_sol_arr, gammas_result_arr
        
        
        
        
        