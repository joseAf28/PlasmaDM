import copy
import torch 
import numpy as np
import sympy as sy
import logging

import src.Simulator as sim
# import ratesSimb as rates
from typing import List, Dict, Callable, Any, Tuple, Optional
from torch.autograd.functional import jacobian


class SimDiff():
    
    def __init__(self, sim, func_optimization, params_default, gamma_exp_data, loss_function):
        
        self.sim = sim
        self.func_optimization = func_optimization
        self.loss_function = loss_function
        
        self.gamma_exp_torch = torch.tensor(gamma_exp_data, dtype=float)
        
        ##!* default initialization to get the rates to update
        new_update_dict = self.func_optimization(params_default)
        
        self.rx_update_vec = [f"r_{rx['id']}" for rx in new_update_dict]
        self.rx_no_update_vec = [f"r_{rx['id']}" for rx in sim.output_parser['reactions_list']  if not f"r_{rx['id']}" in self.rx_update_vec]
        
        self.idx_update_vec, self.idx_update_groups = self._select_reactions_update(sim.output_parser['reactions_list'], new_update_dict)
        self.idx_no_update_vec = [idx for idx, rx in enumerate(sim.output_parser['reactions_list']) if not f"r_{rx['id']}" in self.rx_update_vec]
        
        gamma_idx_vec = [idx for idx, rx in enumerate(self.sim.output_parser['reactions_list']) if rx.get('gamma', False)]
        self.gamma_update_idx = [idx for idx in gamma_idx_vec if idx in self.idx_update_vec]
        self.gamma_no_update_idx = [idx for idx in gamma_idx_vec if idx not in self.idx_update_vec]
        
        self.gamma_update_map = {idx:i for i, idx in enumerate(self.gamma_update_idx)}
        
        self.sym_model_func, self.F_torch = self.create_symbolic_system(self.rx_no_update_vec, self.rx_update_vec)
        
        
    def _select_reactions_update(self, base_reactions: List[Dict[str, Any]], 
                                param_update_instructions: List[Dict[str, Any]]
                                ) -> List[Dict[str, Any]]:
        
        updated_list_total = []
        updated_list_groups = []
        id_to_idx = {rx["id"]: i for i, rx in enumerate(base_reactions)}
        
        for instr in param_update_instructions:
            rx_id = instr.get("id", None)
            rate_type = instr.get("rate", None)
            updates = instr.get("model_dict", {})
            
            if not isinstance(updates, dict):
                logging.warning(f"Skipping invalid model_dict: {instr}")
                continue
            
            if rx_id is not None:
                if rx_id not in id_to_idx:
                    logging.warning(f"Reaction ID {rx_id} not found. Skipping.")
                    continue
                idx_list = [id_to_idx[rx_id]]
            elif rate_type is not None:
                idx_list = [
                    i for i, rx in enumerate(base_reactions)
                    if rx.get("rate") == rate_type
                ]
                if not idx_list:
                    logging.warning(f"No reaction with rate '{rate_type}'. Skipping.")
                    continue
            else:
                if len(updates) == 1:
                    key = next(iter(updates))
                    idx_list = [
                        i for i, rx in enumerate(base_reactions)
                        if key in rx.get("model_dict", {})
                    ]
                    if not idx_list:
                        logging.warning(f"No reaction contains param '{key}'. Skipping.")
                        continue
                else:
                    raise ValueError(f"Ambiguous instruction: {instr}")
            
            updated_list_groups.append(idx_list)
            
            for idx in idx_list:
                updated_list_total.append(idx)
                
        return list(set(updated_list_total)), updated_list_groups
    
    
    
    def _update_reactions_list_for_params(self, base_reactions: List[Dict[str, Any]], 
                                        param_update_instructions: List[Dict[str, Any]]
                                        ) -> List[Dict[str, Any]]:
        
        full_updated = copy.deepcopy(base_reactions)
        for i, group in enumerate(self.idx_update_groups):
            for idx in group:
                full_updated[idx]['model_dict'].update(param_update_instructions[i]['model_dict'])
        
        updated_list = [full_updated[idx] for idx in self.idx_update_vec]
        return updated_list
    
    
    
    def create_symbolic_system(self, no_update_rx, update_rx):
        torch_map = {
            'Add': torch.add,
            'Mul': torch.mul,
        }
        
        den_sym = sy.symbols(' '.join(self.sim.output_parser['species_model']), seq=True)
        update_rx_sym = sy.symbols(' '.join(update_rx), seq=True)
        no_update_rx_sym = sy.symbols(' '.join(no_update_rx), seq=True)
        
        sym_model_func = sy.lambdify((den_sym, update_rx_sym, no_update_rx_sym), self.sim.output_parser['model_sym'], modules=[torch_map, torch])
        
        def F_torch(den_vars, params_vars, no_up_rates, counter_x):
            
            new_update_dict = self.func_optimization(params_vars)
            updated_reactions_list = self._update_reactions_list_for_params(self.sim.output_parser['reactions_list'], new_update_dict)
            exp_data = self.sim.exp_data_arr[counter_x:counter_x+1]
            up_rates = self.sim.sim_rates_handler.compute_rates_simulation(exp_data, updated_reactions_list, flag_arr='torch').squeeze(0)
            
            return torch.stack(sym_model_func(den_vars, up_rates, no_up_rates))
        
        return sym_model_func, F_torch
    
    
    
    def T1_T2_const(self, rates_arr, counter):
        
        name2idx_map = {name:idx for idx, name in enumerate(self.sim.output_parser['species_model'])}
        size_sys = len(self.sim.output_parser['species_model'])
        
        ###! requires no grad
        T1_vec_const = torch.zeros((size_sys), dtype=torch.float64)
        T2_mat_const = torch.zeros((size_sys,size_sys), dtype=torch.float64)
        
        fluxO = self.sim.exp_data_arr[counter]['fluxO']
        
        S_specie_site_label = 'S'
        O2_gas_label = 'O2'
        for idx in self.gamma_no_update_idx:
            
            rx = self.sim.output_parser['reactions_list'][idx]
            left_reactants = rx['left']
            reactant_site_types = [ele.split('_')[1] for ele in left_reactants.keys() if '_' in ele] # e.g. 'F' from 'A_F'
            reactant_names_only = [ele.split('_')[0] for ele in left_reactants.keys() if '_' in ele] # e.g. 'A' from 'A_F'
            reactant_gas_phase = [ele for ele in left_reactants.keys() if '_' not in ele] # e.g. 'O2'
                    
            factor = 2.0 # Default
            if rx.get('gas_specie') == O2_gas_label:
                factor = 1.0
            elif 'F' in reactant_site_types and O2_gas_label in reactant_names_only + reactant_gas_phase : # reactant_names_only from A_F is A.
                factor = 1.0
                
            is_S_site_involved = (S_specie_site_label in reactant_site_types)
            prefactor_site_density = self.sim.const_dict['S0'] if is_S_site_involved else self.sim.const_dict['F0'] 
            
            if len(left_reactants) == 2:
                species = list(left_reactants.keys())
                species_idx = [name2idx_map[ele] for ele in species]
                T2_mat_const[species_idx[0], species_idx[1]] = T2_mat_const[species_idx[0], species_idx[1]] + prefactor_site_density*factor*rates_arr[counter, idx]/fluxO
                
            elif 2 in list(left_reactants.values()):
                specie = next(iter(left_reactants.keys()))
                position = name2idx_map[specie]
                T2_mat_const[position, position] = T2_mat_const[position, position] + prefactor_site_density*factor*rates_arr[counter, idx]/fluxO
                
            else:
                specie = next(iter(left_reactants.keys()))
                position = name2idx_map[specie]
                T1_vec_const[position] = T1_vec_const[position] + prefactor_site_density*factor*rates_arr[counter, idx]/fluxO
            
            
        return T1_vec_const, T2_mat_const
    
    
    
    def T1_T2_grad(self, update_rates: torch.Tensor, counter: int, S_specie_site_label: str = 'S', O2_gas_label: str = 'O2'):
        
        species_list = self.sim.output_parser["species_model"]
        name2idx_map = {name: idx for idx, name in enumerate(species_list)}
        
        size_sys = len(species_list)
        
        T1_vec_grad = torch.zeros((size_sys,), dtype=torch.float64)
        T2_mat_grad = torch.zeros((size_sys, size_sys), dtype=torch.float64)
        
        fluxO = self.sim.exp_data_arr[counter]["fluxO"]
        
        for i, rx_idx in enumerate(self.idx_update_vec):
            rx = self.sim.output_parser["reactions_list"][rx_idx]
            
            bool_gamma = rx.get("gamma", False)
            if not bool_gamma:
                continue
            
            left_reactants = rx["left"]
            reactant_site_types = [k.split("_")[1] for k in left_reactants if "_" in k]
            reactant_names_only = [k.split("_")[0] for k in left_reactants if "_" in k]
            reactant_gas_phase = [k for k in left_reactants if "_" not in k]
            
            rate = update_rates[i]  # a Torch‐scalar
            
            # Compute factor just like in T1_T2_const:
            factor = 2.0
            if rx.get("gas_specie") == O2_gas_label:
                factor = 1.0
            elif "F" in reactant_site_types and O2_gas_label in reactant_names_only + reactant_gas_phase:
                factor = 1.0
                
            is_S_site = (S_specie_site_label in reactant_site_types)
            density_pref = self.sim.const_dict["S0"] if is_S_site else self.sim.const_dict["F0"]
            
            if len(left_reactants) == 2:
                species = list(left_reactants.keys())
                idx0, idx1 = name2idx_map[species[0]], name2idx_map[species[1]]
                T2_mat_grad[idx0, idx1] += density_pref * factor * rate / fluxO
            elif 2 in left_reactants.values():
                sp = next(iter(left_reactants.keys()))
                pos = name2idx_map[sp]
                T2_mat_grad[pos, pos] += density_pref * factor * rate / fluxO
            else:
                sp = next(iter(left_reactants.keys()))
                pos = name2idx_map[sp]
                T1_vec_grad[pos] += density_pref * factor * rate / fluxO
                
        return T1_vec_grad, T2_mat_grad
    
    
    
    
    ###* compute the derivatives of the F and T1 and T2
    def objective_function_grad(self, params, frac_solutions_arr, rates_arr, gamma_predicted_arr, residual_flag=False):
        
        frac_solutions_torch = torch.tensor(frac_solutions_arr, dtype=torch.float64)
        gamma_predicted_torch = torch.tensor(gamma_predicted_arr, dtype=torch.float64)
        
        num_data = frac_solutions_torch.shape[0]
        num_species = frac_solutions_torch.shape[1]
        
        params_torch = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        num_params   = params_torch.numel()   
        no_update_rates = torch.tensor(rates_arr[:, self.idx_no_update_vec], dtype=torch.float64)
        
        grad_gamma = torch.zeros((num_data, num_params), dtype=torch.float64)
        
        for counter in range(num_data):
            
            exp_data_up = self.sim.exp_data_arr[counter:counter+1]
            y_saddle = frac_solutions_torch[counter]
            
            new_update_dict = self.func_optimization(params_torch)
            updated_reactions_list = self._update_reactions_list_for_params(self.sim.output_parser['reactions_list'], new_update_dict)
            
            up_row = self.sim.sim_rates_handler.compute_rates_simulation(exp_data_up, updated_reactions_list, flag_arr='torch').squeeze(0)
            no_up_row = no_update_rates[counter]
            
            Jx = jacobian(lambda den: self.F_torch(den, params_torch, no_up_row, counter), y_saddle, create_graph=True)
            Jtheta = jacobian(lambda theta: self.F_torch(y_saddle, theta, no_up_row, counter), params_torch, create_graph=True)
            
            sol = torch.linalg.lstsq(Jx, Jtheta)
            dy_dtheta = - sol.solution
            
            ###* Check residual
            resid = (Jx @ dy_dtheta + Jtheta).abs().sum().item()
            if resid > 1e-2 and residual_flag:
                print(f"FIxed Point Error residual {resid} for condition {counter}")
            
            T1_vec_const, T2_mat_const = self.T1_T2_const(rates_arr, counter)
            T1_vec_grad, T2_mat_grad = self.T1_T2_grad(up_row, counter)
            
            ##* dgamma/dtheta implicit path 
            T1_total = T1_vec_const + T1_vec_grad
            T2_total = T2_mat_const + T2_mat_grad
            
            mat = T1_total + (T2_total + T2_total.T) @ y_saddle
            d_gamma_implicit = (mat.unsqueeze(0) @ dy_dtheta).squeeze()
            
            ###* dgamma/dtheta explicit path
            if len(self.gamma_update_idx) != 0:
                
                gamma1 = (y_saddle * T1_vec_grad).sum()
                gamma2 = (y_saddle.unsqueeze(0) @ (T2_mat_grad @ y_saddle.unsqueeze(1))).squeeze()
                gamma12 = gamma1 + gamma2
                
                d_gamma_explicit = torch.autograd.grad(
                    outputs=gamma12,
                    inputs=params_torch,
                    retain_graph=True,       
                    create_graph=False       
                )[0]
            else:
                d_gamma_explicit = torch.zeros_like(d_gamma_implicit)
            
            grad_gamma[counter, :] = d_gamma_explicit + d_gamma_implicit
            
            
        factor = (2.0 / self.gamma_exp_torch**2) * (gamma_predicted_torch - self.gamma_exp_torch)
        dJdtheta = (grad_gamma.T @ factor) / float(num_data)
        return dJdtheta
    


    ###* compute the derivatives of the F and T1 and T2
    def objective_function_grad_element_wise(self, params, frac_solutions_arr, rates_arr, gamma_predicted_arr, residual_flag=False):
        
        frac_solutions_torch = torch.tensor(frac_solutions_arr, dtype=torch.float64)
        gamma_predicted_torch = torch.tensor(gamma_predicted_arr, dtype=torch.float64)
        
        num_data = frac_solutions_torch.shape[0]
        num_species = frac_solutions_torch.shape[1]
        
        params_torch = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        num_params   = params_torch.numel()   
        no_update_rates = torch.tensor(rates_arr[:, self.idx_no_update_vec], dtype=torch.float64)
        
        grad_gamma = torch.zeros((num_data, num_params), dtype=torch.float64)
        
        for counter in range(num_data):
            
            exp_data_up = self.sim.exp_data_arr[counter:counter+1]
            y_saddle = frac_solutions_torch[counter]
            
            new_update_dict = self.func_optimization(params_torch)
            updated_reactions_list = self._update_reactions_list_for_params(self.sim.output_parser['reactions_list'], new_update_dict)
            
            up_row = self.sim.sim_rates_handler.compute_rates_simulation(exp_data_up, updated_reactions_list, flag_arr='torch').squeeze(0)
            no_up_row = no_update_rates[counter]
            
            J_den = jacobian(lambda den: self.F_torch(den, params_torch, no_up_row, counter), y_saddle)
            J_theta = jacobian(lambda theta: self.F_torch(y_saddle, theta, no_up_row, counter), params_torch)
            
            sol = torch.linalg.lstsq(J_den, J_theta)
            dy_dtheta = - sol.solution
            
            ###* Check residual
            resid = (J_den @ dy_dtheta + J_theta).abs().sum().item()
            if resid > 1e-2 and residual_flag:
                print(f"FIxed Point Error residual {resid} for condition {counter}")
            
            T1_vec_const, T2_mat_const = self.T1_T2_const(rates_arr, counter)
            T1_vec_grad, T2_mat_grad = self.T1_T2_grad(up_row, counter)
            
            ##* dgamma/dtheta implicit path 
            T1_total = T1_vec_const + T1_vec_grad.detach()
            T2_total = T2_mat_const + T2_mat_grad.detach()
            
            mat = T1_total + (T2_total + T2_total.T) @ y_saddle
            d_gamma_implicit = (mat.unsqueeze(0) @ dy_dtheta).squeeze()
            
            ###* dgamma/dtheta explicit path
            if len(self.gamma_update_idx) != 0:
                
                gamma1 = (y_saddle * T1_vec_grad).sum()
                gamma2 = (y_saddle.unsqueeze(0) @ (T2_mat_grad @ y_saddle.unsqueeze(1))).squeeze()
                gamma12 = gamma1 + gamma2
                
                d_gamma_explicit = torch.autograd.grad(
                    outputs=gamma12,
                    inputs=params_torch,
                    retain_graph=True,       
                    create_graph=False       
                )[0]
            else:
                d_gamma_explicit = torch.zeros_like(d_gamma_implicit)
            
            grad_gamma[counter, :] = d_gamma_explicit + d_gamma_implicit
            
        return grad_gamma
