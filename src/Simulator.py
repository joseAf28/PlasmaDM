import numpy as np
import scipy as sp
import sympy as sy
import time
import logging
from typing import Dict, List, Callable, Any, Optional, Union, Tuple

import src.SimParser as sim_par
import src.SimRater as sim_rater


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


class Simulator():
    """
    Controls and calls the simulation process:
    1. Parses reaction models.
    2. Prepares experimental data and computes reaction rates.
    3. Solves the system of ODEs or finds steady-state solutions.
    4. Computes macroscopic observable: gamma.
    """
    
    def __init__(self, reactions_file: str, const_dict: Dict[str, float], exp_data_file: str, initial_state_dict: Dict[str, float], 
                flag_debugging: bool = True, time_space_params: Union[np.ndarray, Dict[str, float]] = {'start': 0, 'stop': 100, 'points': 500},
                transformations_exp: Optional[Dict[str, Callable]] = None):
        
        self.const_dict: Dict[str, float] = const_dict

        if isinstance(time_space_params, np.ndarray):
            self.timeSpace: np.ndarray = time_space_params
        elif isinstance(time_space_params, dict):
            self.timeSpace = np.linspace(
                time_space_params.get('start', 0),
                time_space_params.get('stop', 100),
                int(time_space_params.get('points', 500))
            )
        else:
            raise TypeError("time_space_params must be a NumPy array or a dictionary.")

        logging.info("Initializing Simulator...")

        ###* Parse reactions and create physical model
        parser = sim_par.SimulatorParser(reactions_file, self.const_dict)
        self.output_parser: Dict[str, Any] = parser.create_physical_model(flag_debugging=flag_debugging)
        
        if 'error' in self.output_parser:
            logging.error(f"Failed to create physical model: {self.output_parser['error']}")
            raise RuntimeError(f"Model creation failed: {self.output_parser['error']}")

        self.model_species: List[str] = self.output_parser['species_model']
        if not self.model_species:
            logging.warning("No species in the model from parser. Simulation might not be meaningful.")
            self.initial_state: List[float] = []
        else:
            try:
                self.initial_state = [initial_state_dict[ele] for ele in self.model_species]
            except KeyError as e:
                logging.error(f"Missing initial state for species: {e}. Model species are: {self.model_species}")
                raise ValueError(f"Invalid initial_state_dict. Missing key for {e}. Expected keys: {self.model_species}")
        
        ###* Prepare experimental data and rate computation setup according to the default conditions
        self.sim_rates_handler = sim_rater.SimulatorRates(self.const_dict, exp_data_file, self.output_parser)
        self.exp_data_arr, self.gamma_exp_data_arr = self.sim_rates_handler.prepare_experimental_data(transformations_exp=transformations_exp)
        self.sim_rates_handler.preload_rates_functions()
        
        logging.info("Simulator initialization complete.")
    
    
    
    def compute_rates(self, exp_data_arr: np.ndarray, reaction_list: List[Dict[str, Any]]) -> np.ndarray:
        return self.sim_rates_handler.compute_rates_simulation(exp_data_arr, reaction_list)
    
    
    
    def _system_ode(self, t: float, X: np.ndarray, rates_vec: np.ndarray) -> np.ndarray:
        try:
            return self.output_parser['model'](tuple(X), tuple(rates_vec))
        except Exception as e:
            logging.error(f"Error in system_ode evaluation: {e}. X={X}, rates_vec={rates_vec}")
            # Return an array of NaNs of the correct size to allow solver to handle error
            return np.full_like(X, np.nan)
    
    
    
    def _solution_check(self, sol_point: np.ndarray, rates_vec: np.ndarray, tolerance: float = 1e-4) -> bool:
        if not self.model_species: # No species, no ODEs to check
            return True
        derivatives = self._system_ode(0, sol_point, rates_vec) # time=0 is arbitrary for steady state check
        absolute_error = np.sum(np.abs(derivatives))
        is_steady = absolute_error < tolerance
        if not is_steady:
            logging.debug(f"Solution check failed: sum(abs(dX/dt)) = {absolute_error:.2e} > {tolerance:.0e}")
        return is_steady
    
    
    
    def solve_ode_for_condition(self, rates_vec: np.ndarray, method: str = "stiff", 
                                odesolver_options: Optional[Dict] = None) -> Tuple[np.ndarray, bool]:
        if not self.model_species: # No species to solve for
            return np.array([]), True
        
        default_options_stiff = {'method': "BDF", 'atol': 1e-6, 'rtol': 1e-6}
        if odesolver_options:
            default_options_stiff.update(odesolver_options)
            
        solution_trajectory: np.ndarray
        success_flag: bool = False
        
        if method == "simple": # Using odeint
            time_points = np.linspace(self.timeSpace[0], self.timeSpace[-1], len(self.timeSpace)) 
            try:
                solution_trajectory = sp.integrate.odeint(
                    func=lambda X, t: self._system_ode(t, X, rates_vec), # order for odeint
                    y0=self.initial_state,
                    t=time_points,
                    atol=default_options_stiff.get('atol'),
                    rtol=default_options_stiff.get('rtol')
                )
                X_final = solution_trajectory[-1]
                success_flag = self._solution_check(X_final, rates_vec)
            except Exception as e:
                logging.error(f"ODE solver (odeint) failed: {e}")
                X_final = np.full_like(self.initial_state, np.nan) # Indicate failure
                
        elif method == "stiff": # Using solve_ivp
            try:
                sol_obj = sp.integrate.solve_ivp(
                    fun=lambda t, X: self._system_ode(t, X, rates_vec),
                    t_span=(self.timeSpace[0], self.timeSpace[-1]),
                    y0=self.initial_state,
                    method=default_options_stiff.get('method', 'BDF'),
                    t_eval=self.timeSpace, # Evaluate at these specific time points
                    atol=default_options_stiff.get('atol'),
                    rtol=default_options_stiff.get('rtol'),
                    **{k:v for k,v in default_options_stiff.items() if k not in ['method','atol','rtol']} # Pass other opts
                )
                if sol_obj.success:
                    X_final = sol_obj.y[:, -1]
                    success_flag = self._solution_check(X_final, rates_vec) # Check if final point is steady
                else:
                    logging.warning(f"ODE solver (solve_ivp) reported failure: {sol_obj.message}")
                    X_final = sol_obj.y[:, -1] if sol_obj.y.size > 0 else np.full_like(self.initial_state, np.nan)

            except Exception as e:
                logging.error(f"ODE solver (solve_ivp) failed with exception: {e}")
                X_final = np.full_like(self.initial_state, np.nan)
        else:
            raise ValueError("Invalid ODE method - choose between 'simple' and 'stiff'")
        
        return X_final, success_flag
    
    
    
    def solve_fixed_point_for_condition(self,
                                        rates_vec: np.ndarray,
                                        max_integration_time_guess: Optional[float] = 10.0, # For initial guess integration
                                        rootfinder_options: Optional[Dict] = None
                                        ) -> Tuple[np.ndarray, bool]:
        
        if not self.model_species:
            return np.array([]), True
        
        refined_guess = np.array(self.initial_state) # Start with the global initial state

        ###* Refine guess with a short ODE integration
        if max_integration_time_guess is not None and max_integration_time_guess > 0:
            short_t_span = (self.timeSpace[0], max_integration_time_guess)
            short_t_eval = np.array([max_integration_time_guess])

            try:
                sol_short = sp.integrate.solve_ivp(
                    fun=lambda t, X: self._system_ode(t, X, rates_vec),
                    t_span=short_t_span,
                    y0=self.initial_state,
                    method="Radau",
                    t_eval=short_t_eval,
                    atol=1e-5, rtol=1e-5
                )
                if sol_short.success and sol_short.y.size > 0:
                    refined_guess = sol_short.y[:, -1]
                else:
                    logging.warning(f"Initial guess integration failed or produced no points: {sol_short.message}")
            except Exception as e:
                logging.warning(f"Exception during initial guess integration: {e}")

        ###* Call the minimizer
        current_rootfinder_options = {'method': 'hybr'} # default
        if rootfinder_options:
            current_rootfinder_options.update(rootfinder_options)

        steady_state_solution: np.ndarray
        success_flag: bool = False
        try:
            root_func = lambda X_ss_guess: self._system_ode(0, X_ss_guess, rates_vec)
            
            sol_obj_root = sp.optimize.root(
                fun=root_func,
                x0=refined_guess,
                method=current_rootfinder_options['method'],
                tol=current_rootfinder_options.get('tol', 1e-6)
            )
            if sol_obj_root.success:
                steady_state_solution = sol_obj_root.x
                success_flag = self._solution_check(steady_state_solution, rates_vec)
                if not success_flag:
                    logging.warning(f"Root finder converged (message: {sol_obj_root.message}), but solution check failed.")
            else:
                logging.warning(f"Fixed point solver (scipy.optimize.root) failed: {sol_obj_root.message}")
                steady_state_solution = sol_obj_root.x # Still return the result it found, might be close
                
        except (ValueError, TypeError) as e:    # Catch specific errors from optimize.root if known
            logging.error(f"Fixed point solver failed with data-related error: {e}")
            steady_state_solution = np.full_like(refined_guess, np.nan)
        except Exception as e:                  # Catch other unexpected errors
            logging.error(f"Fixed point solver failed with unexpected exception: {e}")
            steady_state_solution = np.full_like(refined_guess, np.nan) # Or return `refined_guess`
            
        return steady_state_solution, success_flag
    
    
    
    def _prepare_gamma_reactions(self, S_specie_site_label: str = 'S', O2_gas_label: str = 'O2') -> List[Dict[str, Any]]:
        gamma_reactions_info: List[Dict[str, Any]] = []
        
        for idx, reaction_details in enumerate(self.output_parser['reactions_list']):
            if not reaction_details.get('gamma', False): # Check if 'gamma' key exists and is True
                continue
            
            left_reactants = reaction_details['left']
            reactant_site_types = [ele.split('_')[1] for ele in left_reactants.keys() if '_' in ele] # e.g. 'F' from 'A_F'
            reactant_names_only = [ele.split('_')[0] for ele in left_reactants.keys() if '_' in ele] # e.g. 'A' from 'A_F'
            reactant_gas_phase = [ele for ele in left_reactants.keys() if '_' not in ele] # e.g. 'O2'
            
            factor = 2.0 # Default
            if reaction_details.get('gas_specie') == O2_gas_label:
                factor = 1.0
            elif 'F' in reactant_site_types and O2_gas_label in reactant_names_only + reactant_gas_phase : # reactant_names_only from A_F is A.
                factor = 1.0

            is_S_site_involved = (S_specie_site_label in reactant_site_types)

            gamma_reactions_info.append({
                'rate_constant_index': idx, # Index in the rate_constants_arr
                'reaction_id': reaction_details['id'],
                'reactant_stoichiometry': left_reactants, # e.g. {'A_F': 1, 'O2': 1}
                'factor_for_gamma': factor,
                'is_S_site_related': is_S_site_involved
            })
        return gamma_reactions_info
    
    
    
    def compute_gammas_for_condition(self,
                                    steady_state_solution: np.ndarray, # X_ss for one condition
                                    exp_condition_data: Dict[str, Any], # Single exp_data_arr element
                                    rate_constants_for_condition: np.ndarray, # Single row from rate_constants_arr
                                    gamma_reactions_info: List[Dict[str, Any]],
                                    flux_O_label: str = 'fluxO'
                                    ) -> Dict[str, float]:
        
        if not self.model_species: # No species, no gammas
            return {}
        
        ###* Create a mapping from species name (in model) to its index in steady_state_solution
        species_to_idx_map = {name: i for i, name in enumerate(self.model_species)}
        
        ###! the theoretical model assumes names present in self.model_species
        current_gammas: Dict[str, float] = {}
        total_flux = exp_condition_data.get(flux_O_label)
        
        if total_flux is None or total_flux == 0:
            logging.warning(f"'{flux_O_label}' is missing or zero in experimental data. Gammas will be NaN/Inf.")
            for reaction_info in gamma_reactions_info:
                current_gammas[f"r_{reaction_info['reaction_id']}"] = np.nan
            return current_gammas
        
        
        for rxn_info in gamma_reactions_info:
            rate_k_value = rate_constants_for_condition[rxn_info['rate_constant_index']]
            
            # Calculate Product_i(C_i^nu_ij) - the product of steady-state terms
            ss_concentration_product = 1.0
            try:
                for specie, stoich_nu in rxn_info['reactant_stoichiometry'].items():
                    if specie in species_to_idx_map: # Species is part of the ODE model
                        idx = species_to_idx_map[specie]
                        ss_concentration_product *= steady_state_solution[idx] ** stoich_nu
                    else:
                        logging.error(f"Species '{specie}' in gamma reaction '{rxn_info['reaction_id']}' "
                                    "not found in model species or experimental conditions. Cannot compute gamma.")
                        ss_concentration_product = np.nan # Mark as invalid
                        break
                if np.isnan(ss_concentration_product):
                    current_gammas[f"r_{rxn_info['reaction_id']}"] = np.nan
                    continue
                
            except ZeroDivisionError: # Should not happen 
                ss_concentration_product = np.nan
                logging.warning(f"Potential issue with zero concentration to a power for reaction {rxn_info['reaction_id']}")
            except OverflowError:
                ss_concentration_product = np.inf
                logging.warning(f"Overflow computing steady-state product for reaction {rxn_info['reaction_id']}")
                
            prefactor_site_density = self.const_dict['S0'] if rxn_info['is_S_site_related'] else self.const_dict['F0']
            gamma_value = (rxn_info['factor_for_gamma'] * rate_k_value *
                        ss_concentration_product * prefactor_site_density) / total_flux
            
            current_gammas[f"r_{rxn_info['reaction_id']}"] = gamma_value
            
        return current_gammas
    
    
    
    def solve_all_conditions(self,
                            exp_data_arr: np.ndarray,
                            reactions_list: List[Dict[str, Any]],
                            solver_type: str = "fixed_point", # "fixed_point" or "odeint"
                            solver_options: Optional[Dict] = None,
                            gamma_S_site_label: str = 'S',
                            gamma_O2_label: str = 'O2',
                            gamma_flux_label: str = 'fluxO'
                            ) -> Tuple[np.ndarray, np.ndarray]: # (steady_solutions_all_exp, gammas_all_exp)
        
        rate_constants_arr = self.compute_rates(exp_data_arr, reactions_list)
        
        num_experiments = len(exp_data_arr)
        num_model_species = len(self.model_species)
        
        if num_experiments == 0:
            logging.warning("No experimental data to process.")
            return np.array([]), np.array([])
        if num_model_species == 0 and solver_type != "none": # if no species, solution is empty
            logging.warning("No model species defined. Solver will not run.")
        
        frac_steady_sol_arr = np.full((num_experiments, num_model_species), np.nan, dtype=float)
        gammas_results_list_of_dicts: List[Optional[Dict[str, float]]] = [None] * num_experiments
        
        ###* Pre-process reactions relevant for gamma calculation once
        gamma_reactions_details = self._prepare_gamma_reactions(gamma_S_site_label, gamma_O2_label)
        
        logging.info(f"Starting to solve for {num_experiments} experimental conditions using '{solver_type}'.")
        
        ##! Serial execution
        for i in range(num_experiments):
            logging.debug(f"Processing experimental condition {i+1}/{num_experiments}")
            current_rate_constants = rate_constants_arr[i] # k values for this experiment
            
            if num_model_species == 0: # No species, effectively success with empty solution
                current_solution = np.array([])
                success = True
            elif solver_type == "fixed_point":
                current_solution, success = self.solve_fixed_point_for_condition(current_rate_constants, **(solver_options or {}))
            elif solver_type == "odeint":
                current_solution, success = self.solve_ode_for_condition(current_rate_constants, method="stiff", odesolver_options=solver_options) # Default to stiff
            else:
                raise ValueError("Invalid solver_type. Choose 'fixed_point' or 'odeint'.")
            
            if success:
                frac_steady_sol_arr[i, :] = current_solution
                logging.debug(f"Condition {i+1}: Solver converged.")
            else:
                # Solution might still be somewhat valid, or NaNs. Store it anyway.
                frac_steady_sol_arr[i, :] = current_solution
                logging.warning(f"Condition {i+1} ({exp_data_arr[i]}): Solver failed or solution check failed.")
            
            ###* Compute the gammas
            if gamma_reactions_details and (success or not np.all(np.isnan(current_solution))): # Proceed if solution is not all NaN
                gammas_for_cond = self.compute_gammas_for_condition(current_solution, exp_data_arr[i], current_rate_constants, gamma_reactions_details,gamma_flux_label)
                gammas_results_list_of_dicts[i] = gammas_for_cond
            elif not gamma_reactions_details:
                gammas_results_list_of_dicts[i] = {} # No gamma reactions, empty dict
            else: # Solver failed badly
                gammas_results_list_of_dicts[i] = {f"r_{rxn['reaction_id']}": np.nan for rxn in gamma_reactions_details}
        
        logging.info("All experimental conditions processed.")
        return frac_steady_sol_arr, rate_constants_arr, np.array(gammas_results_list_of_dicts, dtype=object)