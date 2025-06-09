from pathos.multiprocessing import ProcessPool
import numpy as np
import scipy as sp
import os
import logging

use_logging = False


def store_results(results, loss, output_file):
    
    with open(output_file, "w") as f:
        f.write("Model-free optimization results:\n")
        for result, loss in zip(results, loss):
            f.write(f"params: {result}, loss: {loss}\n\n")
        



def pipeline1(opt_object, config, num_workers=9, local_refinemnt=False):
    """
    Performs a hybrid optimization: global search with Differential Evolution,
    followed by local search refinement on the best candidates.
    """
    bounds = config.get("bounds")
    de_num_iterations = config.get("de_num_iterations", 5) # How many separate DE runs
    de_max_generations = config.get("de_max_generations", 100)
    de_polish = config.get("de_polish", False) # DE's internal polishing
    de_pop_size = config.get("de_population_size", 15) # Typical DE parameter

    local_attempts_per_candidate = config.get("local_search_attempts_per_candidate", 3)
    perturbation_factor = config.get("local_search_perturbation_factor", 0.05) # Relative to bounds width
    top_k_candidates_for_local = config.get("local_search_top_k_candidates", 3)

    if bounds is None:
        logging.error("Bounds must be provided in the configuration.")
        return None, None

    bounds_array = np.array(bounds)
    if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
        logging.error("Bounds must be a list of (min, max) tuples.")
        return None, None

    global_search_candidates_params: List[np.ndarray] = []
    global_search_candidates_loss: List[float] = []
    
    ####* Global Search: Differential Evolution 
    if use_logging:
        logging.info(f"Starting Differential Evolution with {de_num_iterations} iteration(s), "
                    f"{de_max_generations} generations each, using {num_workers} worker(s).")
        
    pool = ProcessPool(nodes=num_workers)
    try:
        for i in range(de_num_iterations):
            if use_logging:
                logging.info(f"DE Iteration {i+1}/{de_num_iterations}...")
            de_result = sp.optimize.differential_evolution(
                func=opt_object.objective_function,
                bounds=bounds,
                strategy='best1bin',
                maxiter=de_max_generations,
                popsize=de_pop_size,
                tol=0.01,
                recombination=0.7,
                polish=de_polish,
                disp=True,          # Quieter output, rely on logging
                workers=pool.map,   # Parallel evaluation of population
                updating="deferred" # Good for parallel,
            )
                
            global_search_candidates_params.append(de_result.x)
            global_search_candidates_loss.append(de_result.fun)
                
            if de_result.success:
                logging.warning(f"DE Iteration {i+1} finished. Best Loss: {de_result.fun:.4e}, Params: {de_result.x}")
            else:
                logging.warning(f"DE Iteration {i+1} did not converge successfully (Message: {de_result.message}).")
                
                
            print(f"DE Iteration {i+1} | {de_result.success}")
            print(f"DE resul: {de_result.x} | {de_result.fun}")
            print(f"DE: {de_result}")
            
            print("calls: ", self.total_calls)
            print("itts: ", self.total_itts)
                
    finally:
            pool.close()
            pool.join()
            
    if not global_search_candidates_params:
        logging.error("Differential Evolution failed to produce any candidates.")
        return None, None
        
    # Sort candidates from global search
    sorted_indices = np.argsort(global_search_candidates_loss)
    best_global_candidates_params = [global_search_candidates_params[i] for i in sorted_indices]
    best_global_candidates_loss = [global_search_candidates_loss[i] for i in sorted_indices]
        
    if not local_refinement:
        return best_global_candidates_params[0], best_global_candidates_loss[0]
        
    else:
        if use_logging:
            logging.info("Global search (DE) candidates (top 5 or all):")
            for k in range(min(5, len(best_global_candidates_params))):
                logging.info(f"  Rank {k+1}: Loss={best_global_candidates_loss[k]:.4e}, Params={best_global_candidates_params[k]}")
                    
        # Initialize overall best with the best from DE
        overall_best_params = np.copy(best_global_candidates_params[0])
        overall_best_loss = best_global_candidates_loss[0]
            
        ###* Local Search Refinement
        num_to_refine = min(top_k_candidates_for_local, len(best_global_candidates_params))
        if use_logging:
            logging.info(f"\nStarting local search refinement for the top {num_to_refine} DE candidate(s)...")
        
        for k_candidate_idx in range(num_to_refine):
            current_best_params_for_local = np.copy(best_global_candidates_params[k_candidate_idx])
            current_best_loss_for_local = best_global_candidates_loss[k_candidate_idx]
            
            if use_logging:
                logging.info(f"Refining DE Candidate {k_candidate_idx+1} (Loss: {current_best_loss_for_local:.4e})")
                    
            for attempt in range(local_attempts_per_candidate):
                # Perturb the starting point for local search
                param_ranges = bounds_array[:, 1] - bounds_array[:, 0]
                perturbation = np.random.uniform(-perturbation_factor * param_ranges, perturbation_factor * param_ranges)
                x0_local = current_best_params_for_local + perturbation
                x0_local = np.clip(x0_local, bounds_array[:, 0], bounds_array[:, 1]) # Ensure within bounds
                    
                try:
                    local_result = sp.optimize.minimize(
                            fun=self.objective_function,
                            x0=x0_local,
                            method="L-BFGS-B", # Good for bound-constrained problems
                            bounds=bounds,
                            options={'maxiter': 50, 'ftol': 1e-7, 'gtol': 1e-6} 
                    )
                    if use_logging:
                        logging.info(f"  Local Search (L-BFGS-B) from DE candidate {k_candidate_idx+1}, attempt {attempt+1}: "
                                    f"Loss={local_result.fun:.4e}, Success={local_result.success}, Message='{local_result.message}'")
                            
                    if local_result.success and local_result.fun < current_best_loss_for_local:
                        current_best_loss_for_local = local_result.fun
                        current_best_params_for_local = local_result.x
                            # Update overall best if this local search found a better solution
                        if current_best_loss_for_local < overall_best_loss:
                            overall_best_loss = current_best_loss_for_local
                            overall_best_params = np.copy(current_best_params_for_local)
                            logging.info(f"    New overall best found! Loss: {overall_best_loss:.4e}, Params: {overall_best_params}")
                                
                except Exception as e:
                        logging.error(f"  Local search attempt failed with exception: {e}", exc_info=True)
                
                
    return overall_best_params, overall_best_loss