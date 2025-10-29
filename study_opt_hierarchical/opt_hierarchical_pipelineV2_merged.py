import sys, os
PATH = os.path.dirname(os.path.abspath(os.curdir))
if PATH not in sys.path:
    sys.path.insert(0, PATH)

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from pathos.multiprocessing import ProcessPool
import src.Simulator as sim_system
import src.Optimizer as opt
import scipy as sp
import numpy as np
import torch
import tqdm
import inspect
import h5py


###* Create Simulator object
reactions_file = "../reactions/reactionsCompleteV2.json"

const_dict = {
        "F0": 1.5e15,           # cm^-2
        "S0": 3e13,             # cm^-2
        
        "R": 0.00831442,        # kJ/mol*K
        "kBoltz": 1.380649e-23, # J/K
}

initial_state_dict = {'O_F': 0.1, 'O2_F':0.1 ,'O_S': 0.1, 'Vdb_S':0.1, 
                    'Odb_S': 0.1, 'CO_F': 0.1, 'CO2_F':0.1, 'CO_S': 0.1, 
                    'COdb_S': 0.0}

###* Functions for the data transformation
def compute_flux(const_dict, exp_dict, specie, molar_mass):
    den = exp_dict.get(specie, 0.0)
    v_th = np.sqrt((8.0 * const_dict['R'] * 1000 * exp_dict['Tnw'])/(molar_mass * np.pi))
    flux = 0.25 * v_th * den * 100
    return flux


def compute_remaining_flux(const_dict, exp_dict, molar_mass): 
    den = exp_dict['N'] - exp_dict['O'] - exp_dict['CO']
    v_th = np.sqrt((8.0 * const_dict['R'] * 1000 * exp_dict['Tnw'])/(molar_mass * np.pi))
    flux = 0.25 * v_th * den * 100
    return flux

####? EavgMB data extracted from the Booth et al. 2019 paper
p_data_exp = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.5]
EavgMB_data = [1.04, 0.91, 0.87, 0.83, 0.77, 0.5, 0.001]
interpolator = sp.interpolate.interp1d(p_data_exp, EavgMB_data, kind='linear', fill_value=0.001, bounds_error=False)


transformations_exp = {
    'Tw':       lambda const_dict, exp_dict: exp_dict['Tw'] + 273.15,
    'fluxO' :   lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict,'O', 0.016),
    'fluxO2' :  lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict,'O2', 0.032),
    'fluxO3' :  lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict,'O3', 0.048),
    'fluxC':    lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'C', 0.012),
    'fluxCO':   lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'CO', 0.028),
    'fluxCO2':  lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'CO2', 0.048),
    'EavgMB':   lambda const_dict, exp_dict: interpolator(exp_dict['pressure']).item(),
    'Ion':      lambda const_dict, exp_dict: 1e14 * exp_dict["current"]
}

output_folder_path = "../Buffer_Data"
exp_data_file = "Experimental_data_CO_O_merged.hdf5"
exp_file = os.path.join(output_folder_path, exp_data_file)

sim = sim_system.Simulator(reactions_file, const_dict, exp_file, initial_state_dict, transformations_exp=transformations_exp)


####! bounds observables
lower_bounds = np.array([1e-8, 1e-8, 0.0, \
                    1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                    1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, \
                    1.0, 2000
                    ])

upper_bounds = np.array([5e-1, 1e-2, 30.0, \
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                    5.0, 5000
                    ])


##! define the default parameters
params_default_init = np.array([
                0.02634, 7.67e-4, 10.75, \
                1.0, 1.0, 1e-2, 1e-1, 1e-1, 1e-2, 1e-1, 1e-1, 1e-1, 1e-1, \
                1e-2, 1.0, 1.0, 1.0, 1e-1, 1e-1, 1.0, 1.0, 1.0, 1e-1, 1e-1, 1e-2, 1e-1, 1e-1, \
                3.4, 3000
                ])


params_default_norm = (params_default_init - lower_bounds) * np.reciprocal(upper_bounds - lower_bounds)

###! optimization function
def func_optimization(params_input, flag='numpy'):
    
    ##! normalize variables
    params = [0] * len(params_input)
    for idx, param in enumerate(params_input):
        params[idx] = lower_bounds[idx] + (upper_bounds[idx] - lower_bounds[idx]) * param
    
    A_d, B_d, E_d = params[0:3]
    SF_30, SF_31, SF_32, SF_33, SF_34, SF_35, SF_36, SF_37, SF_38, SF_39 = params[3:13]
    SF_49, SF_50, SF_51, SF_52, SF_53, SF_54, SF_55, SF_56, SF_57, SF_58, SF_59, SF_60, SF_61, SF_62 = params[13:27]
    Emin, Ealpha = params[27:]
    
    if flag=='numpy':
        nu_d_mod = lambda T: 1e15 * (A_d + B_d * np.exp(E_d/(const_dict['R'] * T)))
    elif flag=='torch':
        nu_d_mod = lambda T: 1e15 * (A_d + B_d * torch.exp(E_d/(const_dict['R'] * T)))
    else:
        raise ValueError(f"{flag} does not exist")
    
    dict_mod_vec = [
    {"id": 2, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    {"id": 10, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    {"id": 16, "rate": None, "model_dict": {"Emin": Emin}},
    {"id": 18, "rate": None, "model_dict": {"Emin": Emin}},
    
    {"id": 31, "rate": None, "model_dict": {"SF": SF_31, "nu_d": nu_d_mod}},
    
    {"id": 30, "rate": None, "model_dict": {"SF": SF_30}},
    {"id": 32, "rate": None, "model_dict": {"SF": SF_32}},
    {"id": 33, "rate": None, "model_dict": {"SF": SF_33}},
    {"id": 34, "rate": None, "model_dict": {"SF": SF_34}},
    
    {"id": 35, "rate": None, "model_dict": {"SF": SF_35}},
    {"id": 36, "rate": None, "model_dict": {"SF": SF_36}},
    {"id": 37, "rate": None, "model_dict": {"SF": SF_37}},
    {"id": 38, "rate": None, "model_dict": {"SF": SF_38}},
    {"id": 39, "rate": None, "model_dict": {"SF": SF_39}},
    
    {"id": 44, "rate": None, "model_dict": {"Emin": Emin}},
    
    {"id": 49, "rate": None, "model_dict": {"SF": SF_49}},
    {"id": 50, "rate": None, "model_dict": {"SF": SF_50, "Ealpha": Ealpha}},
    {"id": 51, "rate": None, "model_dict": {"SF": SF_51, "Ealpha": Ealpha}},
    {"id": 52, "rate": None, "model_dict": {"SF": SF_52, "Ealpha": Ealpha}},
    {"id": 53, "rate": None, "model_dict": {"SF": SF_53, "Ealpha": Ealpha}},
    {"id": 54, "rate": None, "model_dict": {"SF": SF_54, "Ealpha": Ealpha}},
    {"id": 55, "rate": None, "model_dict": {"SF": SF_55, "Ealpha": Ealpha}},
    {"id": 56, "rate": None, "model_dict": {"SF": SF_56, "Ealpha": Ealpha}},
    {"id": 57, "rate": None, "model_dict": {"SF": SF_57, "Ealpha": Ealpha}},
    {"id": 58, "rate": None, "model_dict": {"SF": SF_58, "Ealpha": Ealpha}},
    {"id": 59, "rate": None, "model_dict": {"SF": SF_59, "Ealpha": Ealpha}},
    {"id": 60, "rate": None, "model_dict": {"SF": SF_60}},
    {"id": 61, "rate": None, "model_dict": {"SF": SF_61}},
    {"id": 62, "rate": None, "model_dict": {"SF": SF_62}}
    ]
    
    return dict_mod_vec


def loss_function(exp, teo, flag='numpy'):
    
    func = ((teo-exp)**2)/(exp**2)
    if flag == 'numpy':
        return np.mean(func)
    elif flag == 'torch':
        return torch.mean(func)
    else:
        raise ValueError(f"{flag} does not exist")


optimizer = opt.Optimizer(sim, 
                        lambda params: func_optimization(params, 'numpy'), 
                        lambda exp, teo: loss_function(exp, teo, 'numpy')
                        )


###* global variables to keep track of the evolution measures

iter_calls = 0
best_loss = float('inf')
history = {'iters':[], 'best_loss':[], 'type':[]}


###* compute the gradients numerically, we assume we dont have an AD module installed

### order 1 for derivatives

def grads_wise_numeric(params, opt_object, hstep = 1e-5):
    
    global iter_calls
    params_arr = np.array(params)
    _, residual, _ ,_, _, _ = opt_object.objective_function_diff_full(params_arr)
    
    iter_calls += 1
    
    residuals = []
    for i in range(len(params)):
        step_vec = np.zeros_like(params_arr)
        step_vec[i] = hstep
        _, residual_plus, _ ,_, _, _ = opt_object.objective_function_diff_full(params_arr + step_vec)
        grad = (residual_plus-residual)/hstep
        residuals.append(grad)
        
        iter_calls += 1
        
        print("i grad:: ", i, "-", len(params), "grad: ", np.mean(np.abs(grad)))
    
    return np.array(residuals).T


def create_subspaces_num(params, percent_info, opt_object, reg = 1e-6, tau=1e-4):
    
    grad_errors = grads_wise_numeric(params, opt_object)
    G = grad_errors                                   
    F = G.T @ G                                                 
    F_reg = F + reg * np.eye(F.shape[0], dtype=F.dtype)
    
    eigvals, eigvecs = np.linalg.eigh(F_reg)                    

    idx = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]
    
    print("eigen_vals_sorted: ", eigvals_sorted)
    
    total_mass = eigvals_sorted.sum()
    cumulative  = np.cumsum(eigvals_sorted)
    ks = int(np.sum(cumulative < percent_info * total_mass)) + 1
    Vs = eigvecs_sorted[:, :ks]
    
    if ks < len(eigvals_sorted):
        ratios = eigvals_sorted[ks:] / eigvals_sorted[ks]
        kl = int(np.count_nonzero(ratios >= tau))
    else:
        kl = 0
    Vl = eigvecs_sorted[:, ks:ks+kl]

    return {
        'phi0': params,
        'dims': len(eigvals_sorted),
        'V_dims': (ks, kl),
        'Vs': Vs,
        'Vl': Vl,
        'eigvals_sorted': eigvals_sorted,
        'eigvecs_sorted': eigvecs_sorted
    }



def Phi_loss(space_string, vars_opt, opt_object, diff_objec, config, string="loss: "):
    
    if not space_string in ['Vs', 'Vl']:
        raise ValueError(f"{space_string} must be Vs or Vl")
    
    params = np.abs(config['phi0'] + config[space_string] @ vars_opt)
    loss_val, frac_solutions_arr, rates_arr, _, gammas_predicted_arr = opt_object.objective_function_diff(params)
    

    global iter_calls
    global best_loss
    global history
    
    iter_calls += 1
    
    best_loss = loss_val if loss_val < best_loss else best_loss
    
    if iter_calls % 15:
        history['iters'].append(iter_calls)
        history['best_loss'].append(best_loss)
        if space_string == 'Vs':
            history['type'].append("stiff")
        else:
            history['type'].append("sloppy")
    
    if iter_calls % 20:
        print(string, loss_val, "vars: ", vars_opt ,"iter_calls: ", iter_calls)
        
    return loss_val


class TooManyEvals(Exception):
    def __init__(self, message, value=None):
        super().__init__(message)
        self.value = value



def make_limited_fun(fun, max_calls):
    calls = {'n': 0}
    last_value = None
    def wrapped(*args, **kwargs):
        nonlocal last_value
        calls['n'] += 1
        if calls['n'] > max_calls:
            raise TooManyEvals(f"Exceeded {max_calls} function evaluations", args[1] if len(args) > 1 else None)
        return fun(*args, **kwargs)
    return wrapped


####* Algorithm: the same as presented in the report

config_dict_record = []
delta_phi_record = []
delta_s_record = []

###! hyperparameters
filename = "sloppy_num_1e-3_merged3.h5"
percent_info = 0.90
tau = 1e-4
# tolerace = 1e-3
tolerace = 1e-4
max_fun_calls_sloppy = 130

nb_iter = 0
nb_max_iter = 50

####* Initialization

config_dict = create_subspaces_num(params_default_norm, percent_info, optimizer, tau=tau)
phi_prev = config_dict['phi0']


####* Loop
while nb_iter < nb_max_iter:
    
    #### optimization along the stiff space
    init_vec = np.array([0.0 for _ in range(config_dict['V_dims'][0])])
    res_stiff = sp.optimize.minimize(
                lambda params: Phi_loss('Vs', params, optimizer, diff, config_dict, string="loss stiff: "),
                x0=init_vec,
                method='Powell',
                options={
                'xtol': 1e-4,     # tolerance on parameter changes
                'ftol': 1e-4,     # tolerance on function‐value changes
                'maxiter': 200, 
                'maxfev': 150   
                }
    )
    
    config_dict['phi0'] = np.abs(config_dict['phi0'] + config_dict['Vs'] @ res_stiff.x)
    
    #### optimization alogn the sloppy space
    init_vec = np.array([0.0 for _ in range(config_dict['V_dims'][1])])
    wrapped_Phi_loss = make_limited_fun(Phi_loss, max_fun_calls_sloppy)
    try:
        res_sloppy = sp.optimize.minimize(
                lambda params: wrapped_Phi_loss('Vl', params, optimizer, diff, config_dict, string="loss sloppy: "),
                x0 = init_vec,
                method='Nelder-Mead',
                tol=5e-5,
                options={
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'disp': True  # (optional) shows progress
                        }
                )
        psi = res_sloppy.x
                
    except TooManyEvals as e:
        psi = e.value
        
    
    config_dict['phi0'] = np.abs(config_dict['phi0'] + config_dict['Vl'] @ psi)
    
    ### create the new subspace optimization
    Vs_old = config_dict['Vs']
    config_dict_record.append(config_dict)
    
    config_dict = create_subspaces_num(config_dict['phi0'], percent_info, optimizer, tau=tau)
    
    ### check stopping criteria
    M = config_dict['Vs'].T @ Vs_old
    sigmas = np.linalg.svd(M, compute_uv=False)
    delta_s = max(1-sigmas)
    
    delta_phi = np.sqrt(np.sum((config_dict['phi0'] - phi_prev)**2))
    
    delta_s_record.append(delta_s)
    delta_phi_record.append(delta_phi)
        
    print("**"*50)
    print("DELTA PHI: ", delta_phi, "DELTA S: ", delta_s)
    
    if delta_s < tolerace:
        break
        
    nb_iter += 1
    phi_prev = config_dict['phi0']
    


###! record results
with h5py.File(filename, "w") as f:
    
    f.create_dataset("best_loss", data=history['best_loss'])
    f.create_dataset("iters", data=history['iters'])
    f.create_dataset("type", data=history['type'])
    
    f.create_dataset("delta_s", data=delta_s_record)
    f.create_dataset("delta_phi", data=delta_phi_record)
    f.create_dataset("best_params", data=config_dict['phi0'])
