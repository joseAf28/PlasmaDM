import numpy as np
import scipy as sp


def adsorption(const_dict, exp_dict, model_dict):
    F0, S0, R = const_dict["F0"], const_dict["S0"], const_dict["R"]
    Tnw = exp_dict['Tnw']
    SF, E = model_dict['SF'], model_dict['E']
    gas_specie = model_dict['gas_specie']
    
    surface = F0 + S0
    flux = exp_dict["flux" + gas_specie]
    return SF * flux / surface * np.exp(-E / (R * Tnw))


def desorption(const_dict, exp_dict, model_dict):
    R = const_dict["R"]
    Tw = exp_dict['Tw']
    SF, E, nu_d = model_dict['SF'], model_dict['E'], model_dict['nu_d']
    
    return SF * nu_d * np.exp(-E / (R * Tw))


def diffusion(const_dict, exp_dict, model_dict):
    R = const_dict["R"]
    Tw = exp_dict['Tw']
    SF, E, nu_D, factor = model_dict['SF'], model_dict['E'], model_dict['nu_D'], model_dict['factor']
    
    return SF * factor * nu_D * np.exp(-E / (R * Tw))


def recomb_ER(const_dict, exp_dict, model_dict):
    F0, S0, R = const_dict["F0"], const_dict["S0"], const_dict["R"]
    Tnw = exp_dict['Tnw']
    SF, E = model_dict['SF'], model_dict['E']
    gas_specie = model_dict['gas_specie']
    
    surface = F0 + S0
    flux = exp_dict["flux" + gas_specie]
    return SF * flux / surface * np.exp(-E / (R * Tnw))


def recomb_LH(const_dict, exp_dict, model_dict):
    R = const_dict["R"]
    Tw = exp_dict['Tw']
    SF, E, nu_D, factor = model_dict['SF'], model_dict['E'], model_dict['nu_D'], model_dict['factor']
    
    return SF * factor * nu_D * np.exp(-E / (R * Tw))


####* include metastable reactions
###! add the integrals in the exp_dict, eventuallly to increase efficiency
###! add the CO, CO2 and O2 densities later 
###! have the optimization later

def MB_func(E, TavgMB, kBoltz):
    return (2.0/ np.sqrt(np.pi) * ((1.0 / (kBoltz * TavgMB)) ** 1.5) * np.exp(-E / (kBoltz * TavgMB)) * (E**0.5))
    
    
def create_meta(const_dict, exp_dict, model_dict):
    
    SF = model_dict['SF']
    if SF == 0.0:
        return 0.0
    
    E, Emin = model_dict['E'], model_dict['Emin']
    
    F0, S0, R, kBoltz = const_dict["F0"], const_dict["S0"], const_dict["R"], const_dict["kBoltz"]
    Tnw, Tw = exp_dict['Tnw'], exp_dict['Tw']
    EavgMB = exp_dict['EavgMB']
    
    E_di, E_de = model_dict['E_di'], model_dict['E_de']
    gas_specie = model_dict['gas_specie']
    
    TavgMB = EavgMB * 1.60218e-19 / kBoltz
    Emin = Emin * 1.60218e-19
    IntMB = sp.integrate.quad(MB_func, Emin, 40*Emin, args=(TavgMB, kBoltz))[0]
    
    surface = F0 + S0
    flux = exp_dict["Ion"]
    
    ratio_temp = (np.exp(-E_di / (R * Tw)) + np.exp(-E_de / (R * Tw))) \
        / (np.exp(-E_di / (R * 323.15)) + np.exp(-E_de / (R * 323.15)))
    
    return SF * flux / surface * np.exp(-E / (R * Tnw)) * IntMB * ratio_temp


def destroy_meta(const_dict, exp_dict, model_dict):
    
    SF = model_dict['SF']
    if SF == 0.0:
        return 0.0
    
    F0, S0, R, kBoltz = const_dict["F0"], const_dict["S0"], const_dict["R"], const_dict["kBoltz"]
    Tnw = exp_dict['Tnw']
    E, Ealpha = model_dict['E'], model_dict['Ealpha']
    gas_specie = model_dict['gas_specie']
    
    surface = F0 + S0
    flux = exp_dict["flux" + gas_specie]
    Ealpha = Ealpha * kBoltz
    Intalpha = sp.integrate.quad(MB_func, Ealpha, 40*Ealpha, args=(Tnw, kBoltz))[0]
    
    return SF * flux / surface * np.exp(-E / (R * Tnw)) * Intalpha


def recomb_ER_meta(const_dict, exp_dict, model_dict):
    
    SF = model_dict['SF']
    if SF == 0.0:
        return 0.0
    
    F0, S0, R, kBoltz = const_dict["F0"], const_dict["S0"], const_dict["R"], const_dict["kBoltz"]
    Tnw = exp_dict['Tnw']
    E, Ealpha = model_dict['E'], model_dict['Ealpha']
    gas_specie = model_dict['gas_specie']
    
    surface = F0 + S0
    flux = exp_dict["flux" + gas_specie]
    Ealpha = Ealpha * kBoltz
    Intalpha = sp.integrate.quad(MB_func, Ealpha, 40*Ealpha, args=(Tnw, kBoltz))[0]
    
    return SF * flux / surface * np.exp(-E / (R * Tnw)) * (1.0 - Intalpha)




