from typing import Dict, Callable, List, Set, Tuple, Any
import json
import re
import numpy as np
import sympy as sy
import logging

# logger = logging.getLogger(__name__)

class SimulatorParser():
    """
    Parses chemical reactions definitions from a JSON file, 
    constructs symbolic representations of the reactions system,
    and generates callable functions for the ODE system
    """

    def __init__(self, reactions_file: str, const_dict: Dict[str, float]):
        self.reactions_filename: str = reactions_file
        self.const_dict: Dict[str, float] = const_dict
    
    
    @staticmethod
    def parse_side(side_str: str) -> Tuple[Dict[str, int], str]:
        species: Dict[str, int] = {}
        gas: str = ''
        for term in side_str.split('+'):
            term = term.strip()
            m = re.match(r'(\d*)\s*([A-Za-z0-9_]+)', term)
            if not m:
                raise ValueError(f"Cannot parse term '{term}' in side string '{side_str}'")
            
            coeff_str = m.group(1)
            name = m.group(2)
            coeff = int(coeff_str) if coeff_str else 1
            name_split = name.split("_")
            if len(name_split) == 1: 
                gas = name  # This logic implies only one such "gas" term per side.
                            # Or, if it's not a gas but a species without site type.
                            
            elif len(name_split) == 2: # Assumed to be species_site, e.g., A_F
                species[name] = coeff
            else:
                logging.warning(f"Species name '{name}' in '{side_str}' has unexpected format (more than one '_'). Treating as simple species.")

        # The code sets 'gas = name' if len(name_split)==1.
        # This means it's both a species in the `species` dict AND identified as the 'gas_specie' for the reaction.
        # If multiple such species exist on one side, the LAST one parsed becomes the 'gas_specie' - should NOT happen
        return species, gas
    
    
    @staticmethod
    def load_reactions(reactions_filename: str) -> Tuple[List[Dict[str, Any]], List[str], List[str], List[str], List[List[str]]]:
        with open(reactions_filename, 'r') as f:
            raw_reactions = json.load(f)

        reactions_list: List[Dict[str, Any]] = []
        all_species_set: Set[str] = set()

        for r_idx, r_data in enumerate(raw_reactions):
            equation = r_data['equation']
            rate_label = r_data['rate']
            model_dict_params = r_data.get('model_dict', {}) 
            gamma_flag = r_data.get('gamma', False)
            reaction_id = r_data.get('id', r_idx + 1)
            
            if '<->' in equation:
                ###! reversible case is not considered   
                left_str, right_str = equation.split('<->')
                reversible = True
            elif '->' in equation:
                left_str, right_str = equation.split('->')
                reversible = False
            else:
                raise ValueError(f"Reaction equation '{equation}' must contain '->' or '<->'.")
            
            left_species, gas_left = SimulatorParser.parse_side(left_str)
            right_species, gas_right = SimulatorParser.parse_side(right_str)
            
            all_species_set.update(left_species.keys())
            all_species_set.update(right_species.keys())
            
            reactions_list.append({
                'id': reaction_id, 
                'left': left_species,
                'right': right_species,
                'rate': rate_label,      # This rate constant might be used for forward
                'gas_specie': gas_left,  # Gas relevant to reactants
                'model_dict': model_dict_params,
                'gamma': gamma_flag
            })
            
        rates_list = ["r_"+str(r["id"]) for r in reactions_list]
        
        species_list = sorted(list(all_species_set))
        groups_species_split = [ele.split('_') for ele in species_list]
        group_species = list(set([ele[1] for ele in groups_species_split if len(ele) == 2]))
        
        return reactions_list, species_list, rates_list, group_species, groups_species_split
    
    
    @staticmethod
    def make_symbols(name_list: List[str]) -> Tuple[List[sy.Symbol], Dict[str, sy.Symbol]]:
        if not name_list:
            return [], {}
        syms_tuple = sy.symbols(' '.join(name_list))
        syms = list(syms_tuple) if isinstance(syms_tuple, (tuple, list)) else [syms_tuple]
        return syms, dict(zip(name_list, syms))
    
    
    def _define_conservation_laws(self, species_map: Dict[str, sy.Symbol], group_species: List[str],
                                groups_species_split: List[List[str]]) -> Tuple[List[Tuple[sy.Symbol, Any]], Dict[str, List[str]], List[str]]:
        
        conservation_laws_dict: Dict[str, List[str]] = {}
        for group in group_species:
            vec_species: List[str] = []
            for specie_parts in groups_species_split:
                if len(specie_parts) == 2 and specie_parts[1] == group and specie_parts[0] != "V":
                    vec_species.append("_".join(specie_parts))
                    
            vacancy_species_name = "V_" + group
            if vacancy_species_name not in species_map:
                raise ValueError(f"The vacancy state V_{group} is not in species_list. "
                                "Conservation laws require it or a different formulation.")
                
            if not vec_species and vacancy_species_name in species_map: # Only vacancy exists for this group
                conservation_laws_dict[vacancy_species_name] = []      # V_group = 1 - 0
            elif vec_species:
                conservation_laws_dict[vacancy_species_name] = vec_species
        
        old_expressions: List[sy.Symbol] = []
        new_expressions: List[Any] = []

        for old_specie_name, new_specie_names_list in conservation_laws_dict.items():
            old_expressions.append(species_map[old_specie_name])
            new_term_sum = sy.S(0)
            for ele_name in new_specie_names_list:
                new_term_sum += species_map[ele_name]
            new_expressions.append(1.0 - new_term_sum) # site fractions sum to 1
            
        pairs_laws = list(zip(old_expressions, new_expressions))
        
        # Determine effective species list (those not eliminated by conservation laws)
        eliminated_species = {str(law[0]) for law in pairs_laws} # species that are 'old_expressions'
        all_species_in_map = list(species_map.keys())
        eff_species_list = [s for s in all_species_in_map if s not in eliminated_species]
        
        return pairs_laws, conservation_laws_dict, eff_species_list
    
    
    def _build_ode_system(self, eff_species_list: List[str], reactions_list: List[Dict[str, Any]],
                        species_map: Dict[str, sy.Symbol], rates_map: Dict[str, sy.Symbol],
                        pairs_laws: List[Tuple[sy.Symbol, Any]]) -> List[sy.Expr]:
        
        equations_sym: List[sy.Expr] = []
        ratio_S_F = self.const_dict.get("S0", 1.0) / self.const_dict.get("F0", 1.0) # Handle missing keys with default
        if "S0" not in self.const_dict or "F0" not in self.const_dict:
            logging.warning("'S0' or 'F0' not in const_dict. ratio_S_F might be incorrect.")


        for specie_name in eff_species_list:
            # Determine if the current species is an "F-site" species for S_F ratio application
            is_F_site_species = False
            if '_' in specie_name:
                site_type = specie_name.split('_')[1]
                if site_type == "F":
                    is_F_site_species = True
                    
            sum_of_terms_for_species_rate = sy.S(0) # Sympy zero
            
            for reaction in reactions_list:
                rate_symbol = rates_map["r_" + str(reaction["id"])]
                term_contribution = sy.S(0)
                
                # Check if specie is a reactant
                if specie_name in reaction["left"]:
                    stoich_coeff_reactant = reaction["left"][specie_name]
                    product_of_reactant_terms = sy.S(1) # Sympy one
                    
                    has_S_site_reactant = False
                    is_bimolecular_self_reaction = False # e.g. 2A -> B, for A, d[A]/dt = -2*k*A^2
                    
                    for reactant_specie, nu in reaction["left"].items():
                        product_of_reactant_terms *= species_map[reactant_specie] ** nu
                        if reactant_specie == specie_name and nu == 2:
                            is_bimolecular_self_reaction = True # Special handling for d[A]/dt if A+A->
                        if '_' in reactant_specie and reactant_specie.split('_')[1] == "S":
                            has_S_site_reactant = True
                    
                    # Base term for reactant consumption
                    term_contribution = -stoich_coeff_reactant * rate_symbol * product_of_reactant_terms
                    
                elif specie_name in reaction["right"]:
                    stoich_coeff_product = reaction["right"][specie_name]
                    product_of_reactant_terms = sy.S(1) 
                    
                    has_S_site_reactant = False 
                    for reactant_specie, nu in reaction["left"].items(): # Rate based on reactants
                        product_of_reactant_terms *= species_map[reactant_specie] ** nu
                        if '_' in reactant_specie and reactant_specie.split('_')[1] == "S":
                            has_S_site_reactant = True
                    
                    term_contribution = +stoich_coeff_product * rate_symbol * product_of_reactant_terms
                    
                if term_contribution != 0: # If the species participates
                    if is_F_site_species and has_S_site_reactant: # Apply S/F ratio
                        term_contribution *= ratio_S_F
                
                sum_of_terms_for_species_rate += term_contribution
                
            equations_sym.append(sum_of_terms_for_species_rate)
            
        # Substitute conservation laws
        equations_sym_new = [eq.subs(pairs_laws) for eq in equations_sym]
        return equations_sym_new
    
    
    def create_physical_model(self, flag_debugging: bool = True) -> Dict[str, Any]:
        ##* Creates the physical model: symbolic ODEs and a lambdified function.

        logging.info("Creating physical model...")
        
        reactions_list, species_list, rates_list, \
        group_species, groups_species_split = self.load_reactions(self.reactions_filename)
        
        logging.debug(f"All species: {species_list}")
        logging.debug(f"All rates: {rates_list}")
        logging.debug(f"Group species: {group_species}")
        
        species_vars_list, species_map = self.make_symbols(species_list)
        rates_vars_list, rates_map = self.make_symbols(rates_list)
        
        if not species_vars_list: # No species found
            logging.error("No species were parsed. Cannot create model.")
            return { # Return a structure indicating failure or empty model
                'model': lambda X, r: np.array([]),
                'species_model': [],
                'rates_model': [],
                'reactions_list': reactions_list,
                'error': "No species parsed"
            }
        
        pairs_laws, conservation_laws_details, eff_species_list = self._define_conservation_laws(species_map, group_species, groups_species_split)
        logging.info(f"Effective species for ODE model: {eff_species_list}")
        logging.debug(f"Conservation laws applied: {pairs_laws}")
        
        equations_sym_new = self._build_ode_system(eff_species_list, reactions_list, species_map, rates_map, pairs_laws)
        
        if flag_debugging:
            logging.info("System of ODEs (d(species)/dt):")
            for i, specie_name in enumerate(eff_species_list):
                print(f"  d[{specie_name}]/dt = {equations_sym_new[i]}")
                logging.info(f"  d[{specie_name}]/dt = {equations_sym_new[i]}")
                
        # Ensure eff_species_vars are in the same order as eff_species_list for lambdify
        eff_species_vars_sympy = [species_map[name] for name in eff_species_list]

        # Lambdify the system of equations
        if not eff_species_vars_sympy and not equations_sym_new:
            # This case means no effective species, perhaps everything was conserved or no reactions.
            logging.warning("No effective species or equations to lambdify. Model will be empty.")
            model_fun = lambda X, r_vec: np.array([]) # Empty model
        elif not equations_sym_new and eff_species_vars_sympy : # Species exist, but no equations
            logging.warning("Effective species exist, but no dynamic equations. Model will return zeros matching species length.")
            model_fun = lambda X_state, r_param: np.zeros(len(X_state))
        else:
            try:
                model_fun = sy.lambdify((eff_species_vars_sympy, rates_vars_list), equations_sym_new, modules='numpy')
            except Exception as e:
                logging.error(f"Error during lambdify: {e}. Model function might not work.")
                model_fun = lambda X, r_vec: (_ for _ in ()).throw(RuntimeError(f"Lambdify failed: {e}")) # Raises error on call
                
        output_dict: Dict[str, Any] = {} 
        output_dict['model'] = model_fun
        output_dict['species_model'] = eff_species_list # These are the species to be solved for
        output_dict['rates_model'] = rates_list         # These are the rate constants names 'r_id'
        output_dict['reactions_list'] = reactions_list  # Full reaction details
        
        logging.info("Physical model creation complete.")
        return output_dict