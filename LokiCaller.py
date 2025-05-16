
from pathlib import Path
import numpy as np
import logging
import os
import io

logging.basicConfig(level=logging.INFO)


class PhysicalSimulator():
    
    def __init__(self, eng, base_dir, input_file, output_file, x_names):
        self.eng = eng
        self.base_dir = Path(base_dir)
        self.input_path = self.base_dir / 'Input' / input_file
        self.output_path = self.base_dir / 'Output' / output_file
        
        self.x_names = x_names
        
        ### cache key->line index in the input file for faster writes
        self._key2line, self._key2name = self._index_input_file()
    
    
    def _index_input_file(self):
        
        lines = self.input_path.read_text().splitlines()
        mapping_num ={}
        mapping_name = {}
        for idx, line in enumerate(lines):
            if ":" in line:
                key_num = line.split(':', 1)[0].strip()
                mapping_num[key_num] = idx
                
                line_aux = line.split(':')
                key_name = line_aux[0]
                mapping_name[key_num] = key_name
        
        # print(f"Input file keys: {mapping.keys()}")
        return mapping_num, mapping_name
    
    
    def moodify_input_file(self, params):
        
        lines = self.input_path.read_text().splitlines()
        for key, val in params.items():
            
            idx = self._key2line.get(key)
            if idx is not None:
                key_name = self._key2name.get(key)
                lines[idx] = f"{key_name}: {val}"
            else:
                logging.warning(f"Key '{key}' not found in input file.")
            
        self.input_path.write_text('\n'.join(lines))
        logging.debug(f"Input file modified with params: {params}")
    
    
    def _run_matlab(self):
        
        buf = io.StringIO()
        try:
            self.eng.loki(self.input_path.name, nargout=0, stdout=buf, stderr=buf)
        except Exception as e:
            logging.error(f"MATLAB error: {e}")
            logging.debug(buf.getvalue())
            raise
    
    
    def _read_output(self, skiprows=5):
        
        parts = []
        for line in self.output_path.read_text().splitlines()[skiprows:]:
            toks = [tok for tok in line.split() if tok]
            parts.append(toks)
        
        max_len = max(len(p) for p in parts)
        full_rows = [p for p in parts if len(p) == max_len]
        
        arr = np.array(full_rows, dtype=object)
        names = arr[:, 0].tolist()
        values = arr[:, 1].astype(float)
        
        return values, names
    
    
    def run_simulation(self, x_samples, pbar=None):
        
        all_values = []
        
        for i, samples in enumerate(x_samples, start=1):
            params = dict(zip(self.x_names, samples))
            logging.debug(f"Running simulation {i}/{len(x_samples)} with params: {params}")
            
            self.moodify_input_file(params)
            self._run_matlab()
            values, names = self._read_output()
            
            all_values.append(values)
            
            if pbar is not None:
                pbar.update(1)
            
        data = np.vstack(all_values)
        return data, names