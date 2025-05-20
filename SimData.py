import numpy as np
import pandas as pd
import os 
import h5py
import logging 
from typing import Dict, List, Optional, Union


# logger = logging.getLogger(__name__)


class DataLoader():
    """
    Handles loading experimental data from files (like Excel),
    processes it according to a schema, and stores it in an HDF5 buffer file
    """
    
    def __init__(self, schema: Dict[str, str], folder_path: Optional[str]=None, 
                files_path: Optional[List[str]]=None, output_file: str="Experimental_data.hdf5",
                output_folder: str="Buffer_Data", extension: str=".xlsx"):
        
        self.schema: Dict[str, str] = schema
        self.output_folder: str = output_folder
        self.extension: str = extension
        
        if files_path is not None:
            self.data_files_path: List[str] = files_path
        elif folder_path is not None:
            if not os.path.isdir(folder_path):
                raise ValueError(f"Folder path '{folder_path}' does not exist.")
            file_name_list = os.listdir(folder_path)
            self.data_files_path = [os.path.join(folder_path, ele) for ele in file_name_list if ele.endswith(self.extension)]
            if not self.data_files_path:
                logging.warning(f"No files with extension '{self.extension}' found in folder '{folder_path}'.")
        else:
            raise ValueError("Please provide either 'files_path' or 'folder_path'.")
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            logging.info(f"Created output folder: {self.output_folder}")
        
        self.output_file: str = os.path.join(self.output_folder, output_file)
        
        
    def create_hdf5_buffer(self) -> None:
        logging.info(f"Creating HDF5 buffer at: {self.output_file}")
        data_tensor: Dict[str, List[np.ndarray]] = {name: [] for name in self.schema.values()}

        for file_name in self.data_files_path:
            logging.debug(f"Reading data from file: {file_name}")
            try:
                df = pd.read_excel(file_name)
            except FileNotFoundError:
                logging.error(f"File not found: {file_name}. Skipping.")
                continue
            except Exception as e:
                logging.error(f"Error reading Excel file {file_name}: {e}. Skipping.")
                continue

            for name_exp, name_sim in self.schema.items():
                if name_exp in df.columns:
                    vec_aux = df[name_exp].dropna().to_numpy()
                    if vec_aux.size > 0:
                        data_tensor[name_sim].append(vec_aux)
                else:
                    logging.warning(f"Column '{name_exp}' not found in file '{file_name}'.")
                    
        n_size: Optional[int] = None
        processed_data_tensor: Dict[str, np.ndarray] = {}
        
        for name_sim_val in self.schema.values(): 
            if data_tensor[name_sim_val]:
                concatenated_array = np.concatenate(data_tensor[name_sim_val])
                processed_data_tensor[name_sim_val] = concatenated_array
                if n_size is None:
                    n_size = len(concatenated_array)
                elif len(concatenated_array) != n_size:
                    logging.warning(f"Data length mismatch for '{name_sim_val}'. Expected {n_size}, got {len(concatenated_array)}. Padding/truncation may occur.")
            else:
                processed_data_tensor[name_sim_val] = np.array([])
        
        all_lengths = [len(arr) for arr in processed_data_tensor.values() if arr.size > 0]
        if all_lengths:
            n_size = max(all_lengths) # Use max length of any concatenated array
        else:
            n_size = 0 # No data loaded
            
        final_data_tensor: Dict[str, np.ndarray] = {}
        for name, data_array in processed_data_tensor.items():
            if data_array.size > 0 :
                if len(data_array) < n_size:
                    logging.info(f"Padding data for '{name}'. Original length: {len(data_array)}, Target length: {n_size}")
                    padded_array = np.zeros(n_size)
                    padded_array[:len(data_array)] = data_array
                    final_data_tensor[name] = padded_array
                elif len(data_array) > n_size:
                    logging.warning(f"Truncating data for '{name}'. Original length: {len(data_array)}, Target length: {n_size}")
                    final_data_tensor[name] = data_array[:n_size]
                else:
                    final_data_tensor[name] = data_array
            else:
                logging.info(f"No data for '{name}'. Creating empty/zero array of length {n_size}.")
                final_data_tensor[name] = np.zeros(n_size)


        with h5py.File(self.output_file, "w") as file:
            dt = h5py.string_dtype(encoding='utf-8')
            file.create_dataset("File_Data", data=np.array(self.data_files_path, dtype=dt))
            for name, data in final_data_tensor.items():
                file.create_dataset(name, data=data)
        logging.info("HDF5 buffer creation complete.")
        
    
    
    def get_summary(self) -> None:
        if not os.path.exists(self.output_file):
            logging.error(f"HDF5 file not found: {self.output_file}. Cannot generate summary.")
            return
        
        logging.info("Summary of HDF5 Buffer:")
        with h5py.File(self.output_file, "r") as file:
            try:
                file_data_ds = file['File_Data']
                if h5py.check_string_dtype(file_data_ds.dtype):
                    paths = file_data_ds.asstr()[:]
                else:
                    paths = [p.decode('utf-8', 'ignore') for p in file_data_ds[:]]
                logging.info(f"  File Data Sources: {paths}")
            except KeyError:
                logging.warning("  'File_Data' dataset not found in HDF5 file.")

            keys = list(file.keys())
            for key in keys:
                if key == 'File_Data':
                    continue
                try:
                    dataset = file[key]
                    logging.info(f"  Column '{key}': shape {dataset.shape}, dtype {dataset.dtype}")
                except KeyError:
                    logging.warning(f"  Could not access dataset for key '{key}'.")


    def load_data(self, force_update: bool = False) -> None:
        if not os.path.exists(self.output_file) or force_update:
            logging.info("HDF5 buffer needs to be created or updated.")
            self.create_hdf5_buffer()
        else:
            logging.info(f"Using existing HDF5 buffer: {self.output_file}")

        self.get_summary()


