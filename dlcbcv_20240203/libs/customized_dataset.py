print(f'customized_dataset.py is loaded.')

import numpy as np
import torch  # for building and training neural networks
from torch.utils.data import Dataset, DataLoader # for loading and managing datasets

class CrossSectionalTimeSeriesDataset(Dataset):
    def __init__(self, target_data, covariate_data, input_chunk_length=6, listofnp=True):
        """
        Initializes the CrossSectionalTimeSeriesDataset object.

        Args:
            target_data: List[np.ndarray] if cohort-ts or company-ts or company-cohort-ts; or np.ndarray if single time series.
            covariate_data (List[np.ndarray] or np.ndarray)
            input_chunk_length (int): Length of the input sequence.
            listofnp (bool): True of cross sectional time series data (list of numpy arrays); False if acquisition data (single numpy array).

        Note:
            In 'True' mode, target_data and covariate_data are expected to be lists of numpy arrays,
            where each array represents a group in the dataset.
            In 'False' mode, target_data and covariate_data are expected to be single numpy arrays.
            
        Output: List of dictionaries
        """
        self.input_chunk_length = input_chunk_length
        self.listofnp = listofnp
        
        # Initialize empty lists to store the sliced target input, covariate, and ground truth data
        self.targets, self.covariates, self.gts = [], [], []

        if listofnp:
            self._process_time_series_cross_sectional_data(target_data, covariate_data)
        else:
            self._process_time_series_data(target_data, covariate_data)

    def _process_time_series_cross_sectional_data(self, target_data, covariate_data):
        ## for loop over groups (cohorts or companies or company-cohorts)
        for target, covariate in zip(target_data, covariate_data):
            seq_len, _ = target.shape # Get the length of each group's sequence
            num_samples = seq_len - self.input_chunk_length 
            ## for loop over window-length samples (sliding by one data point) within each group
            for idx in range(num_samples):
                self._append_data(target, covariate, idx)

    def _process_time_series_data(self, target_data, covariate_data): 
        num_samples = len(target_data) - self.input_chunk_length # Get the length of time series sequence
        for idx in range(num_samples):
            self._append_data(target_data, covariate_data, idx)

    def _append_data(self, target, covariate, idx):
        self.targets.append(torch.from_numpy(target[idx: idx + self.input_chunk_length]))
        self.covariates.append(torch.from_numpy(covariate[idx: idx + self.input_chunk_length]))
        self.gts.append(torch.from_numpy(np.array(target[idx + self.input_chunk_length])))

    def __len__(self):
        return len(self.targets) # Returns the total number of samples in the dataset.

    def __getitem__(self, idx):
        # Retrieves the data sample at the given index.
        return {
            "target": self.targets[idx],
            "covariate": self.covariates[idx],
            "gt": self.gts[idx]
        }



def collate_fn(data):
    '''
    Take the list of dictionary samples and aggregate them into batches (output as a list format)
    '''
    aggregated = [] # Initialize an empty list to store the aggregated data
    aggregated.append(torch.from_numpy(np.stack([e["target"] for e in data]).astype(np.float32))) # (samples, input_chunk_length, 1)
    aggregated.append(torch.from_numpy(np.stack([e["covariate"] for e in data]).astype(np.float32))) # (samples, input_chunk_length, 1)
    aggregated.append(torch.from_numpy(np.stack([e["gt"] for e in data]).astype(np.float32))) # (samples, 1)
    
    return tuple(aggregated) # train_loader as a list. If not using this collate_fn, train_loader is dictionary



def value_dict_to_np(scaled_value_sequences, TASKS):
    combined_scaled_value_seq = [np.stack([scaled_value_sequences[col][i] for col in TASKS], axis=1) for i in range(len(scaled_value_sequences[TASKS[0]]))]

    return combined_scaled_value_seq