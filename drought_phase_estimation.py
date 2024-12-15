"""
  Return indices of of drought onset, drought, termination based on soil moisture and precipitation
  (defined as Kim et al. (2024)), and the duration of droughts. 
  Input: a soil moisture anomaly (or other standardized drought indices) and/or precipitation in xarray or np.array
"""

import numpy as np
import xarray as xr

"""
  Initial input: Soil and PR time series. 
"""


def select_based_on_threshold(seq, x):
    """
    Select negative X clusters based on a threshold.
    
    Parameters:
        seq (np.ndarray): Sequence of 0s and 1s.
        x (xarray.DataArray): Input data array.
    
    Returns:
        tuple: Sorted clusters and their indices.
    """
    x_std = x.std().values
    x_mean = x.values

    clusters, index_cluster = [], []
    tmp, ind_tmp = [], []

    for i, k in enumerate(seq):
        # Group consecutive drought events
        if k == 1:
            tmp.append(x_mean[i])
            ind_tmp.append(i)
        elif tmp:  # Only append if `tmp` is non-empty
            clusters.append(tmp)
            index_cluster.append(ind_tmp)
            tmp, ind_tmp = [], []

    # Apply criteria: clusters must have at least one value <= -std
    cluster_sort, index_sort = [], []
    for i, c in enumerate(clusters):
        if np.any(np.array(c) <= -x_std):
            cluster_sort.append(c)
            index_sort.append(index_cluster[i])

    return cluster_sort, index_sort


class DroughtSoil:
    """
    Class to detect droughts in soil data based on thresholds.
    """
    
    def __init__(self, object_in, step_pos='yes'):
        """
        Initialize with input data.
        
        Parameters:
            object_in (xarray.DataArray): Input data array.
        """
        self.mean = object_in.values
        self.std = object_in.std().values

        # Detect droughts (negative values)
        seq_corr = np.where(self.mean <= 0., 1, 0)
        seq_corr = np.append(seq_corr, 0)  # Add a trailing 0 for clustering logic
        
        # Initial clustering
        cluster_tmp, index_tmp = select_based_on_threshold(seq_corr, object_in)
        
        # Flatten indices of drought clusters
        idx_flat = [x for cluster in index_tmp for x in cluster]
        tmp = np.zeros_like(self.mean)
        tmp[idx_flat] = 1

        # Merge droughts separated by a single point
        tmp2 = tmp.copy()
        if step_pos=='yes':
            for i in range(1, len(tmp) - 1):
                if tmp[i] == 0 and tmp[i - 1] == 1 and tmp[i + 1] == 1:
                    tmp2[i] = 1

        # Re-cluster after merging
        cluster_sort, index_sort = select_based_on_threshold(tmp2, object_in)

        # Extract drought characteristics
        self.index_all = []
        self.cluster = []
        self.drought = []
        self.onset = []
        self.term = []
        self.duration = []

        for i, c in enumerate(index_sort):
            if not c:
                continue
            
            cval = np.array(cluster_sort[i])
            below_std_indices = np.argwhere(cval <= -self.std).flatten()
            min_index = int(np.argmin(cval))

            # Determine onset and termination indices
            if below_std_indices.size == 0 or c[below_std_indices[-1]] >= len(self.mean) - 1:
                continue

            onset = list(range(c[0], c[below_std_indices[0]] + 1))
            if c[min_index] == c[below_std_indices[-1]]:
                term = [c[min_index] + 1]
            else:
                term = list(range(c[min_index] + 1, c[below_std_indices[-1]] + 2))

            # Save results
            mid = list(range(onset[-1], term[0]))
            if mid:
                self.drought.append(mid)
                self.duration.append(len(mid))
                self.onset.append(onset)
                self.term.append(term)
                self.index_all.append(c)
                self.cluster.append(cval.tolist())





class DroughtPR:
    """
    Class to calculate onset and termination indices for drought clusters based on PR.
    """
    
    def __init__(self, var, drought_cluster, min_nt):  #n_thr=3
        """
        Initialize the DroughtPR class.

        Parameters:
            var (xarray.DataArray): Input variable array.
            drought_cluster (tuple): Contains onset, drought, and termination clusters.
            min_nt (int): Minimum threshold for onset and termination durations.
        """
		        
        onset = drought_cluster[0]        
        drought = drought_cluster[1]        
        term = drought_cluster[2]
        
        self.drought = drought
        self.var = var.values
        self.threshold = min_nt
        
        #=== From the onset and termination, 
        #   count period with positive and negative droughts 
        
        dur_onset = [len(x) for x in onset]
        dur_term = [len(x) for x in term]
        
        #print(onset)
        
        if max(dur_onset)>min_nt:
            nt_onset = max(dur_onset)
        else:
            nt_onset = min_nt
            
        if max(dur_term)>min_nt:
            nt_term = max(dur_term)
            if nt_term<nt_onset:
                nt_term = nt_onset
        else:
            nt_term = min_nt
        
        onset_idx = []
        term_idx = []
        
        for d in onset:
            
            idx = np.arange((int(d[0])-nt_onset),(int(d[0])+1))
            idx = idx[(idx>=0) & (idx<len(self.var))]
            var_onset = self.var[idx]
            sel_onset = (np.argwhere(var_onset<0)).flatten()
            onset_idx.append(idx[sel_onset])
        
        for d in term:
            
            idx = np.arange((int(d[-1])-nt_term), int(d[-1])+1)
            idx = idx[(idx>=0) & (idx<len(self.var))]
            var_term = self.var[idx]
            sel_term = (np.argwhere(var_term>0)).flatten()
            term_idx.append(idx[sel_term])
            
        self.onset = onset_idx
        self.term = term_idx
        self.nt = [nt_onset, nt_term]

