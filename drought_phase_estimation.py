"""
  Return indices of of drought onset, drought, termination
  (defined as Kim et al. (2024)), and the duration of droughts. 
  Input: a soil moisture anomaly (or other standardized drought indices) in xarray or np.array

"""

import numpy as np
import xarray as xr


class drought():
    
    def __init__(self, object_in):  
        
        """
        Estimate drought phases: Onset, termination, drought
        object_in should be a time series of drought in xarray or numpy.array
        """
        
        if type(object_in)==np.ndarray: 
            self.mean = object_in
            self.std = np.nanstd(object_in)
            
        elif type(object_in) == xr.core.dataarray.DataArray:
            self.mean = object_in.values
            self.std = object_in.std().values
            
        #--- Negative are all replaced to 1
        seq_corr = np.where(self.mean <=0., 1, 0 )
        seq_corr =np.append(seq_corr, 0)
        
        #--- Detecting all droughts: Get all 1. 
        
        clusters = []
        index_cluster = []
        
        ind_tmp=[]
        tmp=[]
        
        for i,k in enumerate(seq_corr):
            if k==1:   
                tmp.append(self.mean[i])
                ind_tmp.append(i)
            else:
                if len(tmp)>=1:
                    clusters.append(tmp)
                    index_cluster.append(ind_tmp)
                tmp=[]
                ind_tmp=[]
        
        """
        From the sorted indices, get only clusters that fulfill the criteria:
        Drought indices need to have at least one negative std (or below). 
        """
        
        cluster_sort =[]
        index_sort = []
        
        for i,c in enumerate(clusters):
            criteria = np.array(c)[c<=(-self.std)]
            ind_tmp = index_cluster[i]
            if len(criteria)>0:
                cluster_sort.append(c)
                index_sort.append(ind_tmp)
        
        """
        Separating onset, drought and termination from the sorted indices
        Onset: from dry (or wet) to before <-std
        Terminaton: all from >- std to the end (dry or wet). 
        Drought: excluding onset and termination
        """
        
        onset_i=[]   # Indices that pertain to onsets
        term_i=[]    # Indices that pertain to termination
        drought_i = []   # Indices that pertain to drought
        duration = []    # duration of each drought event (from drought i)
        odt_i = []
        
        for i,c in enumerate(index_sort):
            cval = np.array(cluster_sort[i])
            x = np.where(cval<(-self.std))[0]  # index
            
            # Onset
            # cluster doesn't start from 0
            if c[0]!=0:
                if x[0]==0:
                    onset = [int(c[x[0]]-1)]
                else:
                    onset = np.arange(c[0],c[x[0]]).tolist()
            
            if c[-1]!=len(self.mean):
                if c[x[-1]]==c[-1]:
                    term = [int(c[x[-1]]+1)]
                else:
                    term = np.arange(c[x[-1]]+1, c[-1]+1).tolist()
            
            mid = np.arange(onset[-1]+1,term[0]).tolist()
            
            duration.append(len(mid))
            onset_i.append(onset)
            term_i.append(term)
            drought_i.append(mid)
            odt_i.append(onset + mid + term)
        
        self.index_odt = odt_i
        self.drought = drought_i
        self.onset = onset_i
        self.term = term_i
        self.duration = duration
