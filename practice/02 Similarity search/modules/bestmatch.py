import numpy as np
import math
import copy

from modules.utils import sliding_window, z_normalize
from modules.metrics import DTW_distance


def apply_exclusion_zone(array: np.ndarray, idx: int, excl_zone: int) -> np.ndarray:
    zone_start = max(0, idx - excl_zone)
    zone_stop = min(array.shape[-1], idx + excl_zone)
    array[zone_start : zone_stop + 1] = np.inf

    return array


def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int, max_distance: float = np.inf) -> dict:
    topK_match_results = {
        'indices': [],
        'distances': []
    } 

    dist_profile_len = len(dist_profile)
    dist_profile = np.copy(dist_profile).astype(float)

    for k in range(topK):
        min_idx = np.argmin(dist_profile)
        min_dist = dist_profile[min_idx]

        if (np.isnan(min_dist)) or (np.isinf(min_dist)) or (min_dist > max_distance):
            break

        dist_profile = apply_exclusion_zone(dist_profile, min_idx, excl_zone)

        topK_match_results['indices'].append(min_idx)
        topK_match_results['distances'].append(min_dist)

    return topK_match_results


class BestMatchFinder:
    """
    Base Best Match Finder
    
    Parameters
    ----------
    excl_zone_frac : exclusion zone fraction
    topK : number of the best match subsequences
    is_normalize : z-normalize or not subsequences before computing distances
    r : warping window size
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05) -> None:
        """ 
        Constructor of class BestMatchFinder
        """

        self.excl_zone_frac: float = excl_zone_frac
        self.topK: int = topK
        self.is_normalize: bool = is_normalize
        self.r: float = r


    def _calculate_excl_zone(self, m: int) -> int:
        """
        Calculate the exclusion zone
        
        Parameters
        ----------
        m : length of subsequence
        
        Returns
        -------
        excl_zone: exclusion zone
        """

        excl_zone = math.ceil(m * self.excl_zone_frac)

        return excl_zone



class NaiveBestMatchFinder(BestMatchFinder):
    """
    Naive Best Match Finder
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class NaiveBestMatchFinder
        """

    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        #print(len(ts_data),len(query))

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2):  # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape
        print("SIZES:",N,m)
        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,)) * np.inf
        bsf = np.inf

        bestmatch = {
            'index': [],
            'distance': []
        }

        if self.is_normalize:
            query = z_normalize(query)
        for i in range(1, N - m + 1):
            if self.is_normalize:
                ts_data[i:m] = z_normalize(ts_data[i])
            dist = DTW_distance(query, ts_data[i], self.r)
            if dist < bsf:
                dist_profile[i] = dist
                bestmatch = topK_match(dist_profile, excl_zone, self.topK)
                if np.max(bestmatch['distances']) == self.topK:
                    bsf = np.max(DTW_distance(query, ts_data[i]), self.r)
            #print("CURR:",bestmatch)
        return bestmatch


class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder
    
    Additional parameters
    ----------
    not_pruned_num : number of non-pruned subsequences
    lb_Kim_num: number of subsequences that pruned by LB_Kim bounding
    lb_KeoghQC_num : number of subsequences that pruned by LB_KeoghQC bounding
    lb_KeoghCQ_num : number of subsequences that pruned by LB_KeoghCQ bounding
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 5, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class UCR_DTW
        """        

        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0


    def _LB_Kim(self, subs1: np.ndarray, subs2: np.ndarray) -> float:
        """
        Compute LB_Kim lower bound between two subsequences
        
        Parameters
        ----------
        subs1 : the first subsequence
        subs2 : the second subsequence
        
        Returns
        -------
        lb_Kim : LB_Kim lower bound
        """

        lb_Kim = 0
        
        # INSERT YOUR CODE

        return lb_Kim


    def _LB_Keogh(self, subs1: np.ndarray, subs2: np.ndarray, r: float) -> float:
        """
        Compute LB_Keogh lower bound between two subsequences
        
        Parameters
        ----------
        subs1 : the first subsequence
        subs2 : the second subsequence
        r : warping window size
        
        Returns
        -------
        lb_Keogh : LB_Keogh lower bound
        """

        lb_Keogh = 0

        # INSERT YOUR CODE

        return lb_Keogh


    def get_statistics(self) -> dict:
        """
        Return statistics on the number of pruned and non-pruned subsequences of a time series   
        
        Returns
        -------
        Dictionary containing statistics
        """

        statistics = {
            'not_pruned_num': self.not_pruned_num,
            'lb_Kim_num': self.lb_Kim_num,
            'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
            'lb_KeoghQC_num': self.lb_KeoghQC_num
        }

        return statistics

    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using the naive algorithm

        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of the naive algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2):  # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape
        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,)) * np.inf
        bsf = np.inf

        bestmatch = {
            'index': [],
            'distance': []
        }

        if self.is_normalize:
            query = z_normalize(query)
        for i in range(1, N - m + 1):
            if self.is_normalize:
                ts_data[i:m] = z_normalize(ts_data[i])
            dist = DTW_distance(query, ts_data[i], self.r)
            if dist < bsf:
                dist_profile[i] = dist
                bestmatch = topK_match(dist_profile, excl_zone, self.topK)
                if np.max(bestmatch['distances']) == self.topK:
                    bsf = np.max(DTW_distance(query, ts_data[i]), self.r)

        return bestmatch
