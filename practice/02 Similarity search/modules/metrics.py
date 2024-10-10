import numpy as np

def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    ed_dist = np.sqrt(np.sum((ts1 - ts2)**2))
    return ed_dist

def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))
    norm_ed_dist = ed_dist / np.sqrt(len(ts1))
    return norm_ed_dist

def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    n = len(ts1)
    d = np.zeros((n+1, n+1))
    d[:, 0] = np.inf
    d[0, :] = np.inf
    d[0][0] = 0
    r=n*0.11 #При меньшем значении r сумма не сходится. (хотя обычно при n*0.07 уже сходится часто).
    for i in range(1, n+1):
        for j in range(1, n+1):
            #print("CELL:",i,j)
            if i>j+r or i<j-r:
                d[i,j]=np.inf
                continue
            else:
                d[i][j] = np.power((ts1[i-1] - ts2[j-1]), 2) + np.min([d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]])
    return d[n][n]

