import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize


class PairwiseDistance:
    """
    Distance matrix between time series

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:
        self.metric: str = metric
        self.is_normalize: bool = is_normalize

    @property
    def distance_metric(self) -> str:
        norm_str = ""
        if (self.is_normalize):
            norm_str = "normalized "
        else:
            norm_str = "non-normalized "

        return norm_str + self.metric + " distance"

    def _choose_distance(self):
        if self.metric == 'euclidean':
            return norm_ED_distance if self.is_normalize else ED_distance
        elif self.metric == 'dtw':
            return DTW_distance
        else:
            raise ValueError(f"Metric '{self.metric}' is not supported. Choose 'euclidean' or 'dtw'.")

    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        N = input_data.shape[0]
        RetMat = np.zeros((N, N))

        # Выбор функции расстояния
        dist_func = self._choose_distance()

        # Вычисление матрицы расстояний
        for i in range(0,N):
            for j in range(i + 1, N):
                distance = dist_func(input_data[i], input_data[j])
                RetMat[i, j] = distance
                RetMat[j, i] = distance  # матрица симметрична
        return RetMat
