import numpy as np

from modules.metrics import *
from modules.utils import z_normalize


class DistanceMatrix:
    def __init__(self, metric : str = 'euclidean', normalize : bool = False) -> None:

        self._metric : str = metric
        self._normalize : bool = normalize
        self._shape : tuple[int, int] = (0, 0) 
        self._values : np.ndarray | None = None


    @property
    def values(self) -> np.ndarray:
        return self._values


    @property
    def shape(self) -> tuple[int, int]:
        return self._shape


    @property
    def distance_metric(self) -> str:
        DOP = ""
        if (self._normalize):
            DOP = "normalized "
        else:
            DOP = "non-normalized "

        return DOP + self._metric + " distance"


    def _choose_distance(self):
        """ Choose distance function for calculation of matrix

        Returns
        -------
        dict_func: function reference
        """

        dist_func = None

       # INSERT YOUR CODE

        return dist_func


    def calculate(self, input_data: np.ndarray) -> None:
        """ Calculate distance matrix

        Parameters
        ----------
        input_data: time series set
        """

        # INSERT YOUR CODE
