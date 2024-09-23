import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from typing_extensions import Self

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class TimeSeriesHierarchicalClustering:
    def __init__(self, n_clusters: int = 3, method: str = 'complete') -> None:
        self.n_clusters: int = n_clusters
        self.method: str = method
        self.model: AgglomerativeClustering | None = None
        self.linkage_matrix: np.ndarray | None = None

    def _create_linkage_matrix(self) -> np.ndarray:
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)

        for i, merge in enumerate(self.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([self.model.children_, self.model.distances_, counts]).astype(float)

        return linkage_matrix

    def fit(self, distance_matrix: np.ndarray) -> Self:
        # Создаем модель с использованием предрасчитанных расстояний
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, metric='precomputed', linkage=self.method, compute_distances=True)

        # Получаем метки кластеров
        self.labels_ = self.model.fit_predict(distance_matrix)

        self.linkage_matrix = self._create_linkage_matrix()
        return self

    def fit_predict(self, distance_matrix: np.ndarray) -> np.ndarray:
        self.fit(distance_matrix)
        return self.labels_


    def _draw_timeseries_allclust(self, dx: pd.DataFrame, labels: np.ndarray, leaves: list[int], gs: gridspec.GridSpec, ts_hspace: int) -> None:
        """ 
        Plot time series graphs beside dendrogram

        Parameters
        ----------
        dx: timeseries data with column "y" indicating cluster number
        labels: labels of dataset's instances
        leaves: leave node names from scipy dendrogram
        gs: gridspec configurations
        ts_hspace: horizontal space in gridspec for plotting time series
        """

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        margin = 7

        max_cluster = len(leaves)
        # flip leaves, as gridspec iterates from top down
        leaves = leaves[::-1]

        for cnt in range(len(leaves)):
            plt.subplot(gs[cnt:cnt+1, max_cluster-ts_hspace:max_cluster])
            plt.axis("off")

            # get leafnode name, which corresponds to original data index
            leafnode = leaves[cnt]
            ts = dx[leafnode]
            ts_len = ts.shape[0] - 1

            label = int(labels[leafnode])
            color_ts = colors[label]

            plt.plot(ts, color=color_ts)
            plt.text(ts_len+margin, 0, f'class = {label}')


    def plot_dendrogram(self, df: pd.DataFrame, labels: np.ndarray, ts_hspace: int = 12, title: str = 'Dendrogram') -> None:
        """ 
        Draw agglomerative clustering dendrogram with timeseries graphs for all clusters.

        Parameters
        ----------
        df: dataframe with each row being the time window of readings
        labels: labels of dataset's instances
        ts_hspace: horizontal space for timeseries graph to be plotted
        title: title of dendrogram
        """

        max_cluster = len(self.linkage_matrix) + 1

        plt.figure(figsize=(12, 9))

        # define gridspec space
        gs = gridspec.GridSpec(max_cluster, max_cluster)

        # add dendrogram to gridspec
        # add -1 to give timeseries graphs more space
        plt.subplot(gs[:, 0 : max_cluster - ts_hspace - 1])
        plt.xlabel("Distance")
        plt.ylabel("Cluster")
        plt.title(title, fontsize=16, weight='bold')

        ddata = dendrogram(self.linkage_matrix, orientation="left", color_threshold=sorted(self.model.distances_)[-2], show_leaf_counts=True)

        self._draw_timeseries_allclust(df, labels, ddata["leaves"], gs, ts_hspace)