import os
practice_dir_path = 'D:/2024-Herreinstein-TimeSeriesCourse-main/practice/01 Basics/pythonProject'
os.chdir(practice_dir_path)

#%load_ext autoreload
#%autoreload 2

import numpy as np
import random
from sktime.distances import euclidean_distance, dtw_distance, pairwise_distance
from sklearn.metrics import silhouette_score

import cv2
import imutils
import glob
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.pairwise_distance import PairwiseDistance
from modules.clustering import TimeSeriesHierarchicalClustering
from modules.classification import TimeSeriesKNN, calculate_accuracy
from modules.image_converter import image2ts
from modules.utils import read_ts, z_normalize, sliding_window, random_walk
from modules.plots import plot_ts

import math

#Генерация двух временных рядов.
VR1=list()
VR2=list()
SZ=100
for i in range(SZ):
    VR1.append(random.randint(1, 10))
    VR2.append(random.randint(1, 10))
VR1=np.asarray(VR1)
VR2=np.asarray(VR2)

#Генерация множества временных рядов.
TR=list()
LS=10
while len(TR)<LS:
    VR=list()
    for i in range(SZ):
        VR.append(random.randint(1, 10))
    TR.append(VR)
TR=np.asarray(TR)

#Задание 1. Готово.
def test_distances(dist1: float, dist2: float) -> None:
    np.testing.assert_equal(round(dist1, 5), round(dist2, 5), 'Distances are not equal')
def test_matrices(matrix1 : np.ndarray, matrix2 : np.ndarray) -> None:
    np.testing.assert_equal(matrix1.round(5), matrix2.round(5), 'Matrices are not equal')

def EuclDist(a,b):
    S=0
    for i in range(len(a)):
        S+=(a[i]-b[i])**2
    return math.sqrt(S)
DistOwn1=EuclDist(VR1,VR2)
test_distances(DistOwn1,euclidean_distance(VR1,VR2))
print("Task 1 finished!")


#Задание 2. Готово.
def DTWDist(a,b):
    n = len(a)
    d = np.zeros((n+1, n+1))
    d[:, 0] = np.inf
    d[0, :] = np.inf
    d[0][0] = 0
    for i in range(1, n+1):
        for j in range(1, n+1):
            d[i][j] = np.power((a[i-1] - b[j-1]), 2) + np.min([d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]])
    return d[n][n]
DistOwn2=DTWDist(VR1,VR2)
test_distances(DistOwn2,dtw_distance(VR1,VR2))
print("Task 2 finished!")


#Задание 3. Готово.
A1=pairwise_distance(TR, metric='euclidean')
A2=PairwiseDistance(metric='euclidean').calculate(TR)
B1=pairwise_distance(TR, metric="dtw")
B2=PairwiseDistance(metric='dtw').calculate(TR)
test_matrices(A1,A2)
test_matrices(B1,B2)
print("Task 3 finished!")

#Задание 4.
url = './datasets/part1/CBF_TRAIN.txt' #Датасет содержит 30 искусственно созданных временных рядов.
data = read_ts(url)
ts_set = data.iloc[:, 1:]
labels = data.iloc[:, 0]
#print(ts_set)
ts_set=np.asarray(ts_set)
plot_ts(ts_set)
#print(ts_set)
# Создаем экземпляр класса PairwiseDistance для вычисления матриц расстояний
pairwise_distance = PairwiseDistance()

# 1. Кластеризация с использованием евклидовой метрики
# Вычисляем матрицу расстояний
EDM = PairwiseDistance(metric='euclidean', is_normalize='true').calculate(ts_set)

# Выполняем кластеризацию
HCeuc = TimeSeriesHierarchicalClustering(n_clusters=3, method='average')
ELabs = HCeuc.fit_predict(EDM)

# Визуализируем результаты в виде дендрограммы
HCeuc.plot_dendrogram(ts_set, ELabs, ts_hspace=5, title='Дендрограмма 1')
print("Where dendrogram 1?")

# 2. Кластеризация с использованием DTW
# Вычисляем матрицу расстояний для DTW
dtw_distance_matrix = PairwiseDistance(metric='dtw', is_normalize='true').calculate(ts_set)

# Выполняем кластеризацию
HCdtw = TimeSeriesHierarchicalClustering(n_clusters=3, method='average')
DLabs = HCdtw.fit_predict(dtw_distance_matrix)

# Визуализируем результаты в виде дендрограммы
HCdtw.plot_dendrogram(ts_set, DLabs, ts_hspace=5, title='Дендрограмма 2')
print("Where dendrogram 2?")


n_clusters = list()
for i in range(2,11):
    n_clusters.append(i)
for i in n_clusters:
    HCeuc = TimeSeriesHierarchicalClustering(n_clusters=i, method='average')
    ELabs = HCeuc.fit_predict(EDM)
    HCdtw = TimeSeriesHierarchicalClustering(n_clusters=i, method='average')
    DLabs = HCdtw.fit_predict(dtw_distance_matrix)

    silhouette_DTW = silhouette_score(ts_set, DLabs)
    silhouette_EU = silhouette_score(ts_set, ELabs)
    print("For DTW\t n_clusters =", i, "average silhouette_score is :", silhouette_DTW)
    print("For EU\t n_clusters =", i, "average silhouette_score is :", silhouette_EU)
    print()
    #Чем ближе к 1, тем больше пересечение силуэтов. Поэтому с существенным отрывом лучше эвклидова метрика. По-хорошему чем больше кластеров, тем хуже результаты.
    #Эвклидово расстояние применяется только для рядов одной длины.
    #DTW подходит для временных рядов, имеющих похожие подпоследовательности (смещённые по времени или сжатые/растянутые). В нашем случае данную подпоследовательность выделить тяжело.
print("Task 4 finished!")

#Задание 5. Готово.
def const_find(a):
    mut=sum(a)/len(a)
    S=0
    for i in range(len(a)):
        S+=a[i]**2
    sig=math.sqrt(S/len(a)-mut**2)
    return mut,sig
def norm_ED_distance(a,b):
    n=len(a)
    mut1,sig1=const_find(a)
    mut2,sig2=const_find(b)
    SPVR=0
    for i in range(n):
        SPVR+=a[i]*b[i]
    D=(SPVR-n*mut1*mut2)/(n*sig1*sig2)
    return math.sqrt(abs(2*n*(1-D)))
DistOwn5=norm_ED_distance(VR1,VR2)
test_distances(DistOwn5,euclidean_distance(z_normalize(VR1),z_normalize(VR2)))
print("Task 5 finished!")

#Задание 6.
url1 = './datasets/part2/chf10.csv'
ts1 = read_ts(url1)
url2 = './datasets/part2/chf11.csv'
ts2 = read_ts(url2)


ts_set = np.concatenate((ts1, ts2), axis=1).T
plot_ts(ts_set)
m = 125
subs_set1 = sliding_window(ts_set[0], m, m - 1)
subs_set2 = sliding_window(ts_set[1], m, m - 1)

subs_set = np.concatenate((subs_set1[0:15], subs_set2[0:15]))
labels = np.array([0] * subs_set1[0:15].shape[0] + [1] * subs_set2[0:15].shape[0])

#Дополнительная ссылка на методы иерархической кластеризации и разницу между ними.
#https://stats.stackexchange.com/questions/195446/choosing-the-right-linkage-method-for-hierarchical-clustering

#Рассчитываем расстояния и визуализируем результаты для классической евклидовой метрики.
distance_matrix_euclidean = PairwiseDistance(metric='euclidean', is_normalize=False).calculate(subs_set)
clustering_euclidean = TimeSeriesHierarchicalClustering(n_clusters=2, method='complete').fit(distance_matrix_euclidean)
clustering_euclidean.plot_dendrogram(subs_set, labels, title='Dendrogram for Euclidean Distance')

#Рассчитываем расстояния и визуализируем результаты для нормализованной евклидовой метрики.
distance_matrix_norm_euclidean = PairwiseDistance(metric='euclidean', is_normalize=True).calculate(subs_set)
clustering_norm_euclidean = TimeSeriesHierarchicalClustering(n_clusters=2, method='complete').fit(distance_matrix_norm_euclidean)
clustering_norm_euclidean.plot_dendrogram(subs_set, labels, title='Dendrogram for Normalized Euclidean Distance')

silhouette_eucl = silhouette_score(subs_set,clustering_euclidean.fit_predict(distance_matrix_euclidean))
silhouette_norm_eucl = silhouette_score(subs_set,clustering_norm_euclidean.fit_predict(distance_matrix_norm_euclidean))

print('eucl \t silhouette score:\t\t{0}'.format(silhouette_eucl))
print('eucl norm silhouette score:\t\t{0}'.format(silhouette_norm_eucl))
print("Task 6 finished!")
#Задание 7.



#Задание 8.