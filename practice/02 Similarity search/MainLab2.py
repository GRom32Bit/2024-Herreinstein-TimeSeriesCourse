import numpy as np
import pandas as pd
import math
import timeit
import random
import mass_ts as mts
from IPython.display import display

from sktime.distances import dtw_distance
from modules.metrics import DTW_distance
from modules.distance_profile import brute_force
from modules.prediction import *
from modules.bestmatch import *
from modules.utils import *
from modules.plots import *
from modules.experiments import *

#Задание 1. ГОТОВО.
ts_url = './datasets/part1/ECG.csv'
query_url = './datasets/part1/ECG_query.csv'
ts = read_ts(ts_url).reshape(-1)
query = read_ts(query_url).reshape(-1)
plot_bestmatch_data(ts, query)
#print(ts)
#print(query)
topK = 5
excl_zone_frac = 0.5
excl_zone = math.ceil(len(query) * excl_zone_frac)
is_normalize = True

DOP=brute_force(ts,query)
#print(ts)
#print(query)
naive_bestmatch_results=topK_match(DOP,excl_zone,topK)
print("%%%",naive_bestmatch_results.get("distances"))
plot_bestmatch_results(ts, query, naive_bestmatch_results,1)
print("Task 1 finished!")
#Судя по тому, что мы нашли паттерн, похожий на образец, есть вероятность, что это относится к сердечному заболеванию.
#Судя по минимальным расстояниям, можно было графики и не строить, но для гарантии мы это сделали.

#Задание 2. ГОТОВО.
mass_dists = mts.mass3(ts, query,450) #Последний аргумент - длина фрагмента - не должен быть меньше размера второго.
BEST_DISTS={}
for i in range(topK):
    KD=int(np.asarray(np.where(mass_dists==min(mass_dists))))
    print(KD)
    BEST_DISTS[KD]=min(mass_dists).real
    mass_dists[KD]=max(mass_dists)+1
plot_bestmatch_results(ts, query, BEST_DISTS,2)


print("Task 2 finished!")

#Задание 3.

#Эксперимент 1.
algorithms = ['brute_force', 'mass', 'mass2', 'mass3']

algorithms_params = {
    'brute_force': None,
    'mass': None,
    'mass2': None,
    'mass3': {'segment_len': 2**6},
}


m = 2**6 # length of query
LenCost=2**8
n_list = list()
for i in range(10):
    n_list.append(LenCost*(2**i))  # lengths of time series

exp1_params = {
    'varying': {'n': n_list},
    'fixed': {'m': m}
}

exp1_data = {
    'ts': dict.fromkeys(map(str, n_list), []),
    'query': {str(m): []}
}

task = 'distance_profile'


# generate set of time series and query
# run experiments for measurement of algorithm runtimes


for n in n_list:
    exp1_data['ts'][str(n)] = random_walk(n)
    exp1_data['query'][str(m)] = random_walk(m)

results = {}
for algorithm in algorithms:
    times = run_experiment(algorithm, task, exp1_data, exp1_params, algorithms_params.get(algorithm))
    results[algorithm] = times

# visualize plot with results of experiment
comparison_param = np.array(algorithms)

times_array = np.array([results[alg] for alg in algorithms])
visualize_plot_times(times_array, np.array(algorithms), exp1_params)

# visualize table with speedup
tab_index = algorithms[1:]
tab_columns = [f"n = {n}" for n in n_list]
tab_title = "Speedup MASS relative to the brute force <br> (variable time series length, fixed query length)"

speedup_data = []
brute_force_times = results['brute_force']
for algorithm in algorithms[1:]:
    mass_times = results[algorithm]
    speedup = calculate_speedup(brute_force_times, mass_times)
    speedup_data.append(speedup)

speedup_data = np.array(speedup_data)
visualize_table_speedup(speedup_data, tab_index, tab_columns, tab_title)
print("\t\tExp 1 OK!")

#Эксперимент 2.
n_const = 2**16 # length of time series
m_start=2**4
m_list=list()
for i in range(10):
    m_list.append(m_start*(2**i)) # lengths of queries

exp2_params = {
    'varying': {'m': m_list},
    'fixed': {'n': n_const}
}

exp2_data = {
    'ts': {str(n_const): []},
    'query': dict.fromkeys(map(str, m_list), [])
}

for i in m_list:
    exp2_data['query'][str(i)] = random_walk(i)
    exp2_data['ts'][str(n_const)] = random_walk(n_const)

algorithms_params = {
    'brute_force': None,
    'mass': None,
    'mass2': None,
    'mass3': {'segment_len': m_list},
}
results2 = {}
for algorithm in algorithms:
    times = run_experiment(algorithm, task, exp2_data, exp2_params, algorithms_params.get(algorithm))
    results2[algorithm] = times

comparison_param = np.array(algorithms)

times_array = np.array([results2[alg] for alg in algorithms])
visualize_plot_times(times_array, np.array(algorithms), exp2_params)

tab_index = algorithms[1:]
tab_columns = [f"m = {i}" for i in m_list]
tab_title = "Speedup MASS relative to the brute force <br> (variable query length, fixed time series length)"

speedup_data = []
brute_force_times = results2['brute_force']
for algorithm in algorithms[1:]:
    mass_times = results2[algorithm]
    speedup = calculate_speedup(brute_force_times, mass_times)
    speedup_data.append(speedup)

speedup_data = np.array(speedup_data)
visualize_table_speedup(speedup_data, tab_index, tab_columns, tab_title)

print("\t\tExp 2 OK!")
print("Task 3 finished!")

#Задание 4. ГОТОВО.
VR1=list()
VR2=list()
for i in range(1000):
    K1=random.randint(-10,10)+random.randint(-2,2)
    K2=random.randint(-10,10)+random.randint(-2,2)
    VR1.append(K1)
    VR2.append(K2)
VR1=np.asarray(VR1)
VR2=np.asarray(VR2)
def test_distances(dist1: float, dist2: float) -> None:
    np.testing.assert_equal(round(dist1, 5), round(dist2, 5), 'Distances are not equal')

DistOwn2=DTW_distance(VR1,VR2)
test_distances(DistOwn2,dtw_distance(VR1,VR2))


print("Task 4 finished!")

#Задание 5. ГОТОПО.
topK = 5
r = 0.01
excl_zone_frac = 0.5
is_normalize = True

ts_url = 'datasets\part1\ECG.csv'#'.\datasets\part1\ECG.csv'
query_url = 'datasets\part1\ECG_query.csv'#'.\datasets\part1\ECG_query.csv'
ts_new = read_ts(ts_url).reshape(-1)
query_new = read_ts(query_url).reshape(-1)

naive_bestmatch_results=NaiveBestMatchFinder(excl_zone_frac, topK,is_normalize,r).perform(ts_new,query_new)
print("!!!",naive_bestmatch_results)
plot_bestmatch_results(ts_new, query_new, naive_bestmatch_results,1)

print("Task 5 finished!")

#Задание 6.

#Эксперимент 1.
algorithm = 'naive'
algorithm_params = {
    'topK': 5,
    'excl_zone_frac': 1,
    'normalize': True,
}

#n_list = [2**10, 2**11, 2**12, 2**13, 2**14, 2**15] # lengths of time series
LenCost=2**5
n_list = list()
for i in range(5):
    n_list.append(LenCost*(2**i))  # lengths of time series

r_list = np.round(np.arange(0, 0.6, 0.1), 2).tolist() # sizes of warping window
m = 2**4 # length of query

exp1_params = {
    'varying': {'n': n_list,
                'r': r_list},
    'fixed': {'m': m}
}
exp1_data = {
    'ts': dict.fromkeys(map(str, n_list), []),
    'query': {str(m): []}
}

task = 'best_match'

# generate set of time series and query
for n in n_list:
    exp1_data['ts'][str(n)]=random_walk(n)
exp1_data['query'][str(m)]=random_walk(m)

# run experiments for measurement of algorithm runtimes
res = run_experiment(algorithm, task, exp1_data, exp1_params, algorithm_params)
print(res)

# visualize plot with results of experiment
comparison_param = np.array(r_list)

#comparison_param
visualize_plot_times(res, comparison_param, exp1_params)
print("\t\tExp 1 OK!")
#Эксперимент 2.
m_start=2**2
m_list=list()
for i in range(5):
    m_list.append(m_start*(2**i)) # lengths of queries
r_list = np.round(np.arange(0, 0.6, 0.1), 2).tolist() # sizes of warping window
n = 2**10 # length of time series

exp2_params = {
    'varying': {'m': m_list,
                'r': r_list},
    'fixed': {'n': n}
}
exp2_data = {
    'ts': {str(n): []},
    'query': dict.fromkeys(map(str, m_list), []),
}
for m in m_list:
    exp2_data['query'][str(m)] = random_walk(m)
exp2_data['ts'][str(n)] = random_walk(n)

res2 = run_experiment(algorithm, task, exp2_data, exp2_params, algorithm_params)
print(res2)


comparison_param = np.array(r_list)
visualize_plot_times(res2, comparison_param, exp2_params)

#Время выполнения растёт пропорционально размеру ряда или запроса.
#Ширина полосы сильнее влияет на время выполнения, когда размер запроса высокий.
print("\t\tExp 2 OK!")
print("Task 6 finished!")

#Задание 7.



#Задание 8.



#Задание 9.



