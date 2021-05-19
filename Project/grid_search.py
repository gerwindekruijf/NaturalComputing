from multiprocessing import Pool, cpu_count
import argparse

import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from tqdm import tqdm

from decision_tree import gp, fitness

AUS_columns = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15']
AUS_cat = dict(zip(AUS_columns, [1,0,0,1,1,1,0,1,1,0,1,1,0,0,1]))
with open(r"australian/australian.dat", 'r') as f:
    DATA = pd.DataFrame(
        [
            [(float(token) if '−' not in token else -float(token[1:])) for token in line.split()]
            for line in f.readlines()
            ],
        columns=AUS_columns)
LABELS = 'A15'


THREADS = cpu_count()

def perform_gp(sample_size=100, learners=8, generations=8, pop_size=400, mut_rate=0.1, cross_rate=0.9, max_depth=20, cross_md=10, fit_weights=None, multi_proc=False):
    if fit_weights is None:
        fit_weights = [0.5, 0.5, 0.]

    dfs = [DATA.sample(sample_size).reset_index(drop=True) for _ in range(learners)]
    
    params = [(data.drop(columns=LABELS), AUS_cat, data[LABELS], generations, pop_size, mut_rate, cross_rate, max_depth, cross_md, fit_weights, False) for data in dfs]
    res = []
    if multi_proc:
        with Pool(THREADS) as p:
            res = p.starmap(gp, tqdm(params, total=len(params)))
    else:
        for d in tqdm(params, total=len(params)):
            res.append(gp(*d))

    labels = DATA[LABELS].to_numpy()
    res_class = np.array([tree.classify(DATA) for tree in res])
    ensemble = np.array(stats.mode(res_class))[0]

    equal = ensemble[ensemble == labels]
    TPR = len(equal[equal == 1])/len(labels[labels == 1])
    TNR = len(equal[equal == 0])/len(labels[labels == 0])
    
    print(f"FINAL SCORE: TPR: {TPR} and TNR: {TNR} out of {len(labels)} labels used")

    return TPR, TNR, ensemble


def gs_weights(multi_proc):
    results=[]
    i = 5.
    for idx in range(4):
        j = i + idx
        k = 0.1 * i

        weights = [i/10, j/10, k/10]
        print(f"Performing GP + ensemble for normal values and weights: {weights[0]} TPR + {weights[1]} TNR and {weights[2]} depth")
        TPR , TNR, _ = perform_gp(fit_weights=weights, multi_proc=multi_proc)
        hm = 2*TPR*TNR/(TPR + TNR)
        print(f"results are: TPR: {TPR} and TNR: {TNR}")
        results.append((TPR, TNR, hm, weights))

        k = 0.2 * i
        weights = [i/10, j/10, k/10]
        print(f"Performing GP + ensemble for normal values and weights: {weights[0]} TPR + {weights[1]} TNR and {weights[2]} depth")
        TPR , TNR, _ = perform_gp(fit_weights=weights, multi_proc=multi_proc)
        hm = 2*TPR*TNR/(TPR + TNR)
        print(f"results are: TPR: {TPR} and TNR: {TNR}")
        results.append((TPR, TNR, hm, weights))

    return results


def gs_depth(multi_proc):
    results=[]
    for max_depth in range(15, 25):
        for cross_md in range(5, 10):
            print(f"Performing GP + ensemble for normal values, max_depth: {max_depth} and cross_depth: {cross_md}")
            TPR , TNR, _ = perform_gp(max_depth=max_depth, cross_md=cross_md, multi_proc=multi_proc)
            hm = 2*TPR*TNR/(TPR + TNR)
            results.append((TPR, TNR, hm, [max_depth, cross_md]))

    return results


def gs_pop(multi_proc):
    results=[]
    for generation in range(5, 15):
        for pop_size in range(200, 800, 100):
            print(f"Performing GP + ensemble for normal values, generation: {generation} and pop_size: {pop_size}")
            TPR , TNR, _ = perform_gp(generations=generation, pop_size=pop_size, multi_proc=multi_proc)
            hm = 2*TPR*TNR/(TPR + TNR)
            results.append((TPR, TNR, hm, [generation, pop_size]))

    return results


def gs_rates(multi_proc):
    results=[]
    for m in range(10, 100, 10):
        for c in range(10, 100, 10):
            mut_rate = float(m)/100
            cross_rate = float(c)/100
            print(f"Performing GP + ensemble for normal values, mut_rate: {mut_rate} and cross_rate: {cross_rate}")
            TPR , TNR, _ = perform_gp(mut_rate=mut_rate, cross_rate=cross_rate, multi_proc=multi_proc)
            hm = 2*TPR*TNR/(TPR + TNR)
            results.append((TPR, TNR, hm, [mut_rate, cross_rate]))

    return results


def gs_learners(multi_proc):
    results=[]
    for learners in range(2, 20, 20):
        for pop_size in range(200, 600, 50):
            print(f"Performing GP + ensemble for normal values, learners: {learners} and pop_size: {pop_size}")
            TPR , TNR, _ = perform_gp(learners=learners, pop_size=pop_size, multi_proc=multi_proc)
            hm = 2*TPR*TNR/(TPR + TNR)
            results.append((TPR, TNR, hm, [learners, pop_size]))

    return results
