"""
Run multiple GPs to determine optimal parameter values
"""
from multiprocessing import Pool
import sys
import random as r

import numpy as np
from scipy import stats
from tqdm import tqdm

from decision_tree import dt_gp
from data import DATA, LABEL, CATS, sample_data, train_test_data

# TODO run baseline
def ensemble_learning(
    sample_size=1800, learners=8, generations=8, pop_size=500, mut_rate=0.3, cross_rate=0.6,
    fit_weights=None, max_depth=16, cross_md=9, multi_proc=False, use_all_labels=False):
    """
    Ensemble learning algorithm

    :param sample_size: datasampling size, defaults to 100
    :type sample_size: int, optional
    :param learners: number of GP learners, defaults to 8
    :type learners: int, optional
    :param generations: number of generations, defaults to 8
    :type generations: int, optional
    :param pop_size: number of trees in parent group, defaults to 500
    :type pop_size: int, optional
    :param mut_rate: mutation rate, defaults to 0.3
    :type mut_rate: float, optional
    :param cross_rate: crossover rate, defaults to 0.6
    :type cross_rate: float, optional
    :param fit_weights: importance for all classifications + depth penalty, defaults to None
    :type fit_weights: list, optional
    :param max_depth: maximum depth of a tree, defaults to 16
    :type max_depth: int, optional
    :param cross_md: maximum depth of subtree to combine with child, defaults to 9
    :type cross_md: int, optional
    :param multi_proc: multiprocessing, defaults to False
    :type multi_proc: bool, optional
    :return: correctness rates
    :rtype: list
    """
    # None as default, due to danger of modifying default params
    if fit_weights is None:
        fit_weights = [0.5, 0.5, 0.]

    train, test = train_test_data(DATA)

    dfs = sample_data(train, learners, sample_size, use_all_labels)
    #[DATA.sample(sample_size).reset_index(drop=True) for _ in range(learners)]

    # Specify seed, otherwise could go wrong using multiple threads
    params = [
        (data.drop(columns=LABEL), CATS, data[LABEL], generations, pop_size,
        mut_rate, cross_rate, fit_weights, max_depth, cross_md, False, r.randrange(sys.maxsize)) for data in dfs]
    res = []
    if multi_proc:
        with Pool() as pool:
            res = pool.starmap(dt_gp, tqdm(params, total=len(params)))
    else:
        for func_param in tqdm(params):
            res.append(dt_gp(*func_param))

    labels = test[LABEL].to_numpy()

    # res_class = []
    # if multi_proc:
    #     with Pool() as pool:
    #         def temp(tree):
    #             return tree.classify(DATA)
    #         res_class.append(pool.map(temp, tqdm(res, total=len(res))))
    # else:
    #     for tree in tqdm(res):
    #         res_class.append(tree.classify(DATA))
    # res_class = np.array(res_class)

    res_class = np.array([tree.classify(test) for tree in tqdm(res)])
    ensemble = np.array(stats.mode(res_class))[0]

    equal = ensemble[ensemble == labels]
    result = []
    for i in range(len(fit_weights)-1):
        result.append(len(equal[equal == i])/len(labels[labels==i]))

    print(f"FINAL SCORE: TPR: {result[1]} and TNR: {result[0]} out of {len(labels)} labels used")
    return result


def gs_weights(multi_proc):
    """
    Grid search comparing different weigths

    :param multi_proc: multiprocessing
    :type multi_proc: bool
    :return: parameters and matching results for each iteration
    :rtype: list
    """
    results=[]
    i = 5.
    for idx in range(4):
        j = i + idx
        k = 0.1 * i

        weights = [i/10, j/10, k/10]
        print(
            "Performing GP + ensemble for normal values and"
            f" weights: {weights[0]} TPR + {weights[1]} TNR and {weights[2]} depth")

        t_rates = ensemble_learning(fit_weights=weights, multi_proc=multi_proc)
        h_mean = stats.hmean(t_rates)
        print(f"results are: TPR: {t_rates[1]} and TNR: {t_rates[0]}")
        results.append((t_rates, h_mean, weights))

        k = 0.2 * i
        weights = [i/10, j/10, k/10]
        print(
            "Performing GP + ensemble for normal values and"
            f" weights: {weights[0]} TPR + {weights[1]} TNR and {weights[2]} depth")

        t_rates = ensemble_learning(fit_weights=weights, multi_proc=multi_proc)
        h_mean = stats.hmean(t_rates)
        print(f"results are: TPR: {t_rates[1]} and TNR: {t_rates[0]}")
        results.append((t_rates, h_mean, weights))

    return results


def gs_depth(multi_proc):
    """
    Grid search comparing maximum depth and crossover depth

    :param multi_proc: multiprocessing
    :type multi_proc: bool
    :return: parameters and matching results for each iteration
    :rtype: list
    """
    results=[]
    for max_depth in range(10, 31, 10):
        for cross_md in range(5, 16, 5):
            print(f"Performing GP + ensemble for normal values, max_depth: {max_depth} and cross_depth: {cross_md}")
            t_rates = ensemble_learning(max_depth=max_depth, cross_md=cross_md, multi_proc=multi_proc)
            h_mean = stats.hmean(t_rates)
            results.append((t_rates, h_mean, [max_depth, cross_md]))

    return results


def gs_pop(multi_proc):
    """
    Grid search comparing number of generations and population size

    :param multi_proc: multiprocessing
    :type multi_proc: bool
    :return: parameters and matching results for each iteration
    :rtype: list
    """
    results=[]
    for generation in range(5, 16, 5):
        for pop_size in range(200, 601, 200):
            print(f"Performing GP + ensemble for normal values, generation: {generation} and pop_size: {pop_size}")
            t_rates = ensemble_learning(generations=generation, pop_size=pop_size, multi_proc=multi_proc)
            h_mean = stats.hmean(t_rates)
            results.append((t_rates, h_mean, [generation, pop_size]))

    return results


def gs_rates(multi_proc):
    """
    Grid search comparing mutation- and crossover rate

    :param multi_proc: multiprocessing
    :type multi_proc: bool
    :return: parameters and matching results for each iteration
    :rtype: list
    """
    results=[]
    for mutation in range(0, 51, 25):
        for crossover in range(50, 91, 20):
            mut_rate = float(mutation)/100
            cross_rate = float(crossover)/100
            print(f"Performing GP + ensemble for normal values, mut_rate: {mut_rate} and cross_rate: {cross_rate}")
            t_rates = ensemble_learning(mut_rate=mut_rate, cross_rate=cross_rate, multi_proc=multi_proc)
            h_mean = stats.hmean(t_rates)
            results.append((t_rates, h_mean, [mut_rate, cross_rate]))

    return results


def gs_learners(multi_proc):
    """
    Grid search comparing number of learners and population sizes

    :param multi_proc: multiprocessing
    :type multi_proc: bool
    :return: parameters and matching results for each iteration
    :rtype: list
    """
    results=[]
    # TODO, deze maar dan met alle waardes van de label=1, zelfde aantal negatives
    # TODO, Miss ook een paar over alle train data
    for learners in range(5, 16, 5):
        for sample_size in range(1000, 5001, 2000):
            print(f"Performing GP + ensemble for normal values, learners: {learners} and sample_size: {sample_size}")
            t_rates = ensemble_learning(learners=learners, sample_size=sample_size, multi_proc=multi_proc, use_all_labels=True)
            h_mean = stats.hmean(t_rates)
            results.append((t_rates, h_mean, [learners, sample_size]))

    return results
