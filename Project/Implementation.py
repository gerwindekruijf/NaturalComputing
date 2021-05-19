"""
main module of the code
"""
import argparse
import sys
import random as r

import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from decision_tree import dt_gp
from grid_search import AUS_cat, LABELS, DATA
import grid_search as gs


GRID_SEARCH = {
    "fitness_weights": gs.gs_weights,
    "depth": gs.gs_depth,
    "population": gs.gs_pop,
    "rates": gs.gs_rates,
    "learners": gs.gs_learners,
}
# TODO: optimalizatie voor: 1. Max_depth samen met cross_depth
#                           2. Generaties samen met learners en pop
#                               COMMENTAAR: HEB LEARNERS WEGGELATEN, leek overbodig
#                           3. Cros_rate en mut_rate
#                           4. Learners (optimale waarde uit 2. iig in gebruiken) en sample_size
#                               COMMENTAAR: geen optimale waarde uit 2 :(
#         (runnen proberen onder het uur te houden, elke run is ongeveer 1 minuut)


def norm(data):
    """
    normalize data

    :param data: data
    :type data: dataframe
    :return: data
    :rtype: dataframe
    """
    values = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    values_scaled = min_max_scaler.fit_transform(values)
    return pd.DataFrame(values_scaled, columns=data.columns)


def single_gp(seed):
    """
    generate a single tree using gp

    :param seed: seed used for random
    :type seed: int
    :return: fitness scores
    :rtype: float
    """
    train = DATA.sample(frac=0.8).reset_index(drop=True)
    test = DATA.drop(train.index)

    weights = [0.5, 0.5, 0.]
    res_tree = dt_gp(
        train.drop(columns=LABELS), AUS_cat, train[LABELS], 8, 500, 0.3, 0.6, weights, 20, 10, False, seed)

    labels = test[LABELS]
    gen_labels = res_tree.classify(test.drop(columns=LABELS))

    equal = gen_labels[gen_labels == labels]

    result = []
    for i in range(len(weights)-1):
        result.append(len(equal[equal == i])/len(labels[labels==i]))

    return result



def implement(arguments):
    """
    main program implementation

    :param arguments: optional arguments
    :type arguments: dictionary
    """
    if arguments.test:
        # Kunnen we gebruiken om dingen te testen als we dat willen.
        print("This does nothing")

    elif arguments.onegp:
        print("Generating trees using random data")
        runs = 10

        # Specify seed, otherwise could go wrong using multiple threads
        params = [(r.randrange(sys.maxsize)) for _ in range(runs)]

        results = []
        if arguments.multi:
            results = process_map(single_gp, params)
        else:
            for func_param in tqdm(params):
                results.append(single_gp(func_param))

        results = np.array(results)
        print(f"mean: {np.mean(results, axis=0)} std: {np.std(results, axis=0)} for {runs} runs")

    elif arguments.gridsearch:
        print(f"first the optimalization for {arguments.gridsearch}:")
        results = GRID_SEARCH[arguments.gridsearch](arguments.multi)

        results = sorted(results, key = lambda x: x[2], reverse=True)
        print("results (sorted on harmonic mean of TPR and TNR) found:")
        print(results)

    else:
        gs.ensemble_learning(100, 8, 8, 400, 0.3, 0.7, [0.5, 0.5, 0.], 20, 10, args.multi)


if __name__ == '__main__':
    DATA = norm(DATA)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true",
                        help="testing possibility")
    parser.add_argument("-m","--multi", action="store_true",
                        help="multiprocessing for room simulation")
    parser.add_argument("-og", "--onegp", action="store_true",
                        help="performs no ensemble method + statistical checking of value")
    parser.add_argument("-gs", "--gridsearch", choices=GRID_SEARCH.keys(),
                        help="performs a grid search for a hyperparameter optimalization procedure")
    args = parser.parse_args()
    implement(args)
