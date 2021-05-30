"""
main module of the code
"""
import argparse
import sys
import random as r

import numpy as np
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from decision_tree import dt_gp
from data import CATS, LABEL, DATA, norm, train_test_data
import grid_search as gs


GRID_SEARCH = {
    "fitness_weights": gs.gs_weights,   # weights
    "depth": gs.gs_depth,               # maximum depth and cross rate depth
    "population": gs.gs_pop,            # number of generations and population
    "rates": gs.gs_rates,               # corssover and mutation rate
    "learners": gs.gs_learners,         # learners is ensemble and sample size
}


def single_gp(seed):
    """
    generate a single tree using gp

    :param seed: seed used for random
    :type seed: int
    :return: fitness scores
    :rtype: float
    """
    train, test = train_test_data(DATA)

    weights = [0.5, 0.5, 0.]
    res_tree = dt_gp(
        train.drop(columns=LABEL), CATS, train[LABEL], 8, 500, 0.3, 0.6, weights, 20, 10, False, seed)

    labels = test[LABEL]
    gen_labels = res_tree.classify(test.drop(columns=LABEL))

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
        print("This does something ;)")
        # random_forest()
        # gs.ensemble_learning(multi_proc=args.multi)
        gs.ensemble_learning(3000, 15, 10, 200, 0.25, 0.5, [0.5,0.5,0.1], 10, 5, args.multi, True)

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
        print(f"grid search for {arguments.gridsearch}:")
        results = GRID_SEARCH[arguments.gridsearch](arguments.multi)

        results = sorted(results, key = lambda x: x[2], reverse=True) # TODO change to x[1]
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
