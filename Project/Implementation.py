import argparse

import pandas as pd
import numpy as np
from sklearn import preprocessing

from decision_tree import gp, fitness
from grid_search import AUS_columns, AUS_cat, LABELS, DATA
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
    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=data.columns)


def implement(args):
    if args.test:
        # Kunnen we gebruiken om dingen te testen als we dat willen.
        print("This does nothing")

    elif args.onegp:
        # TODO: multiproc
        print("Generating trees using random data")
        results = []
        n = 8

        for _ in range(n):
            train = DATA.sample(frac=0.8).reset_index(drop=True)
            test = DATA.drop(train.index)

            weights = [0.5, 0.5, 0.]
            res_tree = gp(train.drop(columns=LABELS), AUS_cat, train[LABELS], 8, 400, 0.3, 0.7, 20, 10, weights, True)
            results.append(fitness(res_tree, test.drop(columns=LABELS), weights, test[LABELS]))
        
        results = np.array(results)
        print("mean: ", np.mean(results), "std: ", np.std(results), "for ", n, " runs")

    elif args.gridsearch:
        print(f"first the optimalization for {args.gridsearch}:")
        results = GRID_SEARCH[args.gridsearch](args.multi)

        results = sorted(results, key = lambda x: x[2], reverse=True)
        print("results (sorted on harmonic mean of TPR and TNR) found:")
        print(results)

    else:
        gs.perform_gp(100, 8, 8, 400, 0.3, 0.7, 20, 10, [0.5, 0.5, 0.], args.multi)


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
