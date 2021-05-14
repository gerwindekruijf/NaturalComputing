# import collections
# import matplotlib.pyplot as plt
import pandas as pd
import random as r
import numpy as np
import argparse
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from Decision_tree import gp, fitness

THREADS = cpu_count()

#TODO: normalize the data
AUS_columns = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15']
AUS_cat = dict(zip(AUS_columns, [1,0,0,1,1,1,0,1,1,0,1,1,0,0,1]))
with open(r"australian/australian.dat", 'r') as f:
    DATA = pd.DataFrame([[(float(token) if '−' not in token else -float(token[1:])) for token in line.split()] for line in f.readlines()], columns=AUS_columns)

# LABELS = DATA['A15']
# DATA = DATA.drop(columns='A15')
LABELS = 'A15'

def norm(data):
    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=data.columns)

def perform_gp(sample_size = 100 , learners = 8,  generations = 8, pop_size = 400, mut_rate = 0.1, cros_rate = 0.9, max_depth = 20, cros_md = 10, fit_weights = [float(1/2), float(1/2), float(0)]):
    dfs = [DATA.sample(sample_size).reset_index(drop=True) for _ in range(learners)]
    
    params = [(data.drop(columns=LABELS), AUS_cat, data[LABELS], generations, pop_size, mut_rate, cros_rate, max_depth, cros_md, fit_weights, False) for data in dfs]
    res = []
    if args.multi:
        with Pool(THREADS) as p:
            res = list(tqdm(p.starmap(gp, params), total=len(params)))
    else:
        for d in tqdm(params, total=len(params)):
            res.append(gp(*d))

    labels = DATA[LABELS].to_numpy()
    res_class = np.array([tree.classify(DATA) for tree in res])
    ensemble = np.array(stats.mode(res_class))[0]

    equal = ensemble[ensemble == labels]
    TPR = len(equal[equal == 1])/len(labels[labels == 1])
    TNR = len(equal[equal == 0])/len(labels[labels == 0])

    #correct = (ensemble == labels.to_numpy()).sum()
    
    print(f"FINAL SCORE: TPR: {TPR} and TNR: {TNR} out of {len(labels)} labels used")

    return TPR, TNR, ensemble

def implement(args):
    if args.test:
        #Kunnen we gebruiken om dingen te testen als we dat willen.
    elif args.onegp: ## Zie niet helemaal waar dit fout gaat, split werkt zoals je wil. 
        results = []
        n = 8

        for i in range(n):
            train = DATA.sample(frac = 0.8, random_state=200)
            test = DATA.drop(train.index)

            res_tree = gp(train.drop(columns = LABELS), AUS_cat, train[LABELS], 8, 400, 0.3, 0.7, 20, 10, [float(1/2), float(1/2), float(0)], True)
            results.append(fitness(res_tree, test.drop(columns=LABELS), test[LABELS]))
        
        results = np.array(results)
        print("mean: ", np.mean(results), "std: ", np.std(results), "for ", n, " runs")

    elif args.gridsearch:
        
        print("first the optimalization for the fitness weights:")
        results = []
        i = 5
        for j in range(i - 2, i + 2,1):
            k = float(0.1 * i)

            weights = [float(i/10),float(j/10),float(k/10)]
            print(f"Performing GP + ensemble for normal values and weights: {float(i/10)} TPR + {float(j/10)} TNR and {float(k/10)} depth")
            TPR , TNR, _ = perform_gp(fit_weights = weights)
            hm = 2*TPR*TNR/(TPR + TNR)
            print(f"results are: TPR: {TPR} and TNR: {TNR}")
            results.append((TPR,TNR, hm, float(i/10),float(j/10),float(k/10)))

            k = float(0.2 * i)
            weights = [float(i/10),float(j/10),float(k/10)]
            print(f"Performing GP + ensemble for normal values and weights: {float(i/10)} TPR + {float(j/10)} TNR and {float(k/10)} depth")
            TPR , TNR, _ = perform_gp(fit_weights = weights)
            hm = 2*TPR*TNR/(TPR + TNR)
            print(f"results are: TPR: {TPR} and TNR: {TNR}")
            results.append((TPR,TNR, hm, float(i/10),float(j/10),float(k/10)))
            
        results = sorted(results, key = lambda x: x[2], reverse=True)
        print(f"results (sorted on harmonic mean of TPR and TNR) found: {results}")

        # TODO:optimalizatie voor: 1. Max_depth samen met cross_depth
        #                           2. generaties samen met learners en pop
        #                           3. cros_rate en mut_rate
        #                           4. learners (optimale waarde uit 2. iig in gebruiken) en sample_size
        #         (runnen proberen onder het uur te houden, elke run is ongeveer 1 minuut) 
                


    else:            
        dfs = [DATA.sample(100).reset_index(drop=True) for _ in range(8)]
        
        params = [(data.drop(columns=LABELS), AUS_cat, data[LABELS], 8, 400, 0.3, 0.7, 20, 10, [float(1/2), float(1/2), float(0)], True) for data in dfs]
        res = []
        if args.multi:
            with Pool(THREADS) as p:
                res = list(tqdm(p.starmap(gp, params), total=len(params)))
        else:
            for d in tqdm(params, total=len(params)):
                res.append(gp(*d))

        labels = DATA[LABELS].to_numpy()
        res_class = np.array([tree.classify(DATA) for tree in res])
        ensemble = np.array(stats.mode(res_class))[0]

        equal = ensemble[ensemble == labels]
        TPR = len(equal[equal == 1])/len(labels[labels == 1])
        TNR = len(equal[equal == 0])/len(labels[labels == 0])

        #correct = (ensemble == labels.to_numpy()).sum()
        
        print(f"FINAL SCORE: TPR: {TPR} and TNR: {TNR} out of {len(labels)} labels used")


if __name__ == '__main__':
    DATA = norm(DATA)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true",
                        help="testing possibility")
    parser.add_argument("-m","--multi", action="store_true",
                        help="multiprocessing for room simulation")
    parser.add_argument("-og", "--onegp", action="store_true",
                        help="performs no ensemble method + statistical checking of value")
    parser.add_argument("-gs", "--gridsearch", action = "store_true",
                        help="performs a grid search for a hyperparameter optimalization procedure")
    args = parser.parse_args()
    implement(args)
