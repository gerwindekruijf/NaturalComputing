# import collections
# import matplotlib.pyplot as plt
import pandas as pd
import random as r
import numpy as np
import argparse
from scipy import stats
from sklearn import preprocessing
from tqdm import tqdm
from multiprocessing import Pool

from Decision_tree import gp, fitness

THREADS = 8

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

def implement(args):
    dfs = [DATA.sample(100).reset_index(drop=True) for _ in range(8)]
    
    params = [(data.drop(columns=LABELS), AUS_cat, data[LABELS], 8, 400, 0.3, 0.7, 20, None) for data in dfs]
    res = []
    if args.multi:
        with Pool(THREADS) as p:
            res = list(tqdm(p.starmap(gp, params), total=len(params)))
    else:
        for d in tqdm(params, total=len(params)):
            res.append(gp(*d))

    labels = DATA[LABELS]
    res_class = np.array([tree.classify(DATA) for tree in res])
    ensemble = np.array(stats.mode(res_class))[0]
    correct = (ensemble == labels.to_numpy()).sum()
    
    print(f"FINAL SCORE = {correct/len(labels)} out of {len(labels)}")


if __name__ == '__main__':
    DATA = norm(DATA)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--multi", action="store_true",
                        help="multiprocessing for room simulation")
    args = parser.parse_args()
    implement(args)
