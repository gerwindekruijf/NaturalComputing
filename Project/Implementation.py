# import collections
# import matplotlib.pyplot as plt
import pandas as pd
import random as r
import numpy as np
from tqdm import tqdm

from Decision_tree import gp, fitness

AUS_columns = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15']
AUS_cat = dict(zip(AUS_columns, [1,0,0,1,1,1,0,1,1,0,1,1,0,0,1]))
with open(r"australian/australian.dat", 'r') as f:
    DATA = pd.DataFrame([[(float(token) if '−' not in token else -float(token[1:])) for token in line.split()] for line in f.readlines()], columns=AUS_columns)

LABELS = DATA['A15']
DATA = DATA.drop(columns='A15')

def implement():
    x=gp(DATA, AUS_cat, LABELS, 8, pop_size=400, mutation_rate=0.3, cross_rate=0.7, max_depth=20, cross_max_depth=None)
    #TODO COLUMNS SHOULD CONTAIN NAMES FOR DF TO WORK

# scores_trees = [fitness(tree[0]) for tree in trees]
# depths_trees = [tree[1] for tree in trees]

# print(trees[-1])

# plt.plot([x for x in range(1, len(trees) + 1)], scores_trees, color="green", label="tree score")
# plt.plot([x for x in range(1, len(trees) + 1)], depths_trees, color="red", label="tree depth")
# plt.title("Fitness scores for elapsed iterations")
# plt.legend()
# plt.xlabel("Iteration x")
# plt.ylabel("Fitness score")
# plt.show()

implement()