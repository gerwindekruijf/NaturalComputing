"""
This contains the data sampling and normalization
"""
import math

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


# AUS_columns = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15']
# CATS = dict(zip(AUS_columns, [1,0,0,1,1,1,0,1,1,0,1,1,0,0,1]))
# with open(r"australian/australian.dat", 'r') as f:
#     DATA = pd.DataFrame(
#         [
#             [(float(token) if '−' not in token else -float(token[1:])) for token in line.split()]
#             for line in f.readlines()
#             ],
#         columns=AUS_columns)
# LABEL = 'A15'

DATA = pd.read_csv("creditcard.csv").drop(columns='Time')
CATS = dict(zip(list(DATA.columns.values), [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]))
LABEL = 'Class'


def norm(data):
    """
    Normalize data

    :param data: data
    :type data: dataframe
    :return: data
    :rtype: dataframe
    """
    values = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    values_scaled = min_max_scaler.fit_transform(values)
    return pd.DataFrame(values_scaled, columns=data.columns)


def sample_data(data, samples, size=None, use_all_labels=False):
    """
    Generate samples from the data

    :param data: data
    :type data: dataframe
    :param samples: number of samples
    :type samples: int
    :param size: maximum size of the samples, defaults to None
    :type size: int, optional
    :param use_all_labels: use all true labels, defaults to False
    :type use_all_labels: bool, optional
    :return: list of samples
    :rtype: list
    """
    grouped = data.groupby(LABEL)
    data_per_label = [grouped.get_group(i) for i in data[LABEL].unique()]

    min_sizes = [np.inf] * len(data_per_label)
    if size is not None:
        min_sizes = [math.ceil(len(data[data[LABEL]==key]) * size / len(data)) for key in data[LABEL].unique()]
    sample_sizes = [min(min_sizes[i], len(dat) // samples) for i, dat in enumerate(data_per_label)]

    if use_all_labels:
        sample_sizes[1] = len(data_per_label[1]) // samples

    result = []
    for _ in range(samples):
        sample = data_per_label[0].sample(sample_sizes[0])
        data_per_label[0] = data_per_label[0].drop(sample.index)

        for j in range(1, len(data_per_label)):
            dat_sample = data_per_label[j].sample(sample_sizes[j])
            data_per_label[j] = data_per_label[j].drop(dat_sample.index)

            sample = sample.append(dat_sample)
        result.append(sample)

    return [s.reset_index(drop=True) for s in result]


def random_forest():
    """
    Generate results based on sklearn randomforest
    """
    train, test = train_test_data(DATA)

    train_x = train.drop(columns=LABEL).to_numpy()
    train_y = train[LABEL].to_numpy()

    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)

    test_x = test.drop(columns=LABEL).to_numpy()
    test_y = test[LABEL].to_numpy()
    predictions = clf.predict(test_x)

    equal = predictions[predictions == test_y]
    result = []
    for i in range(2):
        result.append(len(equal[equal == i])/len(test_y[test_y==i]))

    print(f"FINAL SCORE: TPR: {result[1]} and TNR: {result[0]} out of {len(test_y)} labels used")


def train_test_data(data):
    """
    Generate train and test data

    :param data: data
    :type data: dataframe
    :return: train and test data
    :rtype: tuple
    """
    grouped = data.groupby(LABEL)
    data_per_label = [grouped.get_group(i) for i in data[LABEL].unique()]

    train = data_per_label[0].sample(frac=0.8)
    data_per_label[0] = data_per_label[0].drop(train.index)

    for j in range(1, len(data_per_label)):
        dat_sample = data_per_label[j].sample(frac=0.8)
        data_per_label[j] = data_per_label[j].drop(dat_sample.index)

        train = train.append(dat_sample)

    return train.reset_index(drop=True), pd.concat(data_per_label).reset_index(drop=True)
