import numpy as np
import random as r

#artificial 1 
#400 uniform vectors [-1, 1]
#Nc = 2
#w = 0.72, a1=a2=1.49
#2 dim

#Iris
#Nc = 3
#w = 0.72, a1=a2=1.49
#4 dim

def PSO(data, Nc, t_max, w, a1, a2, size):
    # particle is a vector of Nc dimensions
    particles = generate_centroid(data, Nc, 10)

    for t in range(t_max):
        clusters = [[] for i in range(Nc)]

        for i in range(len(particles)):
            for j in range(len(data)):
                ec = [np.linalg.norm(data[j]-c) for c in particles[i]]
                clusters[np.where(ec == np.min(ec))].append(j)

        fit = [fitness(data, clusters, particles[i], Nc) for i in range(len(particles))]
def fitness(data, clusters, particle, Nc):
    J = 0
    for j in range(len(clusters)):
        dist_sum = 0
        for i in clusters[j]:
            temp = np.linalg.norm(data[i]-particle[j])
            dist_sum += temp
        avg_clust = dist_sum / len(clusters[j])
        J += avg_clust
    J /= Nc

    return J


def generate_centroid(data, Nc, amount):
    res = [r.sample(data, k=Nc) for i in range(amount)]
    return res

with open("iris.data", 'r') as f:
    tokens = [[token for token in line.split(',')] for line in f.readlines()]
    print(tokens)
    data_values = np.array([[float(t) for t in token[:4]] for token in tokens])
    data_clusters = [token[4][:-1] for token in tokens]
print(data_values)