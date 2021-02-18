import numpy as np
import random as r
from tqdm import tqdm
from sklearn.cluster import KMeans


def PSO(data, Nc, t_max, p=10):
    # particle is a vector of Nc dimensions
    particles = generate_centroid(data, Nc, p)
    velocities = np.zeros_like(particles)

    local_best = [particles[i] for i in range(p)]
    local_best_fitness = [np.inf for i in range(p)]
    
    global_best = particles[0]
    global_best_fitness = np.inf

    for t in tqdm(range(t_max)):
        clusters = [[] for i in range(Nc)]

        for i in range(len(particles)):
            for j in range(len(data)):
                ec = [np.linalg.norm(data[j]-c) for c in particles[i]]
                clusters[np.where(ec == np.amin(ec))[0][0]].append(j)

        fit = [fitness(data, clusters, particles[i], Nc) for i in range(len(particles))]
        
        local_best = [particles[i] if fit[i] < local_best_fitness[i] else local_best[i] for i in range(p)]
        local_best_fitness = [fit[i] if fit[i] < local_best_fitness[i] else local_best_fitness[i] for i in range(p)]

        global_best_fitness = np.min(local_best_fitness)
        global_best = local_best[np.where(local_best_fitness==global_best_fitness)[0][0]]

        velocities = vel_update(particles, velocities, local_best, global_best)
        particles = move(particles, velocities)

        #print("I:",t,"globalbest",global_best_fitness)
        # print(f'I:{t} \t best:{global_best_fitness}')
    return global_best_fitness
        


def vel_update(particles, velocities, local_best, global_best, w = 0.72, a1=1.49, a2=1.49):
    Nc, d = particles[0].shape
    p = len(particles)
    velocities_new = []

    for i in range(p):
        vel = []
        for j in range(Nc):
            r1 = np.random.uniform(0, 1, d)
            r2 = np.random.uniform(0, 1, d)
            vel.append(w*velocities[i][j] + a1*r1*(local_best[i][j]-particles[i][j]) + a1*r1*(global_best[j]-particles[i][j]))
        velocities_new.append(np.array(vel))
    return velocities_new


def move(particles, velocities):
    return [particles[i]+velocities[i] for i in range(len(particles))]


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
    res = [np.array(r.sample(data, k=Nc)) for i in range(amount)]
    return res


def kmeans(data, Nc):
    km = KMeans(n_clusters=Nc).fit(data)
    clusters = [[] for i in range(Nc)]

    for i in range(len(km.labels_)):
        clusters[km.labels_[i]].append(i)

    return fitness(data, clusters, km.cluster_centers_, Nc)


with open("iris.data", 'r') as f:
    tokens = [[token for token in line.split(',')] for line in f.readlines()]
    data_values = [np.array([float(t) for t in token[:4]]) for token in tokens]
    data_clusters = [token[4][:-1] for token in tokens]

print("Iris")
print(PSO(data_values, 3, 1000))
print(kmeans(data_values, 3))

data_values = [np.random.uniform(-1, 1, 2) for i in range(400)]

print("Artificial")
print(PSO(data_values, 2, 1000))
print(kmeans(data_values, 2))

#artificial 1 
#400 uniform vectors [-1, 1]
#Nc = 2
#w = 0.72, a1=a2=1.49
#2 dim

#Iris
#Nc = 3
#w = 0.72, a1=a2=1.49
#4 dim