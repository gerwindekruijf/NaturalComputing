import math
import random as r
import matplotlib.pyplot as plt



def fitness(locations, order):
    result = 0

    loc1 = locations[order[0]]
    for i in order[1:]:
        loc2 = locations[i]
        dist = math.sqrt( (loc1[1]-loc2[1])**2 + (loc1[0]-loc2[0])**2 )
        result += dist
        loc1 = loc2
    return result


def crossover(p1, p2, N):
    cut = int(N / 4)

    o1, o2 = [], []
    for i in range(N):
        if i >= (cut) and i < (3*cut+1):
            o1.append(p1[i])
            o2.append(p2[i])
        else:
            o1.append(0)
            o2.append(0)

    for i in range(N):
        if i >= (cut) and i < (3*cut+1):
            continue

        for x in (p2[i:] + p2[:i]):
            if x in o1:
                continue
            o1[i] = x
            break

        for x in (p1[(i+1):] + p1[:(i+1)]):
            if x in o2:
                continue
            o2[i] = x
            break

    return o1, o2


def mutation(order, N):
    r.seed()
    [l1, l2] = r.sample(list(range(N)), k=2)
    order[l1], order[l2] = order[l2], order[l1]
    return order


def best_child(children, locations):
    s = sorted(children, key=lambda c: fitness(locations, c))
    return s[0]


def simple_EA(iterations, parents_size = 26):
    locations = []
    with open("file-tsp.txt", 'r') as f:
        locations = [[float(token) for token in line.split()] for line in f.readlines()]

    N = len(locations)
    chance = 1/parents_size
    initial_order = list(range(N))

    # initial population
    parents = [r.sample(initial_order, len(initial_order)) for i in range(parents_size)]

    result = []

    for i in range(iterations):
        children = []
        for j in range(int(len(parents) / 2)):
            # selection
            [pp1, pp2, pp3, pp4] = r.sample(list(range(parents_size)), k=4)

            p1 = parents[pp1] if fitness(locations, parents[pp1]) < fitness(locations, parents[pp2]) else parents[pp2]
            p2 = parents[pp3] if fitness(locations, parents[pp3]) < fitness(locations, parents[pp4]) else parents[pp4]

            # crossover
            o1, o2 = crossover(p1, p2, N)
            children.append(o1)
            children.append(o2)

        # mutation
        r.seed()
        children = [mutation(c, N) if r.random() < chance else c for c in children]

        # store best child
        result.append(best_child(children, locations))

        parents = children

    return [fitness(locations, r) for r in result]


scores = simple_EA(10000)

plt.plot([x for x in range(1, len(scores) + 1)], scores)
plt.title("Fitness scores for elapsed iterations")
plt.xlabel("Iteration x")
plt.ylabel("Fitness score")
plt.show()