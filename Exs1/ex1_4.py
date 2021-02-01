import random as r
import matplotlib.pyplot as plt

from bitarray import bitarray


def closer_to_goal(x, goal):
    count = 0
    for i in range(len(x)):
        if x[i] == goal[i]:
            count += 1
    return count


def one_plus_one_ga(l, iter, mode):
    p = 1/l if l > 0 else 1
    x = bitarray(l)
    goal = bitarray(l)
    goal.setall(True)

    fitness_scores = []
    converged = False

    for i in range(1, iter):
        x_m = x.copy()
        txt_file = open("log-file.txt", 'a')

        for j, xj in enumerate(x_m):
            if r.random() < p:
                x_m[j] = not xj

        if x_m == goal and not converged:
            txt_file.write(f"Succeeded in {i} iterations\n")
            converged = True
        
        if mode:
            score_x_m = closer_to_goal(x_m, goal)
            score_x = closer_to_goal(x, goal)
            if score_x_m >= score_x:
                fitness_scores.append(score_x_m)
                x = x_m
            else:
                fitness_scores.append(score_x)
        else:
            x = x_m
            score = closer_to_goal(x_m, goal)
            fitness_scores.append(score)
    
    if not converged:
        txt_file.write(f"Not succeeded in {iter} iterations\n")

    return fitness_scores


def plot(l, iter):
    scores = one_plus_one_ga(l, iter, True)
    plt.plot([x for x in range(1,iter)], scores)
    plt.title("Fitness scores for elapsed iterations")
    plt.xlabel("Iteration x")
    plt.ylabel("Fitness score")
    plt.show()


for i in range(10):
    plot(100, 1500)