import random as r
import matplotlib.pyplot as plt

from bitarray import bitarray


def closer_to_goal(x, goal):
    count = 0
    for i in range(len(x)):
        if x[i] == goal[i]:
            count += 1
    return count


def one_plus_one_ga(l, iter):
    p = 1/l if l > 0 else 1
    x = bitarray(l)
    goal = bitarray(l)
    goal.setall(True)
    fitness_scores = []

    for i in range(1, iter):
        x_m = x.copy()
        txt_file = open("log-file-2.txt", 'a')

        for j, xj in enumerate(x_m):
            if r.random() < p:
                x_m[j] = not xj

        
        if x_m == goal:
            print(f"Succeeded in {i} iterations")
            txt_file.write(f"Succeeded in {i} iterations\n")
            return
        
        '''
        score = closer_to_goal(x_m, goal)
        # print(score)
        
        if score >= closer_to_goal(x, goal):
            fitness_scores.append(score)
            x = x_m
        else:
            fitness_scores.append(closer_to_goal(x, goal))
        '''
        
        x = x_m
        score = closer_to_goal(x_m, goal)
        fitness_scores.append(score)
        
        # print(score)
    
    print(f"Not succeeded in {iter} iterations")
    txt_file.write(f"Not succeeded in {iter} iterations\n")
    return fitness_scores


for i in range(10):
    scores = one_plus_one_ga(100, 1500)
'''
plt.plot([x for x in range(1,1500)], scores)
plt.title("Fitness scores for elapsed iterations")
plt.xlabel("Iteration x")
plt.ylabel("Fitness score")
plt.show()
'''
