import random as r
from bitarray import bitarray


def one_plus_one_ga(l, iter):
    p = 1/l if l > 0 else 1
    x = bitarray(l)
    goal = bitarray(l)
    goal.setall(True)

    for i in range(1, iter):
        x_m = x
        print(x_m)

        for j, xj in enumerate(x_m):
            if r.random() < p:
                x_m[j] = ~xj
        
        if x_m == goal:
            print(x_m)
            print(f"Succeeded in {i} iterations")
            return
        
        if x_m >= x:
            x = x_m
    
    print(f"Not succeeded in {iter} iterations")


one_plus_one_ga(5, 100)

