import matplotlib.pyplot as plt
import math as m
import numpy as np


# Python version >= 3.8
def calc_cumbinprob(n, p):
    result = 0
    x = m.floor(n / 2) + 1

    for k in range(x, n):
        result += m.comb(n, k) * p**k * (1-p)**(n-k)

    return result


def plot():
    competence_levels = np.linspace(0.1,0.9,17)
    jury = [x for x in range(3,103)]
    results = []
    for p in competence_levels:
        for i in jury:
            results.append(calc_cumbinprob(i,p))
        plt.plot(jury, results, label = str(round(p,2)))
        results = []    
    plt.legend()

    plt.title("Probability of a correct decision")
    plt.xlabel("Jury size")
    plt.ylabel("Probability")
    plt.show()


plot()