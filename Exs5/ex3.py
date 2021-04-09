import matplotlib.pyplot as plt
import math as m
import numpy as np


# Python version >= 3.8
def calc_cumbinprob(n=11, weak_p=0.6, w=1, strong_p=0.75):
    result = 0
    x = m.floor(n / 2) + 1

    for k in range(x, n):
        yes = m.comb(n, k) * weak_p**(k-1) * w * strong_p * (1-weak_p)**(n-k)
        no = m.comb(n, k) * weak_p**k * w * (1-strong_p) * (1-weak_p)**(n-k-1) if n - k - 1 >= 0 else 0
        result += (yes + no)/2

    return result


def AdaBoost_M1(M, N):
    w = 1/N
    
    for m in range(1, M):   
        # a do function
        # How to compute Gm(x)?

        # b
        # How to compute (yi /= Gm(xi))?
        num = np.sum([w * np.identity * 1 for i in range(1,N)])
        denom = np.sum([w for i in range(1,N)])
        err_m = np.divide(num, denom)

        # c
        a_m = np.log((1-err_m)/err_m)

        # d
        w = w * np.exp(a_m * np.identity(N)*1)
    
    return np.sign(np.sum([a_m * 1 for i in range(1,M)]))


# Ex. 3A
print(calc_cumbinprob())

# Ex. 3B
print(calc_cumbinprob(w=1.1))
print(calc_cumbinprob(w=1.2))
# Maybe weight as power instead of multiplication
# or some better method