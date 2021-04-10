import matplotlib.pyplot as plt
import math as m
import numpy as np


# Python version >= 3.8
def calc_cumbinprob(n=11, weak_p=0.6, strong_p=0.75, w=1):
    '''
    Calculate the cumulative binomial probability of a strong 
    classifier and n-1 weak classifiers.

        Parameters:
            n (int): N classifiers including strong one
            weak_p (float): Probability of correct decision
            strong_p (float): Probability of correct decision
            w (float): Weight of strong classifier

        Returns:
            result (float): Cumalative binomial probability
    '''
    result = 0
    x = m.floor(n / 2) + 1
    
    for k in range(x, n):
        strong_vote = m.comb(n, k) * weak_p**(k-1) * w * strong_p * (1-weak_p)**(n-k)
        weak_vote = m.comb(n, k) * weak_p**k * w * (1-strong_p) * (1-weak_p)**(n-k-1)
        result += (strong_vote + weak_vote)/2 if n - k - 1 >= 0 else strong_vote

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