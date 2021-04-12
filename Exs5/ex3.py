import matplotlib.pyplot as plt
import math as m
import numpy as np


# # Python version >= 3.8
# def calc_cumbinprob(n=11, weak_p=0.6, strong_p=0.75, w=1):
#     '''
#     Calculate the cumulative binomial probability of a strong 
#     classifier and n-1 weak classifiers.

#         Parameters:
#             n (int): N classifiers including strong one
#             weak_p (float): Probability of correct decision
#             strong_p (float): Probability of correct decision
#             w (float): Weight of strong classifier

#         Returns:
#             result (float): Cumalative binomial probability
#     '''
#     result = 0
#     x = m.floor(n / 2) + 1
    
#     for k in range(x, n):
#         strong_vote = m.comb(n, k) * weak_p**(k-1) * w * strong_p * (1-weak_p)**(n-k)
#         weak_vote = m.comb(n, k) * weak_p**k * w * (1-strong_p) * (1-weak_p)**(n-k-1)
#         result += (strong_vote + weak_vote)/2 if n - k - 1 >= 0 else strong_vote

#     return result


def calc_cumbinprob_ours(n=11, weak_p=0.6, strong_p=0.75, w=1):
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

    # We demand the weigthed votes to still be equal to the initial amount of votes, this gives that the (decreased) weight of the weak classifiers becomes:

    weight_weak = (11 - w)/10

    # We then only need to consider the amount of weak classifiers needed to still get a majority vote given that we know how the strong classifier votes. This can be determined to be:

    needed_w_strong = m.ceil(((11/2) - w)/(weight_weak))
    print("w_strong: ", needed_w_strong)
    needed_o_strong = m.ceil((11/2)/ (weight_weak)) 
    print("o_strong : ", needed_o_strong)
    strong_vote = 0
    if (needed_w_strong > -1 and needed_o_strong < 11):
        for i in range(needed_w_strong, 10):
            print(i)
            strong_vote += m.comb(n-1, i) * weak_p**i * (1-weak_p)**(n-1-i) * strong_p

        weak_vote = 0
        for i in range(needed_o_strong, 10):
            weak_vote += m.comb(n-1, i) * weak_p**i * (1-weak_p)**(n-1-i) * (1 - strong_p)

        result = strong_vote + weak_vote 
        return result
    else:
        return strong_p

# Ex. 3A
print(calc_cumbinprob_ours())

# Ex. 3B
# def plot():
#     weights = np.linspace(1,7, 13)
#     print(weights)
#     results = []
#     for i in weights:
#         results.append(calc_cumbinprob_ours(w = i))

#     print(results)
#     plt.plot(weights, results)
#     plt.legend()

#     plt.title("Probability of a correct decision")
#     plt.xlabel("Weight of strong classifier")
#     plt.ylabel("Probability")
#     plt.show()


# plot()

# Ex. 3C
errors = np.array([0.25, 0.4])

alphas = np.array([m.log2((1-x)/x) for x in errors])

# Ex. 3D
def weight_vs_error():
    alpha = []
    errors = []
    for error in range(5, 100, 5):
        e = error/100
        errors.append(e)
        alpha.append(m.log2((1-e)/e))
    plt.plot(errors, alpha, label = "weight")
    plt.axvline(0.5, linestyle =  "--",  label = "0.5 divider", color = "red")
    plt.xlabel("error rate")
    plt.ylabel("Weight by AdaBoost")
    plt.title("Adaboost for different base learners")
    plt.legend()
    plt.show()




weight_vs_error()
# Maybe weight as power instead of multiplication
# or some better method