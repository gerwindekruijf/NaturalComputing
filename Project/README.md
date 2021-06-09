# Learning a random forest with Genetic Programming
### (Hyper)parameters within the implemenation:
Here the (hyper)parameters (and their default setting) will be explained. The first of these are the (hyper)parameters of the genetic programming procedure:
- generations: (default = 8) This parameter determines the amount of generations used within the genetic programming procedure. By a generation the threes that survive after a procedure of mutation, cross-over and subsequently selecting the best rated trees is meant.
- pop_size: (default = 400) This parameter determines the amount of trees within each generation.
- mutation_rate: (default = 0.3) This parameter determines the percantage of trees within the population, upon which mutation is performed and as such a new tree is generated.
- cross_rate: (default = 0.7) This parameter determines the amount of new trees that are generated within that generation as percentage of the original population of said generation.
- fit_weights: (default = [0.5,0.5,0]) This parameter sets the weights of TPR, TNR and the depth penalty respictively for the rating function used within the genetic programming procedure. The weights need not be normalized.
- max_depth: (default = 20) This parameter determines the maximum depth that a tree can reach. All possible ways of tree generation (intial, mutation and cross-over) are constrained by this parameter.
- cross_max_depth: (default = 10) This parameter determines the maximum depth of the subtrees that are cross overed within the cross-over operation.
 
The next (hyper)parameters are the parameters that determine the ensemble learning of the random forest implementation:
- sample_size: (default = 100) the size of the sample of the training set that is given to each learner to learn on. If sample_size is bigger than the sample size that would be used if the dataset was evenly distributed over the leaners, the latter sample size will be used.
- learners: (default = 8) This parameter determines the amount of learners that are used. 

The other parameters and arguments given within usage determine the workings of the implementation:
- use_all_labels: (default = False) This parameter determines whether or not all the positive classes are evenly distributed over the samples given to the learners
- -t/--test: If used gives the possibility to test, within implementation.py specified, parts of the implementation.
- -m/--multi: If used turns on multicore learning of the learners.
- -og/--onegp: If used turns off the ensemble learning method and performs only genetic programming to learn a decision tree for a 80/20 split of the training data and subsequently performs statistical analysis.
- -gs/--gridsearch: If used performs a grid search optimization for the, to be included specified part, of the implementation. Specified parts are: fitness_weights, depth, population, rates and learners.
### Examples
Some example lines to use the implemenation are given:

```cmd
  implementation.py --multi
```

```cmd
  implemenation.py -m -gs population
```

In order to run the implemenation using specified hyperparameters for the genetic programming procedure one should either run the first example, or:
```cmd
  implementation.py
``` 
And (for both options) change the parameters at line 97 of implementation.py
