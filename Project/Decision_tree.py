import numpy as np
import pandas as pd
import random as r
from tqdm import tqdm
import sys

# ROOT is the first node of a tree
OPERATORS = ['leaf', '<', '>', '==', 'ROOT']

class Tree:
    """
    A Tree contains a label, operator and value
    """
    def __init__(self, label=None, children=[], parent=(None,0), operator=None, value=None, column=None, tree=None):
        if tree is not None:
            self.operator = tree.operator
            self.column = tree.column
            self.label = tree.label
            self.value = tree.value
            self.parent = parent            # new parent to avoid reference errors

            self.children = []
            i = 0
            for child in tree.children:
                self.children.append(Tree(tree=child,parent=(self,i)))

        else:        
            self.column = column     # Data column for this node
            self.operator = operator # Operator
            self.value = value       # Data comparison value
            self.label = label       # classification label
            self.parent = parent     # Tuple containing parent and childindex for efficiency (no eq function needed for crossover?)

            self.children = []       # Subtrees TODO: maak tuple, zodat je niet meer kinderen kan
            i = 0
            for child in children:
                self.children.append(Tree(tree=child,parent=(self,i)))

    def __str__(self):
        if self.operator == OPERATORS[0]:
            return str(self.label)
        if self.operator == OPERATORS[4]:
            return "Tree: " + str(self.children[0])

        res = str(self.column) + " " + str(self.operator) + " " + str(self.value) + " "
        c = [str(child) for child in self.children]
        return res + f"[{', '.join(c)}]"
    
    def generate(self, data, cat, labels, depth, parent):
        self.parent = parent
        if self not in self.parent[0].children:
            self.parent[0].children.append(self)
        self.operator = r.choice(OPERATORS[0:4])

        if depth == 0 or self.operator == OPERATORS[0]:
            self.operator = OPERATORS[0]
            self.label = r.choice(labels) # Select random classification label
        else:
            self.column = r.choice(data.columns) # Select random column
            if cat[self.column]:
                self.operator = OPERATORS[1]
                self.value = r.choice(data[self.column])
            else:
                self.value = r.random() # Random value from dataset?, Seems more logical
            # We could split the data to get more accurate representation, big no from me though
            # Dit is een GP keuze, geen splitting voor data, maar als het slecht gaat optie om naar te kijken
            for i in range(2):
                child = Tree()
                child.generate(data, cat, labels, depth - 1, (self, i))

        return self

    def classify(self, data):
        # This is a leaf, classify the data
        if self.operator == OPERATORS[0]:
            return np.array([self.label for _ in data.iterrows()])

        if self.operator == OPERATORS[4]:
            return self.children[0].classify(data)

        # Sort data in True or False for this statement
        s = []
        if self.operator == OPERATORS[1]: # ==
            s = (data[self.column] == self.value)
        elif self.operator == OPERATORS[2]: # <
            s = (data[self.column] < self.value)
        elif self.operator == OPERATORS[3]: # >
            s = (data[self.column] > self.value)
        
        s = np.array(s)

        # Classify data in subtrees
        d_t = self.children[0].classify(data[s])
        d_f = self.children[1].classify(data[~s])

        # Build result by combining the results
        t, f = 0, 0
        result = np.array([])
        for i in s:
            if i:
                result = np.append(result, d_t[t])
                t += 1
            else:
                result = np.append(result, d_f[f])
                f += 1

        return result

    def choose(self, index):
        # Return a Tree given the index
        if self.operator == OPERATORS[4]:
            return self.children[0].choose(index)

        if index == 0 or self.operator == OPERATORS[0]:
            return (self, index-1)
    
        res, index = self.children[0].choose(index-1)
        if index == -1:
            return (res, index)

        return self.children[1].choose(index)

    def depth(self):
        # Return the maximum depth of the tree
        d = np.max([c.depth() for c in self.children], initial=0)
        return d if self.operator == OPERATORS[4] else d + 1
    
    def complexity(self):
        # Return the number of nodes of the tree
        c = np.sum([c.complexity() for c in self.children], initial=0)
        return c if self.operator == OPERATORS[4] else c + 1


def generate_trees(data, cat, labels, number, max_initial_depth):
    # generate a number of trees, with an initial maximum depth
    classifiers = list(set(labels))
    return [
        Tree()
        .generate(data, cat, classifiers, max_initial_depth, (Tree(operator=OPERATORS[4]),0))
        .parent[0] for _ in range(number)
        ]
    

def fitness(tree, data, labels):
    # Number of equal elements
    # TODO FITNESS SHOULD TAKE DEPTH INTO ACCOUNT
    gen_labels = tree.classify(data)
    return (labels == gen_labels).sum()


def crossover(tree1, tree2, max_depth=None):
    # Select Trees
    r1, r2 = Tree(tree=tree1, parent=(Tree(operator=OPERATORS[4]),0)), Tree(tree=tree2, parent=(Tree(operator=OPERATORS[4]),0))

    nodes1 = r1.complexity()
    n1 = r.randrange(nodes1)
    c_tree1, _ = r1.choose(n1)
    # Tree should be within max_depth
    # TODO remove ROOT? Does make crossover easier, but could be removed
    while c_tree1.parent[0] is None or max_depth is not None and c_tree1.depth() <= max_depth:
        n1 = r.randrange(nodes1)
        c_tree1, _ = r1.choose(n1)

    nodes2 = r2.complexity()
    n2 = r.randrange(nodes2)
    c_tree2, _ = r2.choose(n2)
    while c_tree2.parent[0] is None or max_depth is not None and c_tree2.depth() <= max_depth:
        n2 = r.randrange(nodes2)
        c_tree2, _ = r2.choose(n2)

    # Switch trees
    temp_parent1 = c_tree1.parent
    temp_parent2 = c_tree2.parent

    c_tree1.parent = c_tree2.parent
    c_tree2.parent = temp_parent1

    temp_parent1[0].children[temp_parent1[1]] = c_tree2
    temp_parent2[0].children[temp_parent1[1]] = c_tree1

    return r1, r2


def mutation(data, cat, labels, tree):
    # changed mutation to random subtree
    rt = Tree(tree=tree, parent=(Tree(operator=OPERATORS[4]),0))
    nodes = rt.complexity()
    n = r.randrange(nodes)
    c_tree, _ = rt.choose(n)
    
    classifiers = list(set(labels))
    c_tree.generate(data, cat, classifiers, c_tree.depth(), c_tree.parent)
    return rt 


def gp(data, cat, labels, generations, pop_size=1000, mutation_rate=0, cross_rate=0.7, max_depth=None, cross_max_depth=None):
    """
    GP algorithm for decision trees based on dataframe

    data: NORMALIZED (because of random) dataframe containing the data FOR THIS GP
    cat: dataframe containing booleans if label is categorical
    labels: True results for each row in data
    """
    seed = r.randrange(sys.maxsize)
    rng = r.seed(seed)
    print("Seed was:", seed)

    # Initial population
    print("Creating initial population...")
    parents = generate_trees(data, cat, labels, pop_size, 3)

    for i in tqdm(range(generations)):
        print("Mutating phase...")
        parents2 = []
        for parent in parents:
            if r.random() < mutation_rate:
                # Mutation
                p1 = mutation(data, cat, labels, parent)
                parents2.append(p1)

        parents.extend(parents2)

        print("Filling phase...")
        parents2 = []
        while len(parents2) < pop_size*cross_rate:
            # Selection (Tournament)
            [pp1, pp2, pp3, pp4] = r.sample(list(range(len(parents))), k=4)

            p1 = parents[pp1] if fitness(parents[pp1], data, labels) < fitness(parents[pp2], data, labels) else parents[pp2]
            p2 = parents[pp3] if fitness(parents[pp3], data, labels) < fitness(parents[pp4], data, labels) else parents[pp4]

            # Crossover
            c1, c2 = crossover(p1, p2, cross_max_depth)
            if max_depth is None or c1.depth() < max_depth:
                parents2.append(c1)
            if max_depth is None or c2.depth() < max_depth:
                parents2.append(c2)

        parents.extend(parents2)


        print("Selecting next generation...")
        s = sorted(parents, key=lambda c: fitness(c, data, labels), reverse=True)

        print(f"Best fitness {i}: {fitness(s[0], data, labels)}")
        print(s[0])

        parents = s[:100] # Next generation
    
    return parents[0]
# TODO: multithreading voor langzame delen
# TODO: Clean up code
