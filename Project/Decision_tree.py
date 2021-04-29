import numpy as np
import random as r
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm

OPERATORS = ['leaf', '==', '<', '>']

class Tree:
    """
    A Tree contains a label, operator and value
    """
    def __init__(self, label=None, children=[], parent=(None,0), operator=None, value=None):
        self.label = label       # Data label, or classification label TODO pas aan zodat het duidelijker is wanneer het gaat om column label of om label
        self.operator = operator # Operator
        self.value = value       # Data comparison value
        self.parent = parent     # Tuple containing parent and childindex for efficiency (no eq function needed for crossover?)
        self.children = []       # Subtrees TODO: maak tuple, zodat je niet meer kinderen kan
        if children is not None:
            for child in children:
                self.children.append(child)

        def __init__(self, pTree):
        self.label = pTree.label       # Data label, or classification label TODO pas aan zodat het duidelijker is wanneer het gaat om column label of om label
        self.operator = pTree.operator # Operator
        self.value = pTree.value       # Data comparison value
        self.parent = pTree.parent     # Tuple containing parent and childindex for efficiency (no eq function needed for crossover?)
        self.children = []             # Subtrees TODO: maak tuple, zodat je niet meer kinderen kan
        if children is not None:
            for child in pTree.children:
                self.children.append(Tree(pTree.child))


    def generate(self, data, cat, labels, depth, parent):
        self.parent = parent

        if depth == 0:
            self.operator == OPERATORS[0]:
            self.label = r.choice(labels) # Select random classification
        else
            self.label = r.choice(data.shape[1]) # Select random column
            if cat[self.label]:
                self.operator = OPERATORS[1]
                self.value = r.choice(data[self.label])
            else:
                self.operator = r.choice(OPERATORS[2:4])
                self.value = r.random()
                # We could split the data to get more accurate representation, big no from me though
                # Dit is een GP keuze, geen splitting voor data, maar als het slecht gaat optie om naar te kijken
                for i in range(2):
                    child = Tree()
                    self.children.append(child.generate(data, cat, labels, depth - 1, (self, i)))

        return self

    def classify(self, data):
        # This is a leaf, classify the data
        if self.operator == OPERATORS[0]:
            return [self.label for _ in data]

        # Sort data in True or False for this statement
        s = []
        if self.operator == OPERATORS[1]: # ==
            s = (data[self.label] == self.value)
        elif self.operator == OPERATORS[2]: # <
            s = (data[self.label] < self.value)
        elif self.operator == OPERATORS[3]: # >
            s = (data[self.label] > self.value)
        
        s = np.array(s)

        # Classify data in subtrees
        d_t = self.children[0].classify(data[s])
        d_f = self.children[1].classify(data[~s])

        # Build result by combining the results
        t, f = 0, 0
        result = np.array()
        for i in s:
            if i:
                result.append(d_t[t])
                t += 1
            else
                result.append(d_f[f])
                f += 1

        return result

    def choose(self, index):
        # Return a Tree given the index
        elif index == 0:
            return (self, -1)
    
        res, index = children[0].choose(index-1)
        if index == -1:
            return (res, index)

        return children[1].choose(index)

    def depth(self):
        # Return the maximum depth of the tree
        return np.max([c.depth() for c in self.children], initial=0) + 1
    
    def complexity(self):
        # Return the number of nodes of the tree
        return np.sum([c.complexity() for c in self.children], initial=1)


def generate_trees(data, cat, labels, number, max_initial_depth):
    # generate a number of trees, with an initial maximum depth
    return [Tree().generate(data, cat, set(labels), max_initial_depth, (None,0)) for _ in range(number)]
    

def fitness(tree, data, labels):
    # Number of equal elements TODO: should this be normalized?
    # TODO FITNESS SHOULD TAKE DEPTH INTO ACCOUNT
    gen_labels = tree.classify(data)
    return (labels == gen_labels).sum()


def crossover(tree1, tree2, max_depth=None):
    # Select Trees
    r1, r2 = Tree(tree1), Tree(tree2)

    nodes1 = r1.complexity()
    n1 = r.randrange(nodes1)
    c_tree1, i_tree1 = r1.choose(n1)
    while max_depth is not None and c_tree1.depth() <= max_depth:
        n1 = r.randrange(nodes1)
        c_tree1, i_tree1 = r1.choose(n1)

    nodes2 = r2.complexity()
    n2 = r.randrange(nodes2)
    c_tree2, i_tree2 = r2.choose(n2)
    while max_depth is not None and c_tree2.depth() <= max_depth:
        n2 = r.randrange(nodes2)
        c_tree2, i_tree2 = r2.choose(n2)

    # Switch trees
    temp_parent = c_tree1.parent

    c_tree1.parent = c_tree2.parent
    c_tree2.parent = temp_parent

    c_tree1.parent.children[i_tree1] = c_tree2
    c_tree2.parent.children[i_tree2] = c_tree1

    return tree1, tree2 # TODO: validate that this works, check with fitness function


def mutation(data, cat, labels, tree):
    # changed mutation to random subtree
    rt = Tree(tree)
    nodes = rt.complexity()
    n = r.randrange(nodes)
    c_tree, _ = rt.choose(n)
    
    c_tree.generate(data, cat, set(labels), c_tree.depth(), c_tree.parent)

    return rt # TODO: validate that this is done using references or copies. If not refeerences, than change code on other areas too


def gp(data, cat, labels, generations, pop_size=1000, mutation=0, cross_rate=0.7, max_depth=None, cross_max_depth=None):
    """
    GP algorithm for decision trees based on dataframe

    data: NORMALIZED dataframe containing the data FOR THIS GP
    cat: dataframe containing booleans if label is categorical
    labels: True results for each row in data
    """
    r.seed()

    # Initial population
    print("Creating initial population...")
    parents = generate_trees(data, cat, labels, pop_size, 3)

    for i in tqdm(range(generations)):
        print("Mutating phase...") #TODO selection
        parents2 = []
        for parent in parents:
            if r.random() < mutation:
                # Mutation#r.sample(list(range(len(parents))), k=1)
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
            if c1.depth() < max_depth:
                parents2.append(c1)
            if c2.depth() < max_depth:
                parents2.append(c2)

        parents.extend(parents2)

        print("Selecting next generation...")
        s = [p for p in sorted(parents, key=lambda c: fitness(c, data, labels))]# if not np.isnan(fitness(child, data, labels)) ]
        print(f"Best fitness {i}: {fitness(s[0], data, labels)}")

        parents = s[:100] # Next generation
    
    return parents[0]
