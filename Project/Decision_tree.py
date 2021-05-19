"""
Generate a decision tree on data using the dt_gp function
"""
import sys
import random as r

import numpy as np
from tqdm import tqdm

# ROOT is the first node of a tree
OPERATORS = ['leaf', '<', '>', '==', 'ROOT']

class Tree:
    """
    A Tree contains a label, operator and value
    """
    def __init__(self, label=None, children=None, parent=(None,0), operator=None, value=None, column=None, tree=None):
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
            self.parent = parent     # Tuple containing parent and childindex for efficiency

            self.children = []       # Subtrees
            if children is not None:
                i = 0
                for child in children:
                    self.children.append(Tree(tree=child,parent=(self,i)))

    def __str__(self):
        if self.operator == OPERATORS[0]:
            return str(self.label)
        if self.operator == OPERATORS[4]:
            return "Tree: " + str(self.children[0])

        res = str(self.column) + " " + str(self.operator) + " " + str(self.value) + " "
        children = [str(child) for child in self.children]
        return res + f"[{', '.join(children)}]"

    def generate(self, data, cat, labels, depth, parent):
        """
        Generate a new Tree from this node

        :param data: train data
        :type data: dataframe
        :param cat: categorical column indication
        :type cat: dictionary
        :param labels: labels this tree may use
        :type labels: list
        :param depth: maximum depth of this tree
        :type depth: int
        :param parent: parent tree
        :type parent: tree
        :return: generated tree
        :rtype: tree
        """
        self.parent = parent
        self.children = []
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
        """
        Classify data given this tree

        :param data: test data
        :type data: dataframe
        :return: classifications
        :rtype: numpy array
        """
        # This is a leaf, classify the data
        if self.operator == OPERATORS[0]:
            return np.array([self.label for _ in data.iterrows()])

        if self.operator == OPERATORS[4]:
            return self.children[0].classify(data)

        # Sort data in True or False for this statement
        splitted_data = []
        if self.operator == OPERATORS[1]: # ==
            splitted_data = (data[self.column] == self.value)
        elif self.operator == OPERATORS[2]: # <
            splitted_data = (data[self.column] < self.value)
        elif self.operator == OPERATORS[3]: # >
            splitted_data = (data[self.column] > self.value)

        splitted_data = np.array(splitted_data)

        # Classify data in subtrees
        d_t = self.children[0].classify(data[splitted_data])
        d_f = self.children[1].classify(data[~splitted_data])

        # Build result by combining the results
        r_t, r_f = 0, 0
        result = np.array([])
        for i in splitted_data:
            if i:
                result = np.append(result, d_t[r_t])
                r_t += 1
            else:
                result = np.append(result, d_f[r_f])
                r_f += 1

        return result

    def choose(self, index):
        """
        Select a tree given the index

        :param index: index of subtree to select
        :type index: int
        :return: tree given the index
        :rtype: tree
        """
        # Pass if root
        if self.operator == OPERATORS[4]:
            return self.children[0].choose(index)

        # return self if leaf or index is 0
        if index == 0 or self.operator == OPERATORS[0]:
            return (self, index-1)

        # go through children and see if it returns index < 0
        res, index = self.children[0].choose(index-1)
        if index == -1:
            return (res, index)

        return self.children[1].choose(index)

    def depth(self):
        """
        Calculate the maximum depth of the tree

        :return: depth
        :rtype: int
        """
        # Return the maximum depth of the tree
        child_depth = np.max([c.depth() for c in self.children], initial=0)
        return child_depth if self.operator == OPERATORS[4] else child_depth + 1

    def complexity(self):
        """
        Calculate the number of nodes within this tree

        :return: number of nodes
        :rtype: int
        """
        # Return the number of nodes of the tree
        child_complexity = np.sum([c.complexity() for c in self.children], initial=0)
        return child_complexity if self.operator == OPERATORS[4] else child_complexity + 1


def generate_trees(data, cat, labels, number, max_initial_depth):
    """
    Generate multiple trees

    :param data: train data
    :type data: dataframe
    :param cat: categorical column indication
    :type cat: dictionary
    :param labels: labels a tree may use
    :type labels: list
    :param number: number of trees
    :type number: int
    :param max_initial_depth: maximum depth of the trees
    :type max_initial_depth: int
    :return: trees
    :rtype: list
    """
    # generate a number of trees, with an initial maximum depth
    classifiers = list(set(labels))
    return [
        Tree()
        .generate(data, cat, classifiers, max_initial_depth, (Tree(operator=OPERATORS[4]),0))
        .parent[0] for _ in range(number)
        ]


def fitness(tree, data, labels, weights):
    """
    Calculate fitness for a tree

    :param tree: tree
    :type tree: tree
    :param data: test data
    :type data: dataframe
    :param labels: labels this tree may use
    :type labels: list
    :param weights: importance for all classifications + depth penalty
    :type weights: list
    :return: fitness score
    :rtype: float
    """
    gen_labels = tree.classify(data)

    equal = gen_labels[gen_labels == labels]

    result = 0.
    for i in range(len(weights)-1):
        result += len(equal[equal == i])/len(labels[labels==i])

    depth_penalty = tree.depth()/(len(data.columns) - 1) * weights[-1]
    return result - depth_penalty


def crossover(tree1, tree2, max_depth=None):
    """
    Create 2 new trees, using the parents

    :param tree1: parent tree 1
    :type tree1: tree
    :param tree2: parent tree 2
    :type tree2: tree
    :param max_depth: maximum depth of subtree to combine with child, defaults to None
    :type max_depth: int, optional
    :return: child trees
    :rtype: tuple
    """
    # Select Trees
    r_tree1 = Tree(tree=tree1, parent=(Tree(operator=OPERATORS[4]),0))
    r_tree2 = Tree(tree=tree2, parent=(Tree(operator=OPERATORS[4]),0))

    nodes1 = r_tree1.complexity()
    rand_node1 = r.randrange(nodes1)
    c_tree1, _ = r_tree1.choose(rand_node1)
    # Subtree should be within max_depth
    while max_depth is not None and c_tree1.depth() > max_depth:
        rand_node1 = r.randrange(nodes1)
        c_tree1, _ = r_tree1.choose(rand_node1)

    nodes2 = r_tree2.complexity()
    rand_node2 = r.randrange(nodes2)
    c_tree2, _ = r_tree2.choose(rand_node2)
    while max_depth is not None and c_tree2.depth() > max_depth:
        rand_node2 = r.randrange(nodes2)
        c_tree2, _ = r_tree2.choose(rand_node2)

    # Switch trees
    temp_parent1 = c_tree1.parent
    temp_parent2 = c_tree2.parent

    c_tree1.parent = c_tree2.parent
    c_tree2.parent = temp_parent1

    temp_parent1[0].children[temp_parent1[1]] = c_tree2
    temp_parent2[0].children[temp_parent1[1]] = c_tree1

    return r_tree1, r_tree2


def mutation(data, cat, labels, tree):
    """
    Mutate a new tree from a parent

    :param data: train data
    :type data: dataframe
    :param cat: categorical column indication
    :type cat: dictionary
    :param labels: labels this tree may use
    :type labels: list
    :param tree: parent tree
    :type tree: tree
    :return: mutated tree
    :rtype: tree
    """
    # changed mutation to random subtree
    r_tree = Tree(tree=tree, parent=(Tree(operator=OPERATORS[4]),0))
    nodes = r_tree.complexity()
    rand_node = r.randrange(nodes)
    c_tree, _ = r_tree.choose(rand_node)

    classifiers = list(set(labels))
    c_tree.generate(data, cat, classifiers, c_tree.depth(), c_tree.parent)
    return r_tree


def dt_gp(
    data, cat, labels, generations, pop_size, mutation_rate,
    cross_rate, fit_weights, max_depth=None, cross_max_depth=None, disp=False, seed=None):
    """
    GP algorithm for decision trees based on dataframe

    :param data: normalized dataframe containing the data for this GP
    :type data: dataframe
    :param cat: categorical column indication
    :type cat: dictionary
    :param labels: labels this gp may use
    :type labels: list
    :param generations: number of generations
    :type generations: int
    :param pop_size: number of trees in parent group
    :type pop_size: int
    :param mutation_rate: mutation rate
    :type mutation_rate: float
    :param cross_rate: crossover rate
    :type cross_rate: float
    :param max_depth: maximum depth of a tree
    :type max_depth: int
    :param cross_max_depth: maximum depth of subtree to combine with child
    :type cross_max_depth: int
    :param fit_weights: importance for all classifications + depth penalty
    :type fit_weights: list
    :param disp: print debug information, defaults to False
    :type disp: bool, optional
    :param seed: seed used for random
    :type seed: int, optional
    :return: best decision tree
    :rtype: tree
    """
    if seed is None:
        seed = r.randrange(sys.maxsize)
    r.seed(seed)
    if disp:
        print("Seed was:", seed)

    # Initial population
    if disp:
        print("Creating initial population...")
    parents = generate_trees(data, cat, labels, pop_size, 3)
    scores = [fitness(p, data, labels, fit_weights) for p in parents]

    for i in tqdm(range(generations)):
        if disp:
            print("Mutating phase...")
        parents2 = []
        for parent in parents:
            if r.random() < mutation_rate:
                # Mutation
                child = mutation(data, cat, labels, parent)
                parents2.append(child)

        parents.extend(parents2)
        scores.extend([fitness(p, data, labels, fit_weights) for p in parents2])

        if disp:
            print("Filling phase...")
        parents2 = []
        while len(parents2) < pop_size*cross_rate:
            # Selection (Tournament)
            [pp1, pp2, pp3, pp4] = r.sample(list(range(len(parents))), k=4)

            parent1 = parents[pp1] if scores[pp1] < scores[pp2] else parents[pp2]
            parent2 = parents[pp3] if scores[pp3] < scores[pp4] else parents[pp4]

            # Crossover
            child1, child2 = crossover(parent1, parent2, cross_max_depth)
            if max_depth is None or child1.depth() < max_depth:
                parents2.append(child1)
            if max_depth is None or child2.depth() < max_depth:
                parents2.append(child2)

        parents.extend(parents2)
        scores.extend([fitness(p, data, labels, fit_weights) for p in parents2])

        if disp:
            print("Selecting next generation...")
        patents_sort = sorted(list(zip(parents, scores)), key=lambda x: x[1], reverse=True)

        if disp:
            print(f"Best fitness {i}: {patents_sort[0][1]}")
            print(patents_sort[0][0])

        parents, scores = map(list,zip(*patents_sort[:pop_size])) # Next generation

    return parents[0]
