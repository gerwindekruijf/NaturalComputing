import numpy as np
import random as r

DATA = []
with open("8data.txt", 'r') as f:
    DATA = [[float(token) for token in line.split()] for line in f.readlines()]

GENERATIONS = 50
CROSSOVER = 0.7
MUTATION = 0

class Tree:
    def init(self, name='root', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def choose(self, p):
        r.seed()

        # Pick this Tree
        if p >= r.random():
            return Tree
        
        # No children
        if len(self.children) == 0:
            return None

        c = [child.choose for child in self.children]
        c = [i for i in c if i]
        
        if len(c) == 0:
            return None
        
        return c[ r.sample(c, k=1)[0] ]

    def replace(self, tree_old, tree_new):
        for i in range(len(children)):
            if children[i] is tree_old:
                children[i] = tree_new.copy()
                return 1
            if children[i].replace(tree_old, tree_new)
                return 1
        
        # no child got replaced
        return 0



    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

    def calculate(self, x):
        result = 0

        if self.name == 'x':
            result = float(x)

        elif self.name == 'log':
            result = np.log(children[0].calculate(x))
        elif self.name == 'exp':
            result = np.exp(children[0].calculate(x))
        elif self.name == 'sin':
            result = np.sin(children[0].calculate(x))
        elif self.name == 'cos':
            result = np.cos(children[0].calculate(x))

        elif self.name == '+':
            result = np.sum([child.calculate(x) for child in self.children])
        elif self.name == '-':
            result = self.children[0].calculate(x) - np.sum([child.calculate(x) for child in self.children[1:]])
        elif self.name == '*':
            result = np.prod([child.calculate(x) for child in self.children])
        elif self.name == '/':
            result = self.children[0].calculate(x) / np.prod([child.calculate(x) for child in self.children[1:]])

        return result


def fitness(tree, data):
    t = [Tree.calculate(x[0]) for x in data]
    y = [x[1] for x in data]
    return np.sum( np.abs(y - t) )


def crossover(tree1, tree2):
    r.seed()
    if CROSSOVER < r.random():
        return tree1, tree2

    c1, c2 = None, None
    while choose1 is None:
        c1 = tree1.choose()

    while choose2 is None:
        c2 = tree2.choose()

    r1 = tree1.replace(c1, c2)
    r2 = tree2.replace(c2, c1)

    # TODO test of dit werkt en niet fout gaat bij de tweede keer overschrijven, we werken met references immer
    if not (r1 and r2):
        raise Exception("Couldn't replace the tree nodes. Your code is not working")
    
    return tree1, tree2 # hoeft denk ik niet, want dit zijn references?

def gp():
    #TODO implement alles