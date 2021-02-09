import numpy as np
import random as r
import collections
import matplotlib.pyplot as plt


DATA = []
GENERATIONS = 50
POP_SIZE = 1000
CROSSOVER = 0.7
MUTATION = 0
SELECT_TREE = 0.1
MAX_INITIAL_DEPTH = 3
OPERATORS = ['x', 'log', 'exp', 'sin', 'cos', '+', '-', '*', '/']


with open("8data.txt", 'r') as f:
    DATA = [[(float(token) if '−' not in token else -float(token[1:])) for token in line.split()] for line in f.readlines()]


class Tree:
    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.children.append(child)

    def __eq__(self, other):
        if isinstance(other, Tree):
            for child in self.children:
                if any([child == oc] for oc in other.children):
                    continue
                return False
            
            for child in other.children:
                if any([child == oc] for oc in self.children):
                    continue
                return False

            return self.name == other.name
        return False

    def __str__(self):
        return f"({self.name}"+ ','.join([str(c) for c in self.children]) +" )"

    def deepcopy(self):
        return Tree(self.name, [c.deepcopy() for c in self.children])
    
    def generate(self, depth):
        r.seed()
        [n] = r.sample(OPERATORS, k=1)
        if n == 'x' or depth == 1:
            self.name = 'x'
            return self

        elif n in OPERATORS[1:5]:
            self.name = n
            child = Tree()
            self.children.append(child.generate(depth - 1))
            return self

        elif n in OPERATORS[5:]:
            [c] = r.sample(list(range(2,5)), k=1) # Number of items in summation, Max = 5
            self.name = n
            for i in range(c):
                child = Tree()
                self.children.append(child.generate(depth - 1))
            return self

    def choose(self, p):
        r.seed()

        # Pick this Tree
        if p >= r.random():
            return self
        
        # No children
        if len(self.children) == 0:
            return None

        c = [child.choose(p) for child in self.children]
        c = [i for i in c if i]
        
        if len(c) == 0:
            return None
        
        [pick] = r.sample(c, k=1)

        return pick

    def replace(self, tree_old, tree_new):
        for i in range(len(self.children)):
            # print(f'{tree_old}---{self.children[i]}')
            if self.children[i] == tree_old:
                self.children[i] = tree_new
                return 1
            if self.children[i].replace(tree_old, tree_new):
                return 1

        # no child got replaced
        return 0

    def calculate(self, x):
        result = 0

        if self.name == 'x':
            result = x

        elif self.name == 'log':
            result = np.log(self.children[0].calculate(x))
        elif self.name == 'exp':
            result = np.exp(self.children[0].calculate(x))
        elif self.name == 'sin':
            result = np.sin(self.children[0].calculate(x))
        elif self.name == 'cos':
            result = np.cos(self.children[0].calculate(x))

        elif self.name == '+':
            result = np.sum([child.calculate(x) for child in self.children])
        elif self.name == '-':
            result = self.children[0].calculate(x) - np.sum([child.calculate(x) for child in self.children[1:]])
        elif self.name == '*':
            result = np.prod([child.calculate(x) for child in self.children])
        elif self.name == '/':
            result = self.children[0].calculate(x) / np.prod([child.calculate(x) for child in self.children[1:]])

        return result

    def depth(self):
        if len(self.children) == 0:
            return 1

        return np.max( [c.depth() for c in self.children] ) + 1


def fitness(tree):
    return np.sum( np.abs( [x[1] - tree.calculate(x[0]) for x in DATA] ) )


def crossover(tree1, tree2):
    sub_tree = None
    while sub_tree is None:
        sub_tree = tree1.choose(SELECT_TREE)
    
    c1 = sub_tree.deepcopy()
    
    sub_tree = None
    while sub_tree is None:
        sub_tree = tree2.choose(SELECT_TREE)
    
    c2 = sub_tree.deepcopy()

    r1 = tree1.replace(c1, c2) if tree1 != c1 else c1
    r2 = tree2.replace(c2, c1) if tree2 != c2 else c2

    # TODO test of dit werkt en niet fout gaat bij de tweede keer overschrijven, we werken met references immer
    if not (r1 and r2):
        raise Exception("Couldn't replace the tree nodes. Your code is not working")
    
    return tree1, tree2


def mutation(tree):
    sub_tree = None
    while sub_tree is None:
        sub_tree = tree1.choose(SELECT_TREE)
    
    c1 = sub_tree.deepcopy()

    c2 = tree.generate(c1.depth())
    
    return tree.replace(c1, c2)
    

def gp():
    result = []

    # Initial population
    parents = []
    for i in range(POP_SIZE):
        t = Tree()
        parents.append(t.generate(MAX_INITIAL_DEPTH))

    for i in range(GENERATIONS):
        children = []
        print("Filling children")
        while len(children) < POP_SIZE:
            r.seed()
            if r.random() < MUTATION:
                [p1] = r.sample(list(range(len(parents))), k=1)
                children.append( mutation(p1) )

                
            elif r.random() < CROSSOVER:
                # Selection (Tournament)
                [pp1, pp2, pp3, pp4] = r.sample(list(range(len(parents))), k=4)

                p1 = parents[pp1] if fitness(parents[pp1]) < fitness(parents[pp2]) else parents[pp2]
                p2 = parents[pp3] if fitness(parents[pp3]) < fitness(parents[pp4]) else parents[pp4]

                # Crossover
                c1, c2 = crossover(p1.deepcopy(), p2.deepcopy())  
                children.extend([c1, c2])
        
        print("Storing child")
        
        # store best child
        s = [child for child in (sorted(children, key=lambda c: fitness(c))) if not np.isnan(fitness(child)) ]
        print(fitness(s[0]))
        print(s[0])
        result.append([s[0], s[0].depth()])

        parents = children
    
    return result


trees = gp()
scores_trees = [fitness(tree[0]) for tree in trees]
depths_trees = [tree[1] for tree in trees]

print(trees[-1])

plt.plot([x for x in range(1, len(trees) + 1)], scores_trees, color="green", label="tree score")
plt.plot([x for x in range(1, len(trees) + 1)], depths_trees, color="red", label="tree depth")
plt.title("Fitness scores for elapsed iterations")
plt.legend()
plt.xlabel("Iteration x")
plt.ylabel("Fitness score")
plt.show()
