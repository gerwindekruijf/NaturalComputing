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
        self.label = label          # Data label, or classification label
        self.operator = operator    # Operator
        self.value = value          # Data comparison value
        self.parent = parent        # Tuple containing parent and childindex for efficiency (no eq function needed for crossover?)
        self.children = []          # Subtrees
        if children is not None:
            for child in children:
                self.children.append(child)

    def generate(self, data, depth, parent):
        self.parent = parent
        self.operator = OPERATORS[0] if depth == 0 else r.choice(OPERATORS)

        if self.operator == OPERATORS[0]:
            self.label = r.choice(data.shape[1]) # select random column
        else
            # We could split the data to get more accurate representation, big no from me though
            self.value = r.choice(data[label])
            for i in range(2):
                child = Tree()
                self.children.append(child.generate(data, depth - 1, (this, i)))

        return self

    # def __eq__(self, other):
    #     if isinstance(other, Tree):
    #         for child in self.children:
    #             if any([child == oc] for oc in other.children):
    #                 continue
    #             return False
            
    #         for child in other.children:
    #             if any([child == oc] for oc in self.children):
    #                 continue
    #             return False

    #         return self.name == other.name
    #     return False

    # def __str__(self):
    #     return f"({self.name}"+ ','.join([str(c) for c in self.children]) +" )"

    # def deepcopy(self):
    #     return Tree(self.name, [c.deepcopy() for c in self.children])

    # def choose(self, p):
    #     r.seed()

    #     # Pick this Tree
    #     if p >= r.random():
    #         return self
        
    #     # No children
    #     if len(self.children) == 0:
    #         return None

    #     c = [child.choose(p) for child in self.children]
    #     c = [i for i in c if i]
        
    #     if len(c) == 0:
    #         return None
        
    #     [pick] = r.sample(c, k=1)

    #     return pick

    # def replace(self, tree_old, tree_new):
    #     for i in range(len(self.children)):
    #         # print(f'{tree_old}---{self.children[i]}')
    #         if self.children[i] == tree_old:
    #             self.children[i] = tree_new
    #             return 1
    #         if self.children[i].replace(tree_old, tree_new):
    #             return 1

    #     # no child got replaced
    #     return 0

    def classify(self, data):
        
        # This is a leaf, classify the data
        if self.operator == OPERATORS[0]:
            return [self.label for _ in data]

        # Sort data in True or False for this statement
        s = np.array()
        for i in data:
            if self.operator == OPERATORS[1]: # ==
                s.append(data[self.label] == self.value)
            if self.operator == OPERATORS[2]: # <
                s.append(data[self.label] < self.value)
            if self.operator == OPERATORS[3]: # >
                s.append(data[self.label] > self.value)

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
        elif index == 0:
            return (self, -1)
    
        res, index = children[0].choose(index)
        if index == -1:
            return (res, index)

        return children[1].choose(index)
        

    def depth(self):
        # Return the maximum depth of the tree
        return np.max([c.depth() for c in self.children], initial=1) 
    
    def complexity(self):
        # Return the number of nodes of the tree
        return np.sum([c.complexity() for c in self.children], initial=1) 

def generate_trees(data, number, max_initial_depth):
    # generate a number of trees, with an initial maximum depth
    return [Tree().generate(data, max_initial_depth, (None,0)) for _ in range(number)]
    

def fitness(tree, data, labels):
    # Number of equal elements TODO: should this be normalized?
    gen_labels = tree.classify(data)
    return (labels == gen_labels).sum()


# def crossover(tree1, tree2):
#     # Select Subtree from parents
#     sub_tree = None
#     while sub_tree is None:
#         sub_tree = tree1.choose(SELECT_TREE)
    
#     c1 = sub_tree.deepcopy()
    
#     sub_tree = None
#     while sub_tree is None:
#         sub_tree = tree2.choose(SELECT_TREE)
    
#     c2 = sub_tree.deepcopy()

#     # Replace one subtree if the tree, except when the entire tree is selected (then change the entire tree)
#     r1 = tree1.replace(c1, c2) if tree1 != c1 else 1
#     r2 = tree2.replace(c2, c1) if tree2 != c2 else 1

#     # Check if both trees had alterations
#     if not (r1 and r2):
#         raise Exception("Couldn't replace the tree nodes. Your code is not working")
    
#     return tree1, tree2


def mutation(data, tree):
    # # Select subtree
    # sub_tree = None
    # while sub_tree is None:
    #     sub_tree = tree1.choose(SELECT_TREE)
    
    # c1 = sub_tree.deepcopy()

    # # Mutate subtree with newly generated tree
    # c2 = tree.generate(c1.depth())
    
    # return tree.replace(c1, c2)

    # changed mutation to random subtree
    nodes = tree.complexity()
    n = r.randrange(nodes)
    c_tree, _ = tree.choose(n)
    
    c_tree.generate(data, c_tree.depth(), c_tree.parent)

    return tree # TODO: validate that this is done using references or copies.



DATA = []

CROSSOVER = 0.7
SELECT_TREE = 0.1

MAX_DEPTH = 20
with open("8data.txt", 'r') as f:
    DATA = [[(float(token) if '−' not in token else -float(token[1:])) for token in line.split()] for line in f.readlines()]

def gp(data, generations, mutation=0):
    r.seed()
    result = []
    pop_size = 1000

    # Initial population
    print("Creating initial population...")
    parents = generate_trees(data, pop_size, 3)

    for i in tqdm(range(generations)):
        children = []
        print("Filling children")
    #     while len(children) < pop_size:
    #         if r.random() < MUTATION:
    #             # Mutation
    #             [p1] = r.sample(list(range(len(parents))), k=1)
    #             c1 = mutation(p1)
    #             if c1.depth() < MAX_DEPTH:
    #                 children.append(c1)

    #         elif r.random() < CROSSOVER:
    #             # Selection (Tournament)
    #             [pp1, pp2, pp3, pp4] = r.sample(list(range(len(parents))), k=4)

    #             p1 = parents[pp1] if fitness(parents[pp1]) < fitness(parents[pp2]) else parents[pp2]
    #             p2 = parents[pp3] if fitness(parents[pp3]) < fitness(parents[pp4]) else parents[pp4]

    #             # Crossover
    #             c1, c2 = crossover(p1.deepcopy(), p2.deepcopy())
    #             if c1.depth() < MAX_DEPTH:
    #                 children.append(c1)
    #             if c2.depth() < MAX_DEPTH:
    #                 children.append(c2)
    #             # children.extend([c1, c2])
        
    #     print("Storing child")
        
    #     # store best child
    #     s = [child for child in (sorted(children, key=lambda c: fitness(c))) if not np.isnan(fitness(child)) ]
    #     print(fitness(s[0]))
    #     print(s[0])
    #     result.append([s[0], s[0].depth()])

    #     parents = children
    
    # return result


# trees = gp()
# scores_trees = [fitness(tree[0]) for tree in trees]
# depths_trees = [tree[1] for tree in trees]

# print(trees[-1])

# plt.plot([x for x in range(1, len(trees) + 1)], scores_trees, color="green", label="tree score")
# plt.plot([x for x in range(1, len(trees) + 1)], depths_trees, color="red", label="tree depth")
# plt.title("Fitness scores for elapsed iterations")
# plt.legend()
# plt.xlabel("Iteration x")
# plt.ylabel("Fitness score")
# plt.show()
