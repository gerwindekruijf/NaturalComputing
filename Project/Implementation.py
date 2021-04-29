from Decision_tree import *

# DATA = []

# with open("8data.txt", 'r') as f:
#     DATA = [[(float(token) if '−' not in token else -float(token[1:])) for token in line.split()] for line in f.readlines()]

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