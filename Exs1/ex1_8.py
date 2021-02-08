
DATA = []
with open("8data.txt", 'r') as f:
    DATA = [[float(token) for token in line.split()] for line in f.readlines()]

