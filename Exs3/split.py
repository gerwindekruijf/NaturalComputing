import argparse
from tqdm import tqdm

def main(args):
    res = list()
    K = int(args.n)
    with open(args.file, 'r') as f:
        word = 0
        for line in tqdm(f.readlines()):
            line = line.rstrip('\n')
            res.extend([(line[i: j], word) for i in range(len(line)) for j in range(i + 1, len(line) + 1) if len(line[i:j]) == K])
            word += 1

    with open(args.file+"_split", 'w') as f:
        f.writelines([f'{r[0]} \n' for r in res])
    with open(args.file+"_ref", 'w') as f:
        f.writelines([f'{r[1]} \n' for r in res])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='filename')
    parser.add_argument('n', help='length of string')
    args = parser.parse_args()

    main(args)
