import argparse
from tqdm import tqdm

def main(args):
    res = list()
    with open(args.file, 'r') as f:
        for line in tqdm(f.readlines()):
            #line = line.rstrip('\n')
            res.extend(line.split())

    with open(args.file+"_format", 'w') as f:
        f.writelines([f'{r} \n' for r in res])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='filename')
    args = parser.parse_args()

    main(args)
