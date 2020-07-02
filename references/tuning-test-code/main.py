import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--text')
args = parser.parse_args()

print(args.text)
