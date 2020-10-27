import argparse

parser = argparse.ArgumentParser()
parser.add_argument("square", help="number to be squared", type=int)
parser.add_argument("--verbosity", "-v", help="increase verbosity", action="count")

args = parser.parse_args()
if args.verbosity == 2:
    print(f"Running {__file__}")
