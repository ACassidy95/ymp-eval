import argparse
import sys

def main(args):
    print("Hello world!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    sys.exit(main(args))