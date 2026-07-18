import argparse
import cv2
import numpy as np
import sys

from matplotlib import pyplot as plt
from PIL import Image

def main(args):
    try:
        with Image.open(args.filename) as base_img:
            cv_img = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2BGR)
            plt.subplot(231),plt.imshow(cv_img, 'gray'),plt.title("Original")
            plt.show()
    except FileNotFoundError:
        print(f"{args.filename} - such a file does not exist sir. Perhaps you must have dreamed it.")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    sys.exit(main(args))