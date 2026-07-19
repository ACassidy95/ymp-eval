import argparse
import cv2 as cv
import numpy as np
import sys

from matplotlib import pyplot as plt
from PIL import Image

P_BORDER_SIZE = 5
P_SIGMA = 0.01

img_stages = {}

def img2cv(img: Image) -> cv.typing.MatLike:
    return np.array(img)

def cv2img(img: cv.typing.MatLike) -> Image:
    return Image.fromarray(img)

def process_image(img_arr: cv.typing.MatLike) -> Image:
    cv_img = cv.cvtColor(img_arr, cv.COLOR_RGB2GRAY)
    cv_img = cv.copyMakeBorder(cv_img, P_BORDER_SIZE, P_BORDER_SIZE, P_BORDER_SIZE, P_BORDER_SIZE, borderType=cv.BORDER_CONSTANT, value=0)
    
    cv_img = cv.resize(cv_img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
    img_stages["Englarged"] = cv_img

    median = np.median(cv_img)
    lower = int(max(0, (1.0 - P_SIGMA) * median))
    upper = int(min(255, (1.0 + P_SIGMA) * median))

    se=cv.getStructuringElement(cv.MORPH_RECT, (8,8))
    bg=cv.morphologyEx(cv_img, cv.MORPH_DILATE, se)

    cv_img = cv.divide(cv_img, bg, scale=255)
    img_stages["Gray Divide"] = cv_img

    _, cv_img = cv.threshold(cv_img, 0, 255, cv.THRESH_OTSU)
    img_stages["Otsu Thresholding"] = cv_img

    return cv2img(cv_img)


def main(args):
    try:
        with Image.open(args.filename) as base_img:
            ci = img2cv(base_img)
            pi = process_image(ci)
            pi.save("./out.webp")

            stgs = len(img_stages)
            cls = 3
            rws = int(stgs / cls)
            for i, (k, v) in enumerate(img_stages.items()):
                plt.subplot(rws, cls, i + 1)
                plt.imshow(v, 'gray')
                plt.title(k)
            plt.show()
                
    except FileNotFoundError:
        print(f"{args.filename} - such a file does not exist sir. Perhaps you must have dreamed it.")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    sys.exit(main(args))