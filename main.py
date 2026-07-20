import argparse
import cv2 as cv
import numpy as np
import pytesseract as pt
import sys

from matplotlib import pyplot as plt
from PIL import Image

P_BORDER_SIZE = 5
P_SIGMA = 0.1

def img2cv(img: Image) -> cv.typing.MatLike:
    return np.array(img)

def cv2img(img: cv.typing.MatLike) -> Image:
    return Image.fromarray(img)

def process_image(img_arr: cv.typing.MatLike, save_intermediates: bool) -> dict|None:
    img_stages = {}
    cv_img = cv.cvtColor(img_arr, cv.COLOR_RGB2GRAY)
    if save_intermediates:
        img_stages["Original_(Grayscale)"] = cv_img
    cv_img = cv.copyMakeBorder(cv_img, P_BORDER_SIZE, P_BORDER_SIZE, P_BORDER_SIZE, P_BORDER_SIZE, borderType=cv.BORDER_CONSTANT, value=0)
    
    cv_img = cv.resize(cv_img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    if save_intermediates:
        img_stages["Englarged"] = cv_img

    median = np.median(cv_img)
    lower = int(max(0, (1.0 - P_SIGMA) * median))
    upper = int(min(255, (1.0 + P_SIGMA) * median))

    se=cv.getStructuringElement(cv.MORPH_RECT, (8,8))
    bg=cv.morphologyEx(cv_img, cv.MORPH_DILATE, se)

    cv_img = cv.divide(cv_img, bg, scale=255)
    if save_intermediates:
        img_stages["Gray_Divide"] = cv_img

    cv_img = cv.bitwise_not(cv_img)
    if save_intermediates:
        img_stages["Inverted"] = cv_img

    _, cv_img = cv.threshold(cv_img, 0, 255, cv.THRESH_OTSU)
    # cv_img = cv.adaptiveThreshold(cv_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
    if save_intermediates:
        img_stages["Thresholded"] = cv_img

    krnl = np.ones((3,3),np.uint8)
    itr = 1
    ero = cv.erode(cv_img, krnl, iterations=itr)
    dil = cv.dilate(cv_img, krnl, iterations=itr)
    opn = cv.morphologyEx(cv_img, cv.MORPH_OPEN, kernel=krnl, iterations=itr)
    clo = cv.morphologyEx(cv_img, cv.MORPH_CLOSE, kernel=krnl, iterations=itr)

    if save_intermediates:
        img_stages["Eroded"] = ero
        img_stages["Dilated"] = dil
        img_stages["Opened"] = opn
        img_stages["Closed"] = clo
    
    if len(img_stages) > 0:
        return img_stages
    else:
        return None

def extract_image_text(img: Image) -> str:
    txt = pt.image_to_string(img)
    return txt

def main(args):
    try:
        with Image.open(args.filename) as base_img:
            ci = img2cv(base_img)
            pis = process_image(ci, True)

            if pis:
                stgs = len(pis)
                cls = 3
                rws = int(stgs / cls) + 1
                for i, (k, v) in enumerate(pis.items()):
                    plt.subplot(rws, cls, i + 1)
                    plt.imshow(v, 'gray')
                    plt.title(k)

                    i = cv2img(v)
                    i.save(f"{args.filename}-{k}.webp")

                    print(f"{k} - {extract_image_text(i)}\n---")
                plt.show()
                
    except FileNotFoundError:
        print(f"{args.filename} - such a file does not exist sir. Perhaps you must have dreamed it.")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    sys.exit(main(args))