import argparse
import os
from datetime import datetime

import cv2


def read_grayscale(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def add_padding(img):
    w, h = img.shape
    s = abs(w - h)
    if w > h:
        return cv2.copyMakeBorder(img, 0, 0, s // 2, s // 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        return cv2.copyMakeBorder(img, s // 2, s // 2, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def resize_image(img):
    return cv2.resize(img, (40, 40))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--train", type=str, metavar='train path', help="Path to train input folder")
    group.add_argument("-tt", "--test", type=str, metavar='test path', help="Path to test input folder")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"[PreProcessing] start time: {start_time}")
    try:
        if args.train:
            path = args.train
            os.mkdir(os.path.join("processed_train"))
        elif args.test:
            path = args.test
            os.mkdir(os.path.join("processed_test"))
        else:
            path = ""
            exit(1)
    except FileExistsError:
        pass

    for char_folder in os.listdir(path):
        char_folder_path = os.path.join(path, char_folder)
        try:
            if args.train:
                os.mkdir(os.path.join("processed_train", char_folder))
            if args.test:
                os.mkdir(os.path.join("processed_test", char_folder))
        except FileExistsError:
            pass
        for char in os.listdir(char_folder_path):
            img = read_grayscale(os.path.join(char_folder_path, char))
            img = add_padding(img)
            img = resize_image(img)
            if args.train:
                cv2.imwrite(os.path.join("processed_train", char_folder, char), img)
            if args.test:
                cv2.imwrite(os.path.join("processed_test", char_folder, char), img)
    print(f"[PreProcessing] total time: {datetime.now() - start_time}")


main()
