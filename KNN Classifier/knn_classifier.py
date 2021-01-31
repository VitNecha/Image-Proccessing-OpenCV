import argparse
import os
import random
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from skimage import feature
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def load_dataset(path):
    # path = input_train
    start_time = datetime.now()
    print(f"[{start_time}] Loading dataset")
    dataset = []
    for char_folder in os.listdir(path):
        char_folder_path = os.path.join(path, char_folder)
        for char in os.listdir(char_folder_path):
            img = cv2.imread(os.path.join(char_folder_path, char), cv2.IMREAD_GRAYSCALE)
            dataset.append((img, char_folder))
    end_time = datetime.now()
    print(f"[{datetime.now()}] done, took: {end_time - start_time}")
    return dataset

def split_dataset(dataset):
    start_time = datetime.now()
    print(f"[{start_time}] Splitting dataset to X and y")
    X, y = zip(*dataset)
    print(f"[{datetime.now()}] done, took: {datetime.now() - start_time}")
    return X, y


def chi_square(h1, h2):
    return 0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str, metavar='train path', help="Path to train input folder")
    parser.add_argument("test_path", type=str, metavar='test path', help="Path to test input folder")
    args = parser.parse_args()

    dataset = load_dataset(args.train_path)  # Load dataset into an array
    X, y = split_dataset(dataset)  # splits dataset into X, y. [X = images, y = labels (same indices)]
    ch_hogs = []
    for ch_img in X:  # for each image, make a HOG
        ch_hogs.append(feature.hog(ch_img, orientations=9,
                                   pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2),
                                   transform_sqrt=False,
                                   block_norm="L2"))

    # Splits dataset into validation set and train set
    rand = random.randint(1, 157)
    X_train, X_test, y_train, y_test = train_test_split(ch_hogs,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=rand,
                                                        shuffle=True,
                                                        stratify=y)
    del X, y

    # ************** Euclidean **************
    euclidean_max_percent = 0
    euclidean_max_index = 0

    # ************** Chi Square **************
    chi_square_max_percent = 0
    chi_square_max_index = 0

    # chi_square_model = KNeighborsClassifier(3, metric=chi_square, metric_params=ch_hogs)
    for i in range(1, 16):
        g_start_time = datetime.now()
        print(f"[{g_start_time}] iteration {i}\n")

        # ************** Euclidean **************

        start_time = datetime.now()
        print(
            f"[{start_time}] start fitting euclidean distance")
        euclidean_model = KNeighborsClassifier(i, metric="euclidean", p=2)
        euclidean_model.fit(X_train, y_train)
        print(f"[{datetime.now()}] done fitting, took: {datetime.now() - start_time}\n")

        euclidean_score = euclidean_model.score(X_test, y_test)
        if euclidean_score > euclidean_max_percent:
            euclidean_max_percent = euclidean_score
            euclidean_max_index = i
            euclidean_best_model = euclidean_model

        # ************** Chi Square **************

        start_time = datetime.now()
        print(f"[{start_time}] start fitting with chi square distance: ")
        chi_square_model = KNeighborsClassifier(i, metric=chi_square)
        chi_square_model.fit(X_train, y_train)
        print(f"[{datetime.now()}] done fitting, took: {datetime.now() - start_time}")

        chi_square_score = chi_square_model.score(X_test, y_test)
        if chi_square_score > chi_square_max_percent:
            chi_square_max_percent = chi_square_score
            chi_square_max_index = i
            chi_square_best_model = chi_square_model

        print(f"\n[{datetime.now()}]iteration {i}, took: {datetime.now() - g_start_time}")

    print(
        f"\n************** Euclidean **************\n euclidean max: {euclidean_max_percent} with {euclidean_max_index} neighbors\n")
    print(
        f"\n************** Chi Square **************\n chi square max: {chi_square_max_percent} with {chi_square_max_index} neighbors\n")
    del X_train, y_train, X_test, y_test, ch_hogs

    ch_hogs = []
    X, y = split_dataset(load_dataset(args.test_path))
    for ch_img in X:
        ch_hogs.append(feature.hog(ch_img, orientations=9,
                                   pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2),
                                   transform_sqrt=False,
                                   block_norm="L2"))

    print(f"\n************** Euclidean Prediction **************\n")

    start_time = datetime.now()
    print(f"[{start_time}] start testing the euclidean model\n")
    euclidean_y_pred = euclidean_best_model.predict(ch_hogs)
    euclidean_class_report = classification_report(y, euclidean_y_pred, output_dict=True)
    euclidean_c_mat = confusion_matrix(y, euclidean_y_pred)

    euclidean_cr = pd.DataFrame(euclidean_class_report).transpose()
    euclidean_cr.to_csv("euclidean_results.csv")
    with open("euclidean_results.csv", "a") as csvfile:
        csvfile.write(np.array2string(euclidean_c_mat, separator=', '))
        csvfile.write(
            f"\n**** k = {euclidean_max_index} neighbors with euclidean function ****")

    print(f"\n[{datetime.now()}] done testing, took: {datetime.now() - start_time}\n")

    print(f"\n************** Chi Square Prediction **************\n")

    start_time = datetime.now()
    print(f"[{start_time}] start testing the chi square model\n")
    chi_square_y_pred = chi_square_best_model.predict(ch_hogs)
    chi_square_class_report = classification_report(y, chi_square_y_pred, output_dict=True)
    chi_square_c_mat = confusion_matrix(y, chi_square_y_pred)

    chi_square_cr = pd.DataFrame(chi_square_class_report).transpose()
    chi_square_cr.to_csv("chi_square_results.csv")
    with open("chi_square_results.csv", "a") as csvfile:
        csvfile.write(np.array2string(chi_square_c_mat, separator=', '))
        csvfile.write(f"\n**** k = {chi_square_max_index} neighbors with chi_square function ****")
    print(f"\n[{datetime.now()}] done testing, took: {datetime.now() - start_time}\n")


main()
