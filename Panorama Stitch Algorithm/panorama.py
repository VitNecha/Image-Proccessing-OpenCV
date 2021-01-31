import argparse
from datetime import datetime

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("p_l", type=str, nargs=1, metavar='path_left_img')
parser.add_argument("p_r", type=str, nargs=1, metavar='path_right_img')
parser.add_argument("p_o", type=str, nargs=1, metavar='path_output')
args = parser.parse_args()


def read_image(path):
    img = cv2.imread(path)
    return img


def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_image(right_img, left_img):
    # get the dims of left and right images
    r_h, r_w, _ = right_img.shape
    l_h, _, _ = left_img.shape

    # save the old ratio of right image
    right_image_ratio = r_h / r_w

    # create new width using desire height and old ratio
    new_w = int(l_h / right_image_ratio)

    # return the image with the new dims
    return cv2.resize(right_img, (new_w, l_h))


def remove_black_rectangle(img):
    gray_res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_res, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y + h, x:x + w]


def main(image_left_path, image_right_path, output_path):
    start_time = datetime.now()
    print(f"start time: {start_time}")
    image_right = read_image(image_right_path)
    image_left = read_image(image_left_path)

    # resize the greater image to the same height as the smaller image but save the ratio between width and height
    r_h, _, _ = image_right.shape
    l_h, _, _ = image_left.shape
    if l_h > r_h:
        image_left = resize_image(image_left, image_right)
    elif r_h > l_h:
        image_right = resize_image(image_right, image_left)

    del r_h, l_h

    # turn them to grayscale
    gray_image_right = to_grayscale(image_right)
    gray_image_left = to_grayscale(image_left)

    # get the dims of each image
    left_h, left_w = gray_image_left.shape
    right_h, right_w = gray_image_right.shape

    # create sift and get the kp and descriptor of each image using grayscale images
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_right, descriptors_right = sift.detectAndCompute(gray_image_right, None)
    keypoints_left, descriptors_left = sift.detectAndCompute(gray_image_left, None)

    # get all the matches using brute force matcher
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(descriptors_right, descriptors_left, k=2)

    # extract the best matches from brute force matches
    matches = []
    ratio = 0.85
    for m1, m2 in raw_matches:
        if m1.distance < ratio * m2.distance:  # m1 is a good match, save it
            matches.append([m1])

    # for display purpose only  ####  here you can see the matches of both photos, only the good ones
    image_panorama = cv2.drawMatchesKnn(gray_image_left, keypoints_left, gray_image_right, keypoints_right, matches,
                                        None)

    # take all the good points that were collected before to 2 arrays
    right_image_kp = np.float32([keypoints_right[m.queryIdx].pt for match in matches for m in match])
    left_image_kp = np.float32([keypoints_left[m.trainIdx].pt for match in matches for m in match])

    # find homography using the right and left images keypoints
    H, status = cv2.findHomography(right_image_kp, left_image_kp, cv2.RANSAC, 5.0)

    # output width and height
    width_panorama = right_w + left_w
    height_panorama = left_h

    # wrap perspective using right image only, will have a black box on both sides
    # the left black box will be filled with the left image so it will look like it stitched together
    # the right black box will be removed later
    res = cv2.warpPerspective(image_right, H, (width_panorama, height_panorama))

    # fill the left black box with the left image
    res[0:left_h, 0:left_w] = image_left

    # remove the right black box
    res = remove_black_rectangle(res)

    cv2.imwrite(output_path, res)
    end_time = datetime.now()
    print(f"end time: {end_time}\nTotal time: {end_time - start_time}")


if args.p_l and args.p_r and args.p_o:
    p_l = args.p_l[0]
    p_r = args.p_r[0]
    p_o = args.p_o[0]
    main(p_l, p_r, p_o)
