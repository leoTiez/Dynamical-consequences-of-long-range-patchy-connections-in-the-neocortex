#!/usr/bin/python3
from modules.thesisUtils import *

import numpy as np
import cv2
import matplotlib.pyplot as plt


def convert_image_to_orientation_map(image, magnitude_threshold=50, num_orientation_ranges=8):
    # Derivatives
    deriv_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    deriv_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)

    # Orientation, Magnitude, Thresholding
    magnitude, orientation = cv2.cartToPolar(deriv_x, deriv_y, angleInDegrees=True)
    _, mask = cv2.threshold(magnitude, magnitude_threshold, 255, cv2.THRESH_BINARY)

    # Define orientation map
    orient_map = np.zeros(image.shape)

    # Set intensities
    orientation_thresh = 0
    orient_step = 360 / float(num_orientation_ranges)
    intensity = 255 / float(num_orientation_ranges)
    intens_step = intensity
    while orientation_thresh < 360:
        if orientation_thresh + orient_step == orient_step:
            orient_map[(mask == 255) & (orientation <= orientation_thresh + orient_step)] = intensity
        elif orientation_thresh + orient_step < 360:
            orient_map[
                (mask == 255) & (orientation > orientation_thresh) & (orientation <= orientation_thresh + orient_step)
            ] = intensity
        else:
            orient_map[(mask == 255) & (orient_step > orientation_thresh)] = intensity

        orientation_thresh += orient_step
        intensity += intens_step

    return orient_map


def create_image_bar(orientation, bar_width=5, size=(50, 50)):
    img = np.zeros(size, dtype='uint8')
    center = (size[0] // 2, size[1] // 2)
    dimx, dimy = size
    end_x = np.minimum(center[0] + dimx * np.cos(np.radians(orientation)), size[0])
    end_y = np.minimum(center[1] + dimy * np.sin(np.radians(orientation)), size[1])
    start_x = np.maximum(center[0] - dimx * np.cos(np.radians(orientation)), 0)
    start_y = np.maximum(center[1] - dimy * np.sin(np.radians(orientation)), 0)
    img = cv2.line(
        img,
        (np.around(start_x).astype('int'), np.around(start_y).astype('int')),
        (np.around(end_x).astype('int'), np.around(end_y).astype('int')),
        255,
        bar_width
    )

    return img


def image_with_spatial_correlation(num_circles=50, radius=5, size_img=(50, 50)):
    image = np.zeros(size_img)
    x_coordinates = np.random.choice(size_img[0], size=num_circles)
    y_coordinates = np.random.choice(size_img[1], size=num_circles)

    for x, y in zip(x_coordinates, y_coordinates):
        intensity = np.random.choice(range(10, 255))
        image = cv2.circle(image, (x, y), radius=radius, color=int(intensity), thickness=-1)

    return image


def test_main():
    image = load_image("monkey50.png")
    orient_map = convert_image_to_orientation_map(image)
    plt.imshow(orient_map, cmap='gray')
    plt.show()

    image_bar = create_image_bar(280)
    plt.imshow(image_bar, cmap='gray')
    plt.show()

    orient_bar = convert_image_to_orientation_map(image_bar, magnitude_threshold=0)
    plt.imshow(orient_bar, cmap='gray')
    plt.show()

    spart_img = image_with_spatial_correlation()
    plt.imshow(spart_img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    test_main()


