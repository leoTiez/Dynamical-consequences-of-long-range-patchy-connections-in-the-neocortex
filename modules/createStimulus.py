#!/usr/bin/python3
from modules.thesisUtils import *

import numpy as np
import cv2
import matplotlib.pyplot as plt


INPUT_TYPE = {
    "plain": 0,
    "perlin": 1,
    # "bar": 2,
    # "circles": 3,
    "natural": 4,
    # "edges": 5,
    "random": 6
}


def convert_image_to_orientation_map(image, magnitude_threshold=50, num_orientation_ranges=8, **kwargs):
    """
    Convert the edges of an image to a an orientation color map
    :param image: The actual image
    :param magnitude_threshold: The threshold that determines an edge
    :param num_orientation_ranges: Number of classes that are discriminated
    :param kwargs: Additional parameters that are passed because of the factory method that should not cause an error
    :return: The image with only the edges colored
    """
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


def create_image_bar(orientation, bar_width=5, size=(50, 50), shuffle=False, **kwargs):
    """
    Create image with a single bar
    :param orientation: Orientation of the bar
    :param bar_width: width of the bar
    :param size: Size of the image
    :param shuffle: Flag to determiner whether image pixels should be shuffled
    :param kwargs: Additional parameters that are passed because of the factory method that should not cause and error
    :return: Image with bar
    """
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

    if shuffle:
        np.random.shuffle(img.reshape(-1))
        img = img.reshape(size)
    return img


def image_with_spatial_correlation(
        num_circles=10,
        radius=10,
        size=(50, 50),
        background_noise=False,
        shuffle=False,
        **kwargs
):
    """
    Create image with circles such that there is a spatial correlation of pixels
    :param num_circles: Number of circles
    :param radius: Radius of each circle
    :param size: Size of the image
    :param background_noise: Flag for creating background noise. If False, the background is black
    :param shuffle: randomly shuffle image pixels
    :return: Image with circles for spatial correlation of color
    """
    x_coordinates = np.random.choice(size[0], size=num_circles)
    y_coordinates = np.random.choice(size[1], size=num_circles)

    image = np.zeros(size)
    for x, y in zip(x_coordinates, y_coordinates):
        if background_noise:
            intensity = np.random.choice(range(0, 256))
        else:
            intensity = np.random.choice(range(10, 256))
        image = cv2.circle(image, (x, y), radius=radius, color=int(intensity), thickness=-1)

    if shuffle:
        np.random.shuffle(image.reshape(-1))
        image = image.reshape(size)

    return image


def perlin_image(size=50, resolution=(5, 5), **kwargs):
    """
    Create an image with spatial correlation using the Perlin noise distribution
    :param size: Size of the image. It takes only a single integer and assumes the image to be quadratic
    :param resolution: Resolution of the Perlin mesh with the randomly sampled vectors
    :param kwargs: Additional parameters that might be passed because of the factory method that should not cause
    an error
    :return: Perlin noise image
    """
    if type(size) == tuple:
        size = size[0]
    perlin_img = perlin_noise(size, resolution=resolution, spacing=1)
    perlin_img -= perlin_img.min()
    perlin_img = 255. * perlin_img / perlin_img.max()
    return perlin_img.astype('int')


def plain_stimulus(size=(50, 50), intensity=255):
    """
    Create a simple image with a single shade everywhere
    :param size: Size of the image
    :param intensity: Pixel intensity
    :return: The created image
    """
    img = np.full(size, intensity)
    return img


def stimulus_factory(input_type=INPUT_TYPE["plain"], **kwargs):
    """
    Stimulus factory method
    :param input_type: The input type that is to be created
    :param kwargs: Additional parameters that are needed for the respective functions that create the image
    :return: The created image
    """
    if input_type == INPUT_TYPE["circles"]:
        input_stimulus = image_with_spatial_correlation(**kwargs)
    elif input_type == INPUT_TYPE["bar"]:
        input_stimulus = create_image_bar(**kwargs)
    elif input_type == INPUT_TYPE["natural"]:
        input_stimulus = load_image(**kwargs)
    elif input_type == INPUT_TYPE["perlin"]:
        input_stimulus = perlin_image(**kwargs)
    elif input_type == INPUT_TYPE["plain"]:
        input_stimulus = plain_stimulus(**kwargs)
    elif input_type == INPUT_TYPE["edges"]:
        input_stimulus = load_image(**kwargs)
        input_stimulus = convert_image_to_orientation_map(input_stimulus, **kwargs)
    elif input_type == INPUT_TYPE["random"]:
        input_stimulus = image_with_spatial_correlation(shuffle=True, **kwargs)
    else:
        raise ValueError("Input type %s is not accepted" % input_type)

    return input_stimulus


def test_main():
    """
    Test main
    :return None
    """
    image = load_image("monkey50.png", path="../test-input/")
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

    perlin_img = perlin_image(50, resolution=(7, 7))
    plt.imshow(perlin_img, cmap="gray")
    plt.show()

    plain_img = plain_stimulus(size=(50, 50), intensity=255)
    plt.imshow(plain_img, cmap="gray", vmin=0, vmax=255)
    plt.show()


if __name__ == '__main__':
    test_main()


