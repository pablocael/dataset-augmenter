import numpy as np
from dataset_augmenter import algorithms

def test_elastic_deformation():

    # just basic test to check if image is indeed being transformed and output dimensions and type are preserved
    test_image = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)

    deformed = algorithms.elastic_deformation(test_image, 3, 3)

    # check if image is being changed
    assert ((test_image - deformed) != 0).any()

    # image shape should not change
    assert deformed.shape == test_image.shape

    # image type should not change
    assert deformed.dtype == test_image.dtype

def test_noise():

    # just basic test to check if image is indeed being transformed and output dimensions and type are preserved
    test_image = np.zeros((28, 28), dtype=np.uint8)

    noised = algorithms.add_gaussian_noise(test_image, sigma=0.05)

    # check if image is being changed
    diff = (test_image - noised)
    assert (diff != 0).any()

    # image shape should not change
    assert noised.shape == test_image.shape

    # image type should not change
    assert noised.dtype == test_image.dtype

    test_image = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)

    noised = algorithms.add_uniform_noise(test_image, intensity=10)

    # convert one of the images to int to avoid uint overflow
    diff = np.int32(test_image) - noised
    assert (diff != 0).any()

    # check if maximum intensity added is within [-10, 10]
    assert (abs(diff) <= 10).all()

    # image shape should not change
    assert noised.shape == test_image.shape

    # image type should not change
    assert noised.dtype == test_image.dtype
