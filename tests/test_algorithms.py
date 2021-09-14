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

