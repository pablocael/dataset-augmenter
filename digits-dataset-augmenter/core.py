"""
Core functionalities for digit-dataset-augmenter package
"""

import cv2
from enum import IntEnum
import numpy as np

from number_generator.core import DigitImageDataset

class DigitAugmentationMethod(IntEnum):
    AUGMENTATION_ELASTIC = 1
    AUGMENTATION_AFFINE = 2
    AUGMENTATION_NOISE = 4

class DigitImageDatasetAugmentator(DigitImageDataset):
    """
    A special digit dataset that will augment the data by using digit deformation
    """

    def __init__(self, labels: np.array, images: np.ndarray, percent_augmentation=1: float, augmentation_method=DigitAugmentationMethod.AUGMENTATION_ELASTIC):

        if augmentation_method not in DigitImageDatasetAugmentator.__accepted_augmentation_methods:
            raise ValueError(f'augmentation_method must be one of the following: {DigitImageDatasetAugmentator.__accepted_augmentation_methods}')

        super(DigitImageDatasetAugmentator, self).__init__(labels, images)
        """
        Construct a digit dataset from list of examples and labels and augments the data

        Parameters
        ----------

        labels:
        a np.array of type uint8 storing each image digit example label

        images:
        digit examples in a numpy array with format (N, height, width), where N is the number of examples


        labels length and images.shape[0] must have same size

        percent_augmentation:
        the percentage, related to original dataset size, of new examples to generate
        if percent_augmentation==1, then the dataset size will be doubled

        augmentation_method:
        the augmentation method to use.

        - elastic: augment digits by deforming it's shape by using an non-rigid transformation
            (see reference "Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis")

        - affine: randomly translates and rotates each example

        - noise: adds noise to examples
        """

        self._percent_augmentation = percent_augmentation
        self._augmentation_method = augmentation_method

        # create a dictionary to store classes examples
        self._digit_examples = {}
        for i in range(10):
            mask = labels == i
            self._digit_examples[i] = images[mask]

        # store sample image shape as (width, height)
        self._sample_shape = images.shape[1:][::-1]

        self._generate_augmented_examples()


    def augment_from_digit_dataset(self, digit_dataset: DigitImageDataset) -> DigitImageDataset:
        """
        Load and generate dataset augmentation from an existing digit_dataset
        """
        pass

    def _generate_augmented_examples(self):

        # basic idea: go to each class and generate self._percent_augmentation new examples using elastic method
        pass
