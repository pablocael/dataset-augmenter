"""
Core functionalities for digit-dataset-augmenter package
"""

import numpy as np
from typing import Iterable

from . import algorithms

class DataTransformer:

    def __call__(self, X: np.ndarray, Y: Iterable):
        """
        Main data trasnform operator special method.

        Parameters
        ----------

        X: an ndarray image with format (N, height, width)
        Input data X entries must by a binary images

        """
        raise NotImplementedError()

class ElasticDeformDataTransformer(DataTransformer):


    def __init__(self, intensity: float, sigma: float):
        """
        Construct an ElasticDeformDataTransformer.

        Parameters
        ----------

        See data-augmenter.algorithms.elastic_deformation for more information on the elastic deformation algorithm.

        - intensity: the strength of the deformation - the larger, the more formation data will suffer
        - sigma: smoothing factor of the deformation - the larger, the smoother the transformation will be. A small value of sigma
            will cause the data to be more locally fragmented (will less local coeherence)
        """

        self._intensity = intensity
        self._sigma = sigma

    def __call__(self, X, Y):

        assert len(X.shape) == 3, 'X data must be have format (N, height, width)'

        shape = X.shape

        result = []
        for i in range(shape[0]):

            deformed = algorithms.elastic_deformation(X[i], intensity=self._intensity, sigma=self._sigma)
            result.append(deformed)

        return np.array(result), Y

class RotationDataTransformer(DataTransformer):


    def __init__(self, min_angle, max_angle):
        """
        Construct an RotationDataTransformer.

        Parameters
        ----------

        - min_angle: minimum rotation angle to be choosen from an uniform distribution
        - max_angle: maximum rotation angle to be choosen from an uniform distribution

        All angles are set in degrees, in the interval (-179, 180)

        min_angle must be smaller than max_angle

        """
        assert min_angle < max_angle, 'min_angle must be smaller than max_angle'

        self._min_angle = np.clip(min_angle, -179, 180)
        self._max_angle = np.clip(max_angle, -179, 180)


    def __call__(self, X, Y):

        assert len(X.shape) == 3, 'X data must be have format (N, height, width)'

        shape = X.shape

        result = []
        for i in range(shape[0]):

            angle_range = self._max_angle - self._min_angle
            rotation_angle = np.random.rand() * angle_range - angle_range / 2
            rotated = algorithms.rotate_image(X[i], angle=rotation_angle)
            result.append(rotated)

        return np.array(result), Y

class DataTransformerPipeline:

    def __init__(self):

        self._transformers = []

    def add_elastic_transformer(self, intensity: float = 3, sigma: float = 5):
        self._transformers.append(ElasticDeformDataTransformer(intensity=intensity, sigma=sigma))

    def add_rotation_transformer(self, min_angle: float, max_angle: float):
        self._transformers.append(RotationDataTransformer(min_angle=min_angle, max_angle=max_angle))

    def perform_transformation(self, X, Y):

        """
        Perform transformation in the whole input dataset

        Parameters
        ----------

        - X: data examples, in format (N, height, width)
        - Y: data labels in format (N,)

        """

        for transformer in self._transformers:
            X, Y = transformer(X, Y)

        return X, Y

    def sample_and_perform_transformation(self, sample_size: int, X: np.ndarray, Y: Iterable):

        """
        Perform transformation in sampled data from input dataset

        Parameters
        ----------

        - sample_size: the number of examples to sample - sample is performed uniformly
        - X: data examples, in format (N, height, width)
        - Y: data labels in format (N,)

        """

        X_sampled, Y_sampled = self._sample_from_data(size=sample_size, X=X, Y=Y)
        return self.perform_transformation(X_sampled, Y_sampled)

    def _sample_from_data(self, size, X, Y):
        """
        Collect the input data for augmentation process.

        Parameters
        ----------

        - X: data examples, in format (N, height, width)
        - Y: data labels in format (N,)
        - size is the number of samples to collect without replacement , 0 > size > X.shape[0]

        """

        assert size > 0 and size <= X.shape[0], 'cannot sample 0 or more than the number of examples in the dataset'

        N = X.shape[0] # original number of examples

        choosen_indices = np.random.choice(np.arange(N), size=size, replace=False)

        input_images = X[choosen_indices]
        input_labels = Y[choosen_indices]

        return input_images, input_labels





