"""
Provides algorithms for image data augmentation
"""

import numpy as np
from stsci.ndimage.interpolation import map_coordinates
from stsci.ndimage import gaussian_filter as gaussian_filter, rotate

def elastic_deformation(image: np.ndarray, intensity: float, sigma: float) -> np.ndarray:
    """
    Performs elastic augmentation on image

    Returns a new image containing the augmented data

    Reference paper:
    - Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis

    Parameters
    ----------

    image: a monochromatic (1 channel) image, format (height, width)
    sigma: the sigma of gaussian blur to apply into the displacement field
    intensity: intensity of the elastic deformation

    Taken from the above paper:

        "If σ is small, the field looks like a completely random
        field after normalization (as depicted in Figure 2, top right).
        For intermediate σ values, the displacement fields look like
        elastic deformation, where σ is the elasticity coefficient."

        So we must choose good values of sigma value.
    """
    assert len(image.shape) == 2

    shape = image.shape

    # create a displacement map for x and y from uniform distribution, and then smooth it with a gaussian
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * intensity
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * intensity


    # create the original regular grid coordinates
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    # apply the displacement fields for original regular index coordinates
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    # spline interpolate coordinates
    # allows indexing and ndarray by floating point and using spline patches to interpolate
    return map_coordinates(image, indices, order=1).reshape(shape)

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:

    """
    Performs rotation on image

    Parameters
    ----------

    image: a monochromatic (1 channel) image, format (height, width)
    angle: rotation angle in degrees

    """
    return rotate(image, angle, reshape=False)

def add_gaussian_noise(image: np.ndarray, sigma: float = 0.05) -> np.ndarray:

    """
    Add gaussian noise to a monochromatic image.

    Parameters
    ----------

    image: a monochromatic (1 channel) image, format (height, width) and dtype = np.uint8
    sigma: the standard deviation of the gaussian distribution.

    This function works by changing the intensity value of each pixel by a normal distribution which mean
    is equal to orignal pixel intensity value and standard deviation is given by sigma.

    """
    assert len(image.shape) == 2

    # sample from standard 0,1 normal
    samples = np.random.randn(*image.shape)

    # convert to desired standard deviation
    samples *= sigma

    samples *= 255 # convert to [-255, 255] space

    return np.uint8(np.clip(image + samples, 0, 255))

def add_uniform_noise(image: np.ndarray, intensity: int) -> np.ndarray:

    """
    Add uniform noise to a monochromatic image.

    Parameters
    ----------

    image: a monochromatic (1 channel) image, format (height, width) and dtype = np.uint8
    intensity: the maximum intensity of noise to be added, must be in the interval [-255, 255]

    This function adds noise sampled from a uniform distribution, centered at each pixel intensity value (mean = 0),
    and with range (-intensity, intensity)

    """
    assert len(image.shape) == 2

    # sample from standard 0,1 normal
    samples = np.random.rand(*image.shape) * 2 * intensity

    return np.uint8(np.clip(image + samples, 0, 255))
