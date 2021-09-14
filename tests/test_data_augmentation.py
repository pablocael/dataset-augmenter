import numpy as np
from dataset_augmenter.core import DataTransformerPipeline

def test_transformer_pipeline():

    # generate 100 images for testing
    test_images = np.random.randint(0, 256, size=(100, 28, 28), dtype=np.uint8)

    # generate 100 labels from 0 to 9
    test_labels = np.random.randint(0, 10, size=100, dtype=np.uint8)

    pipeline = DataTransformerPipeline()

    pipeline.add_elastic_transformer()
    pipeline.add_rotation_transformer(-10, 10)

    # sample 33 images to transform
    transformed_images, transformed_labels = pipeline.sample_and_perform_transformation(sample_size=33, X=test_images, Y=test_labels)

    assert transformed_images.shape[0] == 33
    assert transformed_labels.shape[0] == 33

    # output transformed dimensions should be kept the same
    assert transformed_images.shape[1] == test_images.shape[1]
    assert transformed_images.shape[2] == test_images.shape[2]
    assert transformed_images.dtype == test_images.dtype
