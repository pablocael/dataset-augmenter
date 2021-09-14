
# dataset-augmenter 0.0.1

Introduction
----------------------

This package provides functionalities for data augmentation, such as noise and rigid and non rigid transformations.

Data augmentation is often useful for expanding small datasets in order to increase variance in data (and possible add bias to trained model parameters). Data augmentation can be seen as a form of regularization.

*This package is built for python >= 3.6*

Installation
----------------------
```bash
pip3 install dataset-augmenter==0.0.1
```


Dependencies
----------------------

This package was build with a minimalist paradigm, trying to avoid bloatware package dependency.

Dependencis are the following:

- numpy >= 1.21.2

	Necessary to perform array processing and mathematical operations
 	This package requires 93M of space.
 
- pytest >= 6.2.5
	
	Necessary to perform unit tests
	This package requires 1.9M of space.
	
- argparse >= 1.4.0

	Argparse is used to handle input arguments in a more high level, scalable and organized fashion.
	This package requires just a few KB of free space.
	
- stsci.ndimage >= 0.10.3

	This package is a subset of scipy package.
	Necessary for performing efficient gaussian filtering and coordinate interpolation for data augmentation algorithms.
	This package requires 1M of space.


Usage
----------------------

#### 1. Augmenting a dataset:


```DataTransformerPipeline``` class provides easy to use data transformation operation that can be stacked in any order to augment data.
Several transformations are available: uniform noise, gaussian noise, rotations and elastic deformations.

#### Available data transformers are:
	
- **core.ElasticDeformDataTransformer**:
	Applies an elastic deformation with configurable intensity and smoothness. The implementation was based on the following reference:
	"Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis"
	
	This transformer deforms images by creating a smooth random displacement map, displacing image coordinates and then
	interpolating the new coordinates to generate a smooth rendering of the final deformed image. 
	
	The displacement map is shown in the image below:
	
	![](./doc-images/elastic-distortions.pbm "Elastic Distortion Displacement Map")
	
	The new deformed grid coordinates (above right image) must be interpolated to render an image without gaps. There are several ways of interpolating grid coordinates (such as linear, baricentric, and others), but core.ElasticDeformDataTransformer uses spline patches interpolation.
	
- **core.RotationDataTransformer**:
	Applies a random rotation choosen from a random uniform interval.
	
- **core.GaussianNoiseDataTransformer**:
	Applies gaussian noise to the dataset images using customized standard deviation.
	
- **core.UniformNoiseDataTransformer**:
	Applies uniform noise to the dataset images using customized noise intensity.

#### Using the transformer pipeline:
	
Example 1:

```py
from dataset_augmenter.core import DataTransformerPipeline

def load_data()
	...

original_X, original_label = load_data()

pipeline = DataTransformerPipeline

pipeline.add_elastic_transformer(intensity=2, sigma=2)
pipeline.add_rotation_transformer(min_angle=-10, max_angle=10)

# create 100 new examples by random sampling original data and applying elastic transformation -> rotation.
transformed_X, transformed_label = pipeline.sample_and_perform_transformation(100, original_X, original_label)

```

	
Example 2:

```py
from dataset_augmenter.core import DataTransformerPipeline

def load_data()
	...

original_X, original_label = load_data()

pipeline = DataTransformerPipeline

pipeline.add_gaussian_noise_transformer(sigma=0.08)

# create 100 new examples by random sampling original data and applying gaussian noise.
transformed_X, transformed_label = pipeline.sample_and_perform_transformation(100, original_X, original_label)
```

#### 2. Using number\_generator package with augmented digits datasets:

If one has number\_generator package installed and want to generate number sequences or telephone numbers-like sequences using augmented datasets, its possible by augmenting a dataset, saving it and then configuring the default digits dataset to be used (see number\_generator README section 2.2 to learn how to set a custom digit dataset to be used for number generation).

Generating numbers with augmented digits datasets:

```py

from number_genrator import DigitImageDataset
from dataset_augmenter.core import DataTransformerPipeline


# load original digits dataset 
dataset = DigitImageDataset()
dataset.load('./data/dataset.pickle')

original_X, original_label = dataset.get_data()

pipeline = DataTransformerPipeline

pipeline.add_elastic_transformer(intensity=2, sigma=2)
pipeline.add_rotation_transformer(min_angle=-10, max_angle=10)

transformed_X, transformed_label = pipeline.sample_and_perform_transformation(100, original_X, original_label)

dataset.add_examples(transformed_X, transformed_label)
dataset.save('./data/augmented_digit_dataset.pickle')

```

Then set proper environment variable to point to new dataset:

```console
export NG_DEFAULT_DIGIT_DATASET='./mypath/my_digits_dataset.pickle'
```

Now its possible to generate phone numbers like datasets using the augmented digits dataset:

```py
generate-phone-numbers.py --num-images=200 --min-spacing=5 --max-spacing=10 --image-width=100 --output-path=./
```

Help
----------------------

Support can be provided through email: pablo.cael@gmail.com.

Executable scripts have help info can be access by using the ```--help``` 

Development
----------------------

###  Testing:

number-generator uses pytest. To run the tests, run:

```py
python3 -m pytest
```

on the root directory.


Future Improvement and Features
----------------------

- Add more data transformers such as zoom, translation or other non rigid deformations.
