# Gap CV

<p align="center">
  <img width="345" height="200" src='docs/img/gap.jpg'>
</p>

## Intro

**Gap** is a data engineering framework for machine learning. The **GapCV** is a component of **Gap** for computer vision (CV). The component manages data preparation of images, feeding and serving neural network models, and data management of persistent storage.

The module is written in a modern *object oriented programming (OOP)* abstraction with an *imperative programming* style that fits seamlessly into ML frameworks which are moving into imperative programming, such as **Keras** and **PyTorch**. The bottom layers of the module are written in a *bare metal* style for high performance.

**Gap** was inspired by a meeting of data scientists and machine learning enthusiasts in Portland, OR in May 2018. The first version of **Gap** was available during the summer and the local software community was engaged through meetups, chats, Kaggle groups, and a conference at the Oxford Suites. During the Fall, a decision was made to refactor **Gap** into an industrial grade application, spearheaded by Gap's lead, David Molina, and overseen and assisted by Andrew Ferlitsch, Google AI. 

## Why and Why Now

During the Spring of 2018, many of us had observed advancements in redesign of ML frameworks (such as Keras and PyTorch) to migrate into frameworks which would have broader adoption in the software engineering community. But, the focus was primarily on the model building and not on the data engineering. Across the Internet, between blogs, tutorials and online classes, the examples for data engineering was still a wild west. To us, we saw this as a gap, and hence the name Gap

ML practitioners today recognize the substantial component that data engineering is within the machine learning ecosystem, and the need to modernize, streamline and standardize to meet the needs of the software development community at the same pace as framework advancements are being made on the modeling components.

  ![MLEcoSystem](docs/img/MLEcoSystem.png)

## Summary of Features

### Image Types

The following image formats are supported: 

    * JPEG and JPEG2000
    * PNG
    * TIF
    * GIF
    * BMP
    * 8 and 16 bits per pixel
    * Grayscale (single channel), RGB (three channels) and RGBA (four channels)
    
### Image Set (Dataset) Layouts

The following image set layouts are supported (i.e., can be ingested by Gap):  

    * On-Disk (directory, CSV and JSON)
    * In-Memory (list, numpy)
    * Remote (http)
    
For CSV and JSON, the image data can be embedded (in-memory), local (on-disk) or url paths (remote).

### Image Transformations

The following image transformations are supported: 

    * Color -> Gray
    * Resizing
    * Flattening
    * Normalization and Standardization
    * Data Type Conversion: 8 and 16 bpp integer and 16, 32 and 64 bpp float
    
Transformations can be performed when processing an image set (ingestion) or performed dynamically when feed during training.

### Image Augmentation

The following image augmentations are supported.

    * Rotation
    * Horizontal and Vertical Flip
    * Zoom
    * Brightening
    * Sharpening
    
Image augmentation can be performed dynamically in-place during feeding (training) of a neural network.

### Image Feeding

The following image feeding mechanisms are supported:

    * Splitting 
    * Shuffling
    * Iterative
    * Generative
    * Mini-batch
    * Stratification
    * One-Hot Label Encoding

When feeding, shuffling is handled using indirect indexing, maintaining the location of data in the heap. One-hot encoding of labels is performed
dynamically when the feeder is instantiated.

### In-Memory Management

The following are supported for in-memory management:

    * Contiguous Memory (Numpy)
    * Streaming
    * Indirect Indexing (Shuffling w/o moving memory)
    * Data Type Reduction
    * Collection Merging
    * Asynchronous and Concurrent Processing
    
Collections of image data, which are otherwise disjoint, can be merged efficiently and with label one-hot encoding performed dynamically for feeding neural networks from otherwise disparate sources.

### Persistent Storage Management

The following are supported for on-disk management:

    * HDF5 Storage and Indexing
    * Metadata handling
    * Distributed

## Pip Installation: 

The **GapCV** framework is supported on Windows, MacOS, and Linux. It has been packaged for distribution via PyPi on launch.

  1. install [miniconda](https://conda.io/miniconda.html)  

  2. install conda virtual environment and required packages  
      + Create an environment with: `conda create -n gap python==3.5 jupyter pip`  
      + Activate: `source activate gap`  
      + `pip install gapcv`

  3. exiting conda virtual environment:  
      + Windows: `deactivate`  
      + Linux/macOS: `source deactivate`

## Setup.py Installation:

To install **GapCV** via setup.py:

  1. clone from the Github repo.  
      + `git clone https://github.com/gapml/CV.git`

  2. install the GapML setup file.  
      + access folder `cd CV`  
      + `python setup.py install`

## Quick Start

Image preparation, neural network feeding and management of image datasets is handled through the class object `Images`. We will provide here a brief discussion on the
various ways of using the `Images` class.

The initializer has no required (positional) parameters. All the parameters are optional (keyword) parameters. The most frequently used parameters are:

        Images( name, dataset, labels, config ) 
        
            name   : the name of the dataset (e.g., 'cats_n_dogs')
            dataset: the dataset of images
            labels : the labels
            config : configuration settings

### Preparing Datasets

The first step is to transform the images in an image dataset into machine learning ready data. How the images are transformed is dependent on the image source and the configuration settings. By default, all images are transformed to:

        1. RGB image format
        2. Resized to (128, 128)
        3. Float32 pixel data type
        4. Normalization
        
In this quick start section, we will briefly cover preparing datasets that are on-disk, remotely stored and in-memory.
 
*Directory*

A common format for image datasets is to stored them on disk in a directory layout. The layout consists of a root (parent) directory and one or more subdirectories. Each subdirectory is a
class (label), such as *cats*. Within the subdirectory are one or more images which belong to that class. Below is an example:

                    cats_n_dogs
                  /             \  
                cats            dogs
                /                  \
            c1.jpg ...          d1.jpg ...
            
The following instantiation of the `Images` class object will load the images from local disk into in-memory according the default transformation settings.  Within memory, the set of transformed images will be grouped into two classes: cats, and dogs.      

```python
images = Images(dataset='cats_n_dogs')
```

Once loaded, you can get information on the transformed data as properties of the `Images` class. Below are a few frequently used properties.

```python
print(images.name)      # will output the name of the dataset: cats_and_dogs
print(images.count)     # will output the total number of images in both cats and dogs
print(images.classes)   # will output the class to label mapping: { 'cats': 0, 'dogs': 1 }
print(images.images[0]) # will output the numpy arrays for each transformed image in the class with label 0 (cats).
print(images.labels[0]) # will output the label for each transformed image in the class with label 0 (cats).
```

Several of the builtin functions have been overridden for the `Images` class. Below are a few frequently used overriden builtin functions:

```python
print(len(images))      # same as images.count
print(images[0])        # same as images.images[0]
```

*List*

Alternatively, local on-disk images maybe specified as a list of paths, with corresponding list of labels. Below is an example where the `dataset` parameter is specified as a list of
paths to images, and the `labels` parameter is a list of corresponding labels.

```python
images = Images(name='cats_and_dogs', dataset=['cats/1.jpg', 'cats/2.jpg', ... 'dogs/1.jpg'], labels=[0, 0, ... 1])
```

Alternately, the image paths maybe specified as remote locations using URL paths. In this case, a HTTP request will be made to fetch the contents of the image from the remote site.

```python
images = Images(name='cats_and_dogs', dataset=['http://mysite.com/cats/1.jpg', 'http://mysite.com/cats/2.jpg', ... ], labels=[0, 0, ...])
```

*Memory*

If the dataset is already in memory, for example a curated dataset that is part of a framework (e.g., CIFAR-10 in Keras), the in-memory multi-dimensional numpy arrays for the curated images and labels are passed as the values to the `dataset` and `labels` parameter.

```python
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train = Images('cifar10', dataset=x_train, labels=y_train)
test  = Images('cifar10', dataset=x_test,  labels=y_test)
```

*CSV*

A dataset can be specified as a CSV (comma separated values) file. Both US (comma) and EU (semi-colon) standard for separators are supported. Each row in the CSV file corresponds to an image
and corresponding label. The image may be local on-disk, remote or embedded. Below are some example CSV layouts:

        *local on-disk*
            label,image
            'cat','cats/c1.jpg'
            'dog','dogs/d1.jpg'
            ...
            
        *remote*
            label,image
            'cat','http://mysite.com/c1.jpg'
            'dog','http://mysite.com/d1.jpg'
            ...
            
        *embedded pixel data*
            label,name
            'cat','[ embedded pixel data ]'
            'dog','[ embedded pixel data ]'
            
For CSV, the `config` parameter is specified when instantiating the `Images` class object, to set the settings for:

        header      # if present, CSV file has a header; otherwise it does not.
        image_col   # the column index (starting at 0) of the image field.
        label_col   # the column index (starting at 0) of the label field.
        
```python
images = Images(dataset='cats_n_dogs.csv', config=['header', 'image_col=0', 'label_col=1'])
```

*JSON*

### Feeding Datasets

### Managing Datasets (Persistent Storage)

## Reference

## Testing
