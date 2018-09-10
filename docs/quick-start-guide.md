# Natural Language Processing for PDF/TIFF/Image Documents - Computer Vision for Image Data

Users Guide  
High Precision Natural Language Processing for PDF/TIFF/Image Documents and Computer Vision for Images  
Users Guide, Gap v0.9.2

## 1 Introduction

The target audience for this users guide are your software developers whom will be integrating the core inner block into your product and/or service. It is not meant to be a complete reference guide or comprehensive tutorial, but a brief get started guide.

To utilize this module, the **Gap** framework will automatically install:

## 2 VISION Module
### 2.1 Image Processing

CV preprocessing of images requires the <b style='class:saddlebrown'>VISION</b> module.

To preprocess an image for computer vision machine learning, you create an `Image` (class) object, passing as parameters the path to the image, the corresponding label and a path for storing the preprocessed image data, the original image, optionally a thumbnail, and metadata. The label must be specified as an integer value. Below is a code example.

```python
from gapml.vision import Image
image = Image("yourimage.jpg", 101, "storage_path")
```

The above will generate the following output files:

    storage_path/yourimage.h5 # preprocessed image and raw data and optional thumbnail

Alternately, the image path may be an URL; in which case, an HTTP request is made to obtain the image data from the remote location. 

```python
image = Image("http://yourimage.jpg", 101, "storage_path")
```

The `Image` class supports processing of JPEG, PNG, TIF,  BMP and GIF images. Images maybe of any pixel size, and number of channels (i.e. Grayscale, RGB and RGBA).

Alternately, the input may be raw pixel data as a numpy array.

    raw = [...], [...], […] ]
    image = Image(raw, 101, "storage_path")

### 2.2 Image Processing Settings (Config)

CV Preprocessing of the image may be configured for several settings when instantiating an `Image` object with the optional `config` parameter, which consists of a list of one or more predefined options.

```python
image = Image("yourimage.jpg", 101, "storage_path", config=[options])
```

options:
    gray     | grayscale        # convert to grayscale (single channel)
    normal   | normalize        # normalize the pixel data for values between 0 .. 1
    flat     | flatten          # flatten the pixel data into a 1D vector
    resize=(height,width)       # resize the image
    thumb=(height,width)        # generate a thumbnail
    nostore	                    # do not store the preprocessed image, raw and thumbnail data

Example

```python
image = Image("image.jpg", 101, "path", config=['flatten', 'thumb=(16,16)'])
# will preprocess the image.jpg into machine learning ready data as a 1D vector, and
# store the raw (unprocessed) decompressed data, preprocessed data and 16 x 16 
```

### 2.3 Get Properties of Preprocessed Image Data

After an image has been preprocessed, several properties of the preprocessed image data can be obtained from the `Image` class properties:

```python
name	- The root name of the image.
type	- The image format (e.g., png).
shape	- The shape of the preprocessed image data (e.g., (100, 100,3) ).
data	- The preprocessed image data as a numpy array.
raw	- The unprocessed decompressed image data as a numpy array.
size	- The byte size of the original image.
thumb – The thumbnail image data as a numpy array.
```
```python
image = Image("yourimage.jpg", "storage_path", 101)
print(image.shape)
```

Will output something like:

    (100,100,3)

### 2.4 Asynchronous Processing

To enhance concurrent execution between a main thread and worker activities, the `Image` class supports asynchronous processing of the image. Asynchronous processing will occur if the optional parameter `ehandler` is set when instantiating the `Image` object. Upon completion of the processing, the `ehandler` is called, where the `Image` object is passed as a parameter.

```python
def done(i):
    """ Event Handler for when processing of image is completed """
    print("DONE", i.image)
# Process the image asynchronously
image = Image("yourimage.png", "storage_path", 101, ehandler=done)
```

### 2.5 Image Reloading

Once an `Image` object has been stored, it can later be retrieved from storage, reconstructing the `Image` object. An `Image` object is first instantiated, and then the `load()` method is called specifying the image name and corresponding storage path. The image name and storage path are used to identify and locate the corresponding stored image data.

```python
# Instantiate an Image object
image = Image()
# Reload the image's data from storage
image.load( "myimage.png", "mystorage" )
```

### 2.6 Image Collection Processing

To preprocess a collection of images for computer vision machine learning, you create an `Images` (class) object, passing as parameters a list of the paths to the images, a list of the corresponding label and a path for storing the collection of preprocessed image data, the original images and optionally thumbnails. Each label must be specified as an integer value. Below is a code example.

```python
from gapml.images import Images
images = Images(["image1.jpg", "image2.jpg"], labels=[101, 102], name=' c1')
```

The above will generate the following output files: 

    train/c1.h5 # preprocessed image data

The `Images` object will implicitly add the 'nostore' setting to the configuration parameter of each `Image` object created. This will direct each of the `Image` objects to not store the corresponding image data in an HD5 file. 

Instead, upon completion of the preprocessing of the collection of image data, the entire collection of preprocessed data is stored in a single HD5 file.

Alternately, the list of image paths parameter may be a list of directories containing images. 

```python
images = Images(["subfolder1", "subfolder2"], labels=[101, 102], name=' c1')
```

Alternately, the list of labels parameter may be a single value; in which case the label value applies to all the images. 

```python
images = Images(["image1.jpg", "image2.jpg"], labels=101, name=' c1') 
```

### 2.7 Image Collection Processing Settings (Config)

Configuration settings supported by the `Image` class may be specified as the optional `config` parameter to the `Images` object, which are then passed down to each `Image` object generated for the collection. 

```python
# Preprocess each image by normalizing the pixel data and then flatten into a 1D vector
images = Images(["image1.jpg", "image2.jpg"], "train", labels=[101, 102], config=['normal', 'flatten'])
```

### 2.8 Get Properties of a Collection

After a collection of images has been preprocessed, several properties of the preprocessed image data can be obtained from the `Images` class properties:

```python
name – The name of the collection file.
time – The time to preprocess the image.
data – List of Image objects in the collection.
len() – The len() operator will return the number of images in the collection.
[] – The index operator will access the image objects in sequential order.
```
```python
# Access each Image object in the collection
for ix in range(len(images)):
    image = images[ix]
```

### 2.9 Splitting a Collection into Training and Test Data

Batch, mini-batch and stochastic feed modes are supported. The percentage of data that is test (vs. training) is set by the `split` property, where the default is 0.2. Optionally, a mini-batch size is set by the `minibatch` property. Prior to the split, the data is randomized.

The `split` property when called as a getter will return the training data, training labels, test data, and test labels, where the data and labels are returned as numpy lists, and the labels have been one-hot encoded.

```python
# Set 30% of the images in the collection to be test data
images.split = 0.3

# Get the entire training and test data and corresponding labels as lists.
X_train, X_test, Y_train, Y_test = images.split
```

Alternately, the `next()` operator will iterate through the image data, and corresponding label, in the training set. 

```python
# Set 30% of the images in the collection to be test data
images.split = 0.3

# Iterate through the training data
while ( data, label = next(images) ) is not None:
    pass
```

Training data can also be fetched in minibatches. The mini batch size is set using the `minibatch` property. The `minibatch` property when called as a getter will return a generator. The generator will iterate through each image, and corresponding label, of the generated mini-batch. Successive calls to the `minibatch` property will iterate through the training data.

```python
# Set 30% of the images in the collection to be test data
images.split = 0.3

# Train the model in mini-batches of 30 images
images.minibatch = 30

# loop for each mini-batch in training data
for _ in range(nbatches)

# create the generator
g = images.minibatch

# iterate through the mini-batch
for data, label in g:
    pass
```

The `split` property when used as a setter may optionally take a seed for initializing the randomized shuffling of the training set.

```python
# Set the seed for the random shuffle to 42
images.split = 0.3, 42
```

### 2.10 Image Augmentation

Image augmentation is supported. By default, images are not augmented. If the property `augment` is set to `True`, then for each image generated for feeding (see `next()` and minibatch) an additional image will be generated. The additional image will be a randomized rotation between -90 and 90 degrees of the corresponding image. For example, if a training set has a 1000 images, then 2000 images will be feed when the property augment is set to True, where 1000 of the images are the original images, and another 1000 are the generated augmented images.

```python
images.split = 0.3, 42

# Enable image augmentation
images.augment = True

# Iterate through the training data, where every other image will be an augmented image
while ( data, label = next(images) ) is not None:
   pass
```

### 2.11 Asynchronous Collection Processing

To enhance concurrent execution between a main thread and worker activities, the `Images` class supports asynchronous processing of the collection of images. Asynchronous processing will occur if the optional parameter `ehandler` is set when instantiating the Images object. Upon completion of the processing, the ehandler is called, where the `Images` object is passed as a parameter.

```python
def done(i):
    """ Event Handler for when processing of collection of images is completed """
    print("DONE", i.images)

# Process the collection of images asynchronously
images = Images(["img1.png", "img2.png"], "train", labels=[0,1], ehandler=done)
```

### 2.12 Collection Reloading

Once an `Images` object has been stored, it can later be retrieved from storage, reconstructing the `Images` object, and corresponding list of `Image` objects. An `Image`s object is first instantiated, and then the `load()` method is called specifying the collection name and corresponding storage path. The collection name and storage path are used to identify and locate the corresponding stored image data.

```python
# Instantiate an Images object
images = Images()

# Reload the collection of image data from storage
images.load( "mycollection", "mystorage" )
```

Proprietary Information  
Copyright ©2018, Epipog, All Rights Reserved