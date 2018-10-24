"""
Image Data Processing
Copyright 2018(c), GapML
"""

import os
import time
import cv2

import numpy as np

# Import requests for HTTP request/response handling
import requests

# Import pillow for Python image manipulation for GIF and JP2K
from PIL import Image as PILImage

def processImage(image, resize, flatten=False):
    """ Lowest level processing of an raw pixel data.
        :param image  : raw pixel data as 2D (grayscale) or 3D (color) matrix.
        :type  image  : numpy matrix
        :param resize : scale (downsample or upsample) image to a specifid height, width.
        :type  resize : tuple(height, width)
        :param flatten: wether to flatten the image into a 1D vector or not.
        :type  flatten: bool
        :return       : a processed image as a numpy matrix (or vector if flattened).
    """

    # resize each image to the target size (e.g., 50x50) and flatten into 1D vector
    if flatten:
        return cv2.resize(image, resize, interpolation=cv2.INTER_AREA).flatten()
    # resize each image to the target size (e.g., 50x50)
    else:
        return cv2.resize(image, resize, interpolation=cv2.INTER_AREA)

GRAYSCALE = cv2.IMREAD_GRAYSCALE
COLOR     = cv2.IMREAD_COLOR

def loadImageDisk(file, colorspace=COLOR, resize=(128, 128), flatten=False):
    """ Loads an image from disk
        :param file      : a file path to an image.
        :type  file      : string
        :param colorspace: the colorspace (grayscale or color).
        :type  colorspace: int
        :param resize    : scale (downsample or upsample) image to a specifid height, width.
        :type  resize    : tuple(height, width)
        :param flatten   : whether to flatten the image into a 1D vector or not.
        :type  flatten   : bool
        :return          : a processed image as a numpy matrix (or vector if flattened).
    """
    # retain original file information
    basename = os.path.splitext(os.path.basename(file))
    name = basename[0]
    type = basename[1][1:].lower()
    size = os.path.getsize(file)

    try:
        if type in [ 'gif', 'jp2k']:
            image = PILImage.open(file)
            if colorspace == GRAYSCALE:
                image = image.convert('L')
            else:
                image = image.convert('RGB')
            image = np.array(image)
        else:
            image = cv2.imread(file, colorspace)
    except Exception as e:
        return None, None, None, None, e
    shape = image.shape
    # retain the original shape
    shape = image.shape

    try:
        image = processImage(image, resize, flatten)
        return image, shape, size, name, type, None
    except Exception as e:
        return None, None, None, None, e

def loadImageRemote(url, colorspace=COLOR, resize=(128, 128), flatten=False):
    """ Loads an image from a remote location
        :param: image    : the image as raw pixel data as numpy matrix.
        :type   image    : numpy matrix
        :param colorspace: the colorspace (grayscale or color).
        :type  colorspace: int
        :param resize    : scale (downsample or upsample) image to a specifid height, width.
        :type  resize    : tuple(height, width)
        :param flatten   : whether to flatten the image into a 1D vector or not.
        :type  flatten   : bool
        :return          : a processed image as a numpy matrix (or vector if flattened).
    """
    try:
        response = requests.get(url, timeout=10)
    except Exception as e:
        return None, None, None, None, e

    # read in the image data
    data = np.fromstring(response.content, np.uint8)
    # retain the original size
    size = len(data)
    # retain original image information
    name = ''
    type = ''

    # decode the image
    try:
        image = cv2.imdecode(data, colorspace)
    except Exception as e:
        return None, None, None, None, e

    # retain the original shape
    shape = image.shape

    try:
        image = processImage(image, resize, flatten)
        return image, shape, size, name, type, None
    except Exception as e:
        return None, None, None, None, e

def loadImageMemory(image, colorspace=COLOR, resize=(128, 128), flatten=False):
    """ Loads an image from memory
        :param: image     : the image as raw pixel data as numpy matrix.
        :param colorspace: the colorspace (grayscale or color).
        :type  colorspace: int
        :param resize    : scale (downsample or upsample) image to a specifid height, width.
        :type  resize    : tuple(height, width)
        :param flatten   : whether to flatten the image into a 1D vector or not.
        :type  flatten   : bool
        :return           : a processed image as a numpy matrix (or vector if flattened).
    """
    # retain the original shape
    shape = image.shape
    # retain original image information
    name = ''
    type = ''
    # assume each element is a float32
    size = shape[0] * 4
    if len(shape) > 1:
        size *= shape[1] * 4
    if len(shape) > 1:
        size *= shape[2] * 4

    # Special Case: Image already flattened
    if len(image.shape) == 1:
        return image, shape, size, name, type, ''

    #  Grayscale conversion
    if colorspace == GRAYSCALE:
        # single channel (assume grayscale)
        if len(shape) == 2:
            pass
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Color Conversion
    elif colorspace == COLOR:
        if len(shape) == 3:
            pass
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    try:
        image = processImage(image, resize, flatten)
        return image, shape, size, name, type, None
    except Exception as e:
        return None, None, None, None, e

NORMAL_0_1  = 0  # normalization between 0 and 1
NORMAL_N1_1 = 1  # normalization between -1 and 1
STANDARD    = 2  # standardization with mean at 0

def loadImages(files, colorspace=COLOR, resize=(128, 128), flatten=False, normal=NORMAL_0_1, datatype=np.float32):
    """ Load a collection of images
        :param files: list of file paths of images on-disk, or remote images, or in-memory images.
        :type  files: list[strings] or numpy array of matrixes
        :param colorspace: the colorspace (grayscale or color).
        :type  colorspace: int
        :param resize    : scale (downsample or upsample) image to a specifid height, width.
        :type  resize    : tuple(height, width)
        :param flatten   : whether to flatten the image into a 1D vector or not.
        :type  flatten   : bool
        :param normal    : method to normalize pixels
        :type  normal    : int
        :param datatype  : the data type of the pixel data after processed into machine learning ready data
        :param type      : type
        :return          : tuple(machine learning ready data, image names, image types, image shapes, image sizes, processing errors, processing time)
    """
    start_time = time.time()

    collection = []
    names = []
    types = []
    sizes = []
    shapes = []
    errors = []

    # Config Dictionary
    config = {'colorspace': colorspace,
              'resize'    : resize,
              'flatten'   : flatten}

    # Functions Dictionary
    switcher = {0: loadImageRemote,
                1: loadImageDisk,
                2: loadImageMemory}

    # List of files on Disk
    if isinstance(files[0], str):
        if files[0].startswith('http'):
            argument = 0
        else:
            argument = 1
    # List of matrixes in memory
    elif isinstance(files[0], np.ndarray):
        argument = 2

    # Get the function from switcher dictionary
    func = switcher.get(argument)
    # Execute the function

    for item in files:
        image, shape, size, name, _type, error = func(item, **config)
        # append each in-memory image into a list
        collection.append(image)
        # append the metadata for each image into a list
        names.append(bytes(name, 'utf-8'))
        types.append(bytes(_type, 'utf-8'))
        sizes.append(size)
        shapes.append(shape)
        errors.append(error)

    # once the collection is assembled, convert the list to a multi-dimensional numpy array
    collection = np.asarray(collection).astype(datatype)

    # Do not normalize if requesting to keep data in original integer bits per pixel (bpp)
    if datatype not in [np.uint8, np.int8, np.uint16, np.int16]:
        # color, unflatten
        if len(collection.shape) == 4:
            bpp = type(collection[0][0][0][0])
        # grayscale, unflatten
        elif len(collection.shape) == 3:
            bpp = type(collection[0][0][0])
        # flatten
        else:
            bpp = type(collection[0][0])

        # original pixel data is 8 bits per pixel
        if bpp in [np.uint8, np.int8]:
            if normal == NORMAL_0_1:
                collection /= 255.0
            elif normal == NORMAL_N1_1:
                collection = collection / 127.5 - 1
            elif normal == STANDARD:
                collection = (collection - np.mean(collection)) / np.std(collection)
        # original pixel data is 16 pixel
        elif bpp in [np.uint16, np.int16]:
            if normal == NORMAL_0_1:
                collection /= 65535.0
            elif normal == NORMAL_N1_1:
                collection = collection / 32767.5 - 1
            elif normal == STANDARD:
                collection = (collection - np.mean(collection)) / np.std(collection)

    return  collection, names, types, sizes, shapes, errors, time.time() - start_time

def loadDirectory(dir, colorspace=COLOR, resize=(128, 128), flatten=False, normal=NORMAL_0_1, datatype=np.float32):
    """ Load a Directory based dataset, where the toplevel subdirectories are the classes.
    :param files: list of file paths of images on-disk, or remote images, or in-memory images.
    :type  files: list[strings] or numpy array of matrixes
    :param colorspace: the colorspace (grayscale or color).
    :type  colorspace: int
    :param resize    : scale (downsample or upsample) image to a specifid height, width.
    :type  resize    : tuple(height, width)
    :param flatten   : whether to flatten the image into a 1D vector or not.
    :type  flatten   : bool
    :param normal    : method to normalize pixels
    :type  normal    : int
    :param datatype  : the data type of the pixel data after processed into machine learning ready data
    :param type      : type
    :return          : TODO
    """

    collections = []
    labels = []
    names = []
    types = []
    sizes = []
    shapes = []
    err = []
    total_elapsed = 0

    # Get the list (generator) of the subdirectories of the parent directory of the dataset
    subdirs = os.scandir(dir)
    os.chdir(dir)
    for subdir in subdirs:
        # skip entries that are not subdirectories or hidden directories (start with dot)
        if not subdir.is_dir() or subdir.name[0] == '.':
            continue
        files = os.listdir(subdir.name)
        os.chdir(subdir.name)
        collection, names, types, sizes, shapes, err, elapsed = loadImages(files, colorspace, resize, flatten, normal, datatype)
        os.chdir('..')
        collections.append(collection)
        labels.append(subdir)
    os.chdir('..')

    return collections, labels
