"""
Image Data Processing
Copyright 2018(c), GapML
"""

import os
import time
import datetime
import csv
import json
import cv2
import h5py

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

    # resize each image to the target size (e.g., 50x50)
    image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    # flatten into 1D vector
    if flatten:
        image = image.flatten()

    return image

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
        if type in ['gif', 'jp2k']:
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

def loadImages(files, colorspace=COLOR, resize=(128, 128), 
               flatten=False, normal=NORMAL_0_1, datatype=np.float32):
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
        :param datatype  : the data type of the pixel data after processed into ML ready data
        :param type      : type
        :return          : tuple(machine learning ready data, image names, image types, 
                                 image shapes, image sizes, processing errors, processing time)
    """
    start_time = time.time()

    collection = []
    names = []
    types = []
    sizes = []
    shapes = []
    errors = []

    for item in files:
        if isinstance(files[0], str):
            if files[0].startswith('http'):
                image, shape, size, name, _type, error = loadImageRemote(item, colorspace, resize, flatten)
            else:
                image, shape, size, name, _type, error = loadImageDisk(item, colorspace, resize, flatten)
        elif isinstance(files[0], np.ndarray):
            image, shape, size, name, _type, error = loadImageMemory(item, colorspace, resize, flatten)
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
        collect_type = {4:         type(collection[0][0][0][0]), # color, unflatten
                        3:         type(collection[0][0][0]),    # grayscale, unflatten
                        'flatten': type(collection[0][0])}       # flatten
        
        len_collect_shape = len(collection.shape)
        if len_collect_shape not in [4, 3]:
            len_collect_shape = 'flatten'

        bpp = collect_type[len_collect_shape].__name__

        normalize = {'int8': {0: (lambda collection: collection / 255.0), # original pixel data is 8 bits per pixel
                              1: (lambda collection: collection / 127.5 - 1),
                              2: (lambda collection: (collection - np.mean(collection)) / np.std(collection))},
                     'int6': {0: (lambda collection: collection / 65535.0), # original pixel data is 16 bits per pixel
                              1: (lambda collection: collection / 32767.5 - 1),
                              2: (lambda collection: (collection - np.mean(collection)) / np.std(collection))}}

        if bpp in ['uint8', 'int8', 'uint16', 'int16']:
            i = 4
            if bpp in ['uint16', 'int16']:
                i += 1
            bpp = bpp[-i:]
            collection = normalize[bpp][normal](collection)

    return collection, names, types, sizes, shapes, errors, time.time() - start_time

def createH5(dir, author, source, description, date, labels,
             total_elapsed, colorspace, resize, datatype, n_images,
             label, collection, n_label, elapsed, names, types, shapes, sizes):
    """ Create h5 file
        :param dir:
        :param author:
        :param source:
        :param description:
        :param date:
        :param labels:
        :param total_elapsed:
        :param colorspace:
        :param resize:
        :param datatype:
        :param n_images:
        :param label:
        :param collection:
        :param n_label:
        :param elapsed:
        :param names:
        :param types:
        :param shapes:
        :param sizes:
    """
    # Create the HDF5 file
    with h5py.File('../' + dir + '.h5', 'a') as hf:
        # Global Atttributes
        hf.attrs['name'] = dir
        hf.attrs['author'] = author
        hf.attrs['source'] = source
        hf.attrs['description'] = description
        hf.attrs['date'] = date
        hf.attrs['labels'] = labels
        hf.attrs['total_elapsed'] = total_elapsed
        hf.attrs['shape'] = resize
        hf.attrs['pixel_data_type'] = str(datatype.__name__)
        if colorspace == COLOR:
            hf.attrs['channel'] = str(['R', 'G', 'B'])
        else:
            hf.attrs['channel'] = str(['K'])
        hf.attrs['count'] = n_images

        # Create groups and attributes
        group = hf.create_group(label)
        group.attrs['class'] = label
        group.attrs['count'] = len(collection)
        group.attrs['label'] = n_label
        group.attrs['time'] = elapsed

        # Create dataset and attributes
        dset = group.create_dataset("data", data=collection)
        dset.attrs['names'] = names
        dset.attrs['types'] = types
        dset.attrs['shapes'] = shapes
        dset.attrs['sizes'] = sizes

def loadDirectory(dir, colorspace=COLOR, resize=(128, 128), flatten=False,
                  normal=NORMAL_0_1, datatype=np.float32, storage=True):
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
    :param datatype  : the data type of the pixel data after processed into ML ready data
    :param type      : type
    :return          : TODO
    """

    collections = []
    labels = []
    err = []
    total_elapsed = 0
    n_images = 0 # total number of images

    if os.path.isfile(dir + '.h5'):
        os.remove(dir + '.h5')

    # Get the list (generator) of the subdirectories of the parent directory of the dataset
    subdirs = os.scandir(dir)
    os.chdir(dir)

    for subdir in subdirs:
        # skip entries that are not subdirectories or hidden directories (start with dot)
        if not subdir.is_dir() or subdir.name[0] == '.':
            continue
        files = os.listdir(subdir.name)
        os.chdir(subdir.name)
        collection, names, types, sizes, shapes, errors, elapsed = loadImages(files, colorspace, resize, flatten, normal, datatype)
        os.chdir('..')
        collections.append(collection)
        label = bytes(subdir.name, 'utf-8')
        labels.append(label)
        err.append(errors)

        # increment the mapping of class name to label
        n_label = labels.index(label)

        # maintain count of total images
        n_images += len(collection)

        # total time elapsed
        total_elapsed += elapsed

        # storage
        if storage:
            createH5(dir=dir,
                     author='ABC Company',
                     source='http://....',
                     description='blah blah',
                     date=str(datetime.datetime.now()),
                     labels=labels,
                     total_elapsed=total_elapsed,
                     colorspace=colorspace,
                     resize=resize,
                     datatype=datatype,
                     n_images=n_images,
                     label=subdir.name,
                     collection=collection,
                     n_label=n_label,
                     elapsed=elapsed,
                     names=names,
                     types=types,
                     shapes=shapes,
                     sizes=sizes)

    os.chdir('..')

    return collections, labels

def loadCSV(csv, header, image_col, label_col, colorspace=COLOR, resize=(128, 128),
            flatten=False, normal=NORMAL_0_1, datatype=np.float32):
    """ Read a dataset from a CSV file
    :param csv      : the CSV file
    :type  csv      : str
    :param header   : whether the file has a header
    :type  header   : bool
    :param image_col: the column of the image location
    :param label_col: the column of the image label
    """
    with open(csv, newline='') as csvf:
        reader = csv.reader(csvf, delimiter=',')
        for row in reader:
            image = row[image_col]
            label = row[label_col]

    # TODO

def loadJSON(json, image_key, label_key, colorspace=COLOR, resize=(128, 128),
             flatten=False, normal=NORMAL_0_1, datatype=np.float32):
    """ Read a dataset from a JSON file
    :param json     : the JSON file
    :type  json     : str
    :param image_key: the key name of the image location
    :param label_key: the key name of the image label
    """
    with open(json) as jsonf:
        data = json.loads(jsonf)
    pass

def loadMemory():
    pass
