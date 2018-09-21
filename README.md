# Gap : NLP/CV Data Engineering Framework

## Natural Language Processing for PDF, TIFF, and Camera Captured Documents, and
## Computer Vision Processing for Images

### Framework

The Gap NLP/CV data engineering framework provides an easy to get started into the world of machine learning for your unstructured data in PDF documents, scanned documents, TIFF facsimiles and camera captured documents, and your image data in image files and image repositories.

*CV , v0.9.4 (Pre-launch: beta)*

  - Automatic storage and retrieval with high performance HDF5 files.
  - Automatic handling of mixed channels (grayscale, RGB and RGBA) and pixel size.
  - Programmatic control of resizing.
  - Programmatic control of conversion into machine learning ready data format: decompression, normalize, flatten.
  - Programmatic control of minibatch generation.
  - Programmatic control of image augmentation.
  - Asynchronous processing of images.
  - Automatic generation of CV machine learning ready data.

The framework consists of a sequence of Python modules which can be retrofitted into a variety of configurations. The framework is designed to fit seamlessly and scale with an accompanying infrastructure. To achieve this, the design incorporates:

  - Problem and Modular Decomposition utilizing Object Oriented Programming Principles.
  - Isolation of Operations and Parallel Execution utilizing Functional Programming Principles.
  - High Performance utilizing Performance Optimized Python Structures and Libraries.
  - High Reliability and Accuracy using Test Driven Development Methodology.

## Audience

This framework is ideal for any organization planning to do:

  * Data extraction from their repository of documents into an RDBMS system for CART analysis, linear/logistic regressions,            
    or generating word vectors for natural language deep learning (DeepNLP).
  * Generating machine learning ready datan from their repository of images for computer vision.

## License

The source code is made available under the Apache 2.0 license: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Prerequites

The Gap framework extensively uses a number of open source applications/modules. The following applications and modules will be downloaded onto your computer/laptop when the package **OR** setup file is installed.

  1. Artifex's Ghostscript - extracting text from text PDF
  2. ImageMagic's Magick - extracting image from scanned PDF
  3. Google's Tesseract - OCR of scanned/image captured text
  4. NLTK (Natural Language Toolkit) - stemming/lemmatizer/parts of speech annotation
  5. unidecode - romanization of latin character codes
  6. numpy - high performance in-memory arrays (tensors)
  7. HDF5 - high performance of on-disk data (tensors) access
  8. openCV - image manipulation and processing for computer vision
  9. imutils - image manipulation for computer vision
  
## Pip Installation: 

The Gap framework is supported on Windows, MacOS, and Linux. It has been packaged for distribution via PyPi on launch.

  1. install [miniconda](https://conda.io/miniconda.html)

  2. (optional)  
      + Create an environment with: `conda create -n gap python==3.6 jupyter`  
      + Activate: `source activate gap`
      + Deactivate: `source deactivate`

  3. install GapML:  
      + `pip install gapcv`

## Setup.py Installation:

To install GapML via setup.py:

  1. clone from the Github repo.  
      + `git clone https://github.com/gapml/CV.git`

  2. install the GapML setup file. 
      + `python setup.py install`

## Modules

The framework provides the following pipeline of modules to support your data and knowledge extraction from both digital and scanned PDF documents, TIFF facsimiles and image captured documents.

#### <span style='color: saddlebrown'>VISION</span>

The splitter module is the CV entry point into the pipeline. It consists of a Images and Image class. The Images class handles the storage and (random access) batch retrieval of CV machine learning ready data, using open source openCV image processing, numpy high performance arrays (tensors) and HDF5 high performance disk (tensor) access. The Image class handles preprocessing of individual images into CV machine learning ready data. The batch and image preprocessing can be done synchronously or asynchronously, where in the latter case an event handler signals when the preprocessing of an image or batch has been completed and the machine learning ready data is accessible.

The vision module handles:

  - Mixed image size, format, resolution, number of channels
  - Decompression, Resizing, Normalizing, Flattening

[Specification](specs/vision_spec.docx)

## User's Guide

The User's (Programming) Quick Start Guide can be found [here](specs/quick%20start%20guide.docx)

## Releases


## Testing

The GAP framework is developed using Test Driven Development methodology. The automated unit tests for the framework use pytest, which is a xUnit style form of testing (e.g., jUnit, nUnit, jsUnit, etc).

#### Installation and Documentation

The pytest application can be installed using pip:

    pip install pytest

Online documentation for [pytest](https://docs.pytest.org)

#### Execution

The following are the pre-built automated unit tests, which are located under the subdirectory tests:

    image_test.py       # Tests the Image and Images Class in the Vision Module

The automated tests are executed as follows:

  1. From directory root enter `cd tests`

  2. To run tests with coverage: 

    pytest -v image_test.py

#### Code Coverage

Information on the percent of code that is covered (and what source lines not covered) by the automated tests is obtained using pytest-cov. This version of pytest is installed using pip:

    pip install pytest-cov

  1. From directory root enter `cd tests`

  2. To run tests with coverage:

Testing with code coverage is executed as follows:

    pytest --cov=gampml.vision image_test.py

        Statements=652, Missed=56, Percent Covered: 91%
