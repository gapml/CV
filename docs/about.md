# GapML CV  
# Computer Vision for Images

## Framework
The Gap CV open source framework provides an easy to get started into the world of machine learning for your computer vision for your image data.

+	Automatic image preparation (resizing, sampling) and storage (HD5) for convolutional neural networks.

The framework consists of a sequence of Python modules which can be retrofitted into a variety of configurations. The framework is designed to fit seamlessly and scale with an accompanying infrastructure. To achieve this, the design incorporates:

+	Problem and Modular Decomposition utilizing Object Oriented Programming Principles.  
+	Isolation of Operations and Parallel Execution utilizing Functional Programming Principles.  
+	High Performance utilizing Performance Optimized Python Structures and Libraries.  
+	High Reliability and Accuracy using Test Driven Development Methodology.

The framework provides the following pipeline of modules to support your data and knowledge extraction for preparing and storing image data for computer vision.

This framework is ideal for any organization planning to do data extraction from their repository of images for computer vision with convolutional neural networks (CNN).

## VISION

The vision module provides preprocessing and storage of images into machine learning ready data. The module supports a wide variety of formats: JPG, PNG, BMP, and TIF, and number of channels (grayscale, RGB, RGBA).

Images can be processed incrementally, or in batches.  Preprocessing options include conversion to grayscale, resizing, normalizing and flattening. The machine ready image data is stored and retrievable from high performance HD5 file.

The HD5 storage provides fast and random access to the machine ready image data and corresponding labels.

Preprocessing can be done either synchronously or asynchronously, where in the latter case an event handler signals when the preprocessing has been completed and the machine ready datta is accessible.

Further disclosure requires an Non-Disclosure Agreement.
 
## MODULES

  ![modules](img/modules.png)

Proprietary and Confidential Information  
Copyright ©2018, Epipog, All Rights Reserved