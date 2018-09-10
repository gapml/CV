# GapML CV  
# Computer Vision for Images

## Framework
The Gap NLP/CV open source framework provides an easy to get started into the world of machine learning for your unstructured data in PDF documents, scanned documents, TIFF facsimiles,  camera captured documents, and computer vision for your image data.

+	Automatic OCR of scanned and camera captured images.  
+	Automatic Text Extraction from documents.  
+	Automatic Syntax Analysis.  
+	Programmatic control for data extraction or redaction (de-identification)  
    -	Names, Addresses, Proper Places  
    -	Social Security Numbers, Data of Birth, Gender  
    -	Telephone Numbers  
    -	Numerical Information (e.g., medical, financial, …) and units of measurement.  
    -	Unit conversion from US Standard to Metric, and vice-versa  
    -	Unicode character recognition  
+	Machine Training of Document and Page Classification.  
+	Automatic image preparation (resizing, sampling) and storage (HD5) for convolutional neural networks.

The framework consists of a sequence of Python modules which can be retrofitted into a variety of configurations. The framework is designed to fit seamlessly and scale with an accompanying infrastructure. To achieve this, the design incorporates:

+	Problem and Modular Decomposition utilizing Object Oriented Programming Principles.  
+	Isolation of Operations and Parallel Execution utilizing Functional Programming Principles.  
+	High Performance utilizing Performance Optimized Python Structures and Libraries.  
+	High Reliability and Accuracy using Test Driven Development Methodology.

The framework provides the following pipeline of modules to support your data and knowledge extraction from both digital and scanned PDF documents, TIFF facsimiles and image captured documents, and for preparing and storing image data for computer vision.

This framework is ideal for any organization planning to do data extraction from their repository of documents into an RDBMS system for CART analysis or generating word vectors for natural language deep learning (DeepNLP), and/or computer vision with convolutional neural networks (CNN).

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