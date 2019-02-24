""" setup Gap-ML
Copyright, 2018(c), Andrew Ferlitsch
Autor: David Molina @virtualdvid
"""

from setuptools import setup, find_packages

# setup components
with open('README.md', 'r', encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    'numpy',
    'h5py',
    'imutils',
    'requests',
    'opencv-python',
    'Pillow']

tests_require = [
    'pytest',
    'pytest-cov']

package_data = {'gapcv': ['train/*']}

project_urls = {'Documentation': 'https://gapml.github.io/CV/',
                'Source Code': 'https://github.com/gapml/CV'}

# https://pypi.org/pypi?%3Aaction=list_classifiers
classifiers = [
    'Development Status :: 3 - Alpha',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: MacOS',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.0',
    'Programming Language :: Python :: 3.1',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7']

setup(
    name='gapcv',
    version='0.9.8',
    description='NLP and CV Data Engineering Framework',
    author='Andrew Ferlitsch',
    author_email='aferlitsch@gmail.com',
    license='Apache 2.0',
    url='https://github.com/gapml/CV',
    project_urls=project_urls,
    long_description=long_description,
    packages=find_packages(exclude=["*.tests",
                                    "*.tests.*",
                                    "tests.*",
                                    "tests"]),
    install_requires=install_requires,
    tests_require=tests_require,
    package_data=package_data,
    classifiers=classifiers
)
