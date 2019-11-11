""" setup Gap-ML
Copyright, 2018(c), Andrew Ferlitsch
Autor: David Molina @virtualdvid
"""

from setuptools import setup, find_packages

# setup components
with open('README.md', 'r', encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

INSTALL_REQUIRES = [
    'numpy',
    'h5py',
    'imutils',
    'requests',
    'opencv-python',
    'pillow',
    'tqdm',
    'mypy'
]

TESTS_REQUIRE = [
    'pytest',
    'pytest-cov',
    'pytest-mypy'
]

PACKAGE_DATA = {
    'gapcv': ['gapcv/*']
}

PROJECT_URLS = {
    'Documentation': 'https://gapml.github.io/CV/',
    'Source Code': 'https://github.com/gapml/CV'
}

# https://pypi.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS = [
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
    'Programming Language :: Python :: 3.7'
]

setup(
    name='gapcv',
    version='1.0rc4',
    description='CV Data Engineering Framework',
    author='Andrew Ferlitsch',
    author_email='aferlitsch@gmail.com',
    license='Apache 2.0',
    url='https://github.com/gapml/CV',
    project_urls=PROJECT_URLS,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests"
        ]
    ),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    package_data=PACKAGE_DATA,
    classifiers=CLASSIFIERS
)
