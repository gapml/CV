"""
Copyright(c), Google, LLC (Andrew Ferlitsch)
Copyright(c), Virtualdvid (David Molina)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import time
import unittest
from shutil import copy, rmtree
import cv2
import numpy as np
import pytest
from gapcv.vision import Image, Images

class MyTest(unittest.TestCase):
    """ My Test """

    def setup_class(self):
        """ Setup Class"""
        pass

    def teardown_class(self):
        """ Teardown Class """
        pass

    ### Images

    def test_001(self):
        """ Images Constructor - no arguments """
        images = Images()
        self.assertEqual(images.images, None)
        self.assertEqual(images.labels, None)
        self.assertEqual(len(images), 0)
        self.assertEqual(images.count, 0)
        self.assertEqual(images.dir, './')
        self.assertEqual(images.name, 'unnamed')
        self.assertEqual(images.time, 0)
        self.assertEqual(images.elapsed, "00:00:00")
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.dtype, np.float32)
        self.assertEqual(images.shape, (0,))
        self.assertEqual(images.classes, None)
        self.assertEqual(images.author, '')
        self.assertEqual(images.src, '')
        self.assertEqual(images.desc, '')
        self.assertEqual(images.license, '')

    def test_002(self):
        """ Images Constructor - no images, config/store """
        images = Images(config=['store'])

    def test_003(self):
        """ Images Constructor - no images, _dir argument """
        images = Images(_dir='./tmp')
        self.assertEqual(images.dir, './tmp/')
        with pytest.raises(TypeError):
            images = Images(_dir=2)

    def test_004(self):
        """ Images Constructor - no images, _dir property """
        images = Images()
        images.dir = './tmp'
        self.assertEqual(images.dir, './tmp/')
        images.dir = './tmp2/'
        self.assertEqual(images.dir, './tmp2/')
        with pytest.raises(TypeError):
            images.dir = 2

        rmtree('tmp')
        rmtree('tmp2')

    def test_005(self):
        """ Images Constructor - no images, name argument """
        images = Images(name='foo')
        self.assertEqual(images.name, 'foo')
        with pytest.raises(TypeError):
            images = Images(name=2)

    def test_006(self):
        """ Images Constructor - no images, name property """
        images = Images()
        images.name = 'foo'
        self.assertEqual(images.name, 'foo')
        with pytest.raises(TypeError):
            images.name = 2

    def test_007(self):
        """ Images Constructor - no images, config argument """
        images = Images(config=None)
        images = Images(config=[])
        with pytest.raises(TypeError):
            images = Images(config=7)
        with pytest.raises(TypeError):
            images = Images(config='7')
        with pytest.raises(AttributeError):
            images = Images(config=[7])
        with pytest.raises(AttributeError):
            images = Images(config=['foo'])
        images = Images(config=['flat'])
        images = Images(config=['flatten'])
        images = Images(config=['gray'])
        images = Images(config=['grayscale'])
        images = Images(config=['store'])
        images = Images(config=['stream'])
        images = Images(config=['header'])
        images = Images(config=['uint8'])
        self.assertEqual(images.dtype, np.uint8)
        images = Images(config=['uint16'])
        self.assertEqual(images.dtype, np.uint16)
        images = Images(config=['float16'])
        self.assertEqual(images.dtype, np.float16)
        images = Images(config=['float32'])
        self.assertEqual(images.dtype, np.float32)
        images = Images(config=['float64'])
        self.assertEqual(images.dtype, np.float64)
        images = Images(config=['license='])
        self.assertEqual(images.license, '')
        images = Images(config=['src='])
        self.assertEqual(images.src, '')
        images = Images(config=['author='])
        self.assertEqual(images.author, '')
        images = Images(config=['desc='])
        self.assertEqual(images.desc, '')
        with pytest.raises(AttributeError):
            images = Images(config=['resize='])
        with pytest.raises(AttributeError):
            images = Images(config=['resize=()'])
        with pytest.raises(AttributeError):
            images = Images(config=['resize=(1,)'])
        with pytest.raises(AttributeError):
            images = Images(config=['resize=(0,3)'])
        with pytest.raises(AttributeError):
            images = Images(config=['resize=(3,0)'])
        with pytest.raises(AttributeError):
            images = Images(config=['resize=(-1,3)'])
        with pytest.raises(AttributeError):
            images = Images(config=['resize=(3,-1)'])
        images = Images(config=['resize=(10,20)'])
        with pytest.raises(AttributeError):
            images = Images(config=['norm='])
        with pytest.raises(AttributeError):
            images = Images(config=['normalization='])
        with pytest.raises(AttributeError):
            images = Images(config=['norm=2'])
        images = Images(config=['norm=pos'])
        images = Images(config=['norm=zero'])
        images = Images(config=['norm=std'])
        with pytest.raises(AttributeError):
            images = Images(config=['image_col='])
        with pytest.raises(AttributeError):
            images = Images(config=['image_col=A'])
        with pytest.raises(AttributeError):
            images = Images(config=['image_col=-1'])
        images = Images(config=['image_col=0'])
        with pytest.raises(AttributeError):
            images = Images(config=['label_col='])
        with pytest.raises(AttributeError):
            images = Images(config=['label_col=A'])
        with pytest.raises(AttributeError):
            images = Images(config=['label_col=-1'])
        images = Images(config=['label_col=0'])
        with pytest.raises(AttributeError):
            images = Images(config=['sep='])
        images = Images(config=['sep=A'])
        with pytest.raises(AttributeError):
            images = Images(config=['image_key='])
        images = Images(config=['image_key=A'])
        with pytest.raises(AttributeError):
            images = Images(config=['label_key='])
        images = Images(config=['label_key=B'])

    def test_008(self):
        """ Images Constructor - no images, labels argument """
        images = Images(labels=1)
        images = Images(labels=[1])
        images = Images(labels=np.asarray([1]))
        images = Images(labels='cats')
        images = Images(labels=['cats', 'dogs'])
        with pytest.raises(TypeError):
            images = Images(labels=3.2)
        with pytest.raises(AttributeError):
            images = Images(labels=[])
        with pytest.raises(TypeError):
            images = Images(labels=[3.2])
        with pytest.raises(TypeError):
            images = Images(labels=np.asarray([1.6]))

    def dummy(self):
        """ Dummy """
        pass

    def test_009(self):
        """ Images Constructor - no images, ehandler argument """
        images = Images(ehandler=self.dummy)
        images = Images(ehandler=(self.dummy, 6))
        with pytest.raises(TypeError):
            images = Images(ehandler=1)
        with pytest.raises(TypeError):
            images = Images(ehandler=(1, 2))

    def test_010(self):
        """ Images - directory - bad arguments """
        with pytest.raises(OSError):
            images = Images('foo', 'noexist_dir')
        with pytest.raises(OSError):
            images = Images('foo', 'func_test.py')

    def test_011(self):
        """ Images - CSV - bad arguments """
        with pytest.raises(OSError):
            images = Images('foo', 'noexist.csv', config=['image_col=0', 'label_col=0'])
        with pytest.raises(OSError):
            images = Images('foo', 'http://noexist.csv', config=['image_col=0', 'label_col=0'])
        with pytest.raises(OSError):
            images = Images('foo', 'https://noexist.csv', config=['image_col=0', 'label_col=0'])

        f = open('files/empty.csv', 'w')
        f.close()
        with pytest.raises(AttributeError):
            images = Images('foo', 'files/empty.csv', config=['image_col=-1', 'label_col=0'])
        with pytest.raises(AttributeError):
            images = Images('foo', 'files/empty.csv', config=['image_col=A', 'label_col=0'])
        with pytest.raises(AttributeError):
            images = Images('foo', 'files/empty.csv', config=['image_col=A', 'label_col=0'])
        with pytest.raises(AttributeError):
            images = Images('foo', 'files/empty.csv', config=['label_col=-1', 'image_col=0'])
        with pytest.raises(AttributeError):
            images = Images('foo', 'files/empty.csv', config=['label_col=A', 'image_col=0'])
        with pytest.raises(AttributeError):
            images = Images('foo', 'files/empty.csv', config=['label_col=', 'image_col=0'])
        with pytest.raises(AttributeError):
            images = Images('foo', 'files/empty.csv', config=['image_col=0'])
        with pytest.raises(AttributeError):
            images = Images('foo', 'files/empty.csv', config=['label_col=0'])
        with pytest.raises(ValueError):
            images = Images('foo', 'files/empty.csv', config=['label_col=0', 'image_col=0'])

        f = open('files/empty.csv', 'w')
        f.write('1,2\n')
        f.close()
        with pytest.raises(IndexError):
            images = Images('foo', 'files/empty.csv', config=['label_col=2', 'image_col=0'])
        with pytest.raises(IndexError):
            images = Images('foo', 'files/empty.csv', config=['label_col=0', 'image_col=2'])

        f = open('files/empty.csv', 'w')
        f.write('header,header\n')
        f.write('1,2\n')
        f.close()
        with pytest.raises(IndexError):
            images = Images('foo', 'files/empty.csv',
                            config=['label_col=2', 'image_col=0', 'header'])
        with pytest.raises(IndexError):
            images = Images('foo', 'files/empty.csv',
                            config=['label_col=0', 'image_col=2', 'header'])

        os.remove('files/empty.csv')

    def test_012(self):
        """ Images - JSON - bad arguments """

        # non-exist file
        with pytest.raises(OSError):
            images = Images('foo', 'noexist.json', config=['image_key=image', 'label_key=label'])
        with pytest.raises(OSError):
            images = Images('foo', 'http://noexist.json',
                            config=['image_key=image', 'label_key=label'])
        with pytest.raises(OSError):
            images = Images('foo', 'https://noexist.json',
                            config=['image_key=image', 'label_key=label'])

        # missing arguments
        f = open('files/empty.json', 'w')
        f.close()
        with pytest.raises(AttributeError):
            images = Images('foo', 'files/empty.json', config=[])
        with pytest.raises(AttributeError):
            images = Images('foo', 'files/empty.json', config=['image_key=image'])
        with pytest.raises(AttributeError):
            images = Images('foo', 'files/empty.json', config=['label_key=key'])
        with pytest.raises(ValueError):
            images = Images('foo', 'files/empty.json',
                            config=['image_key=image', 'label_key=image'])

        # bad format
        f = open('files/empty.json', 'w')
        f.write('{"0"\n')
        f.close()
        with pytest.raises(OSError):
            images = Images('foo', 'files/empty.json',
                            config=['image_key=image', 'label_key=label'])

        # missing keys
        f = open('files/test.json', 'w')
        f.write("[")
        f.write('{"label": 0, "image": "files/1.jpg"},\n')
        f.write('{"label": 0, "image": "files/2.jpg"},\n')
        f.write('{"label": 0, "image": "files/3.jpg"}\n')
        f.write("]")
        f.close()
        with pytest.raises(IndexError):
            images = Images('foo', 'files/test.json', config=['image_key=image', 'label_key=foo'])
        with pytest.raises(IndexError):
            images = Images('foo', 'files/test.json', config=['image_key=foo', 'label_key=label'])

        os.remove('files/test.json')
        os.remove('files/empty.json')

    def test_013(self):
        """ Images - list - bad arguments """

        # no images, no labels
        with pytest.raises(TypeError):
            images = Images('foo', [])

        # mismatch in number of labels
        with pytest.raises(AttributeError):
            images = Images('foo', ['a'], [1, 2])

    def test_014(self):
        """ Images - memory - bad arguments """
        # no images, no labels
        memory = np.asarray([])
        with pytest.raises(TypeError):
            images = Images('foo', memory)

        # mismatch in number of labels
        memory = np.asarray([[1]])
        with pytest.raises(AttributeError):
            images = Images('foo', memory, [1, 2])

    def test_015(self):
        """ Images - directory - no images """
        if os.path.isdir('files/empty'):
            rmtree('files/empty')
        os.mkdir('files/empty')

        images = Images('foo', 'files/empty')
        self.assertEqual(len(images), 0)
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.classes, {})
        self.assertEqual(images.labels, [])
        self.assertEqual(images.images, [])
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 0)

        f = open('files/empty/foo.txt', 'w+')
        f.close()
        images = Images('foo', 'files/empty')
        self.assertEqual(len(images), 0)
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.classes, {})
        self.assertEqual(len(images.labels), 0)
        self.assertEqual(images.images, [])
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 0)

        if not os.path.isdir('files/empty/tmp1'):
            os.mkdir('files/empty/tmp1')
        if not os.path.isdir('files/empty/tmp2'):
            os.mkdir('files/empty/tmp2')
        images = Images('foo', 'files/empty')
        self.assertEqual(len(images), 0)
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.classes, {})
        self.assertEqual(len(images.labels), 0)
        self.assertEqual(images.images, [])
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 0)

        if not os.path.isdir('files/empty/.tmp'):
            os.mkdir('files/empty/.tmp')
        f = open('files/empty/.tmp/1.jpg', 'w+')
        f.close()
        images = Images('foo', 'files/empty')
        self.assertEqual(len(images), 0)
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.classes, {})
        self.assertEqual(len(images.labels), 0)
        self.assertEqual(images.images, [])
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 0)

        images = Images('foo', 'files/empty', config=['store'])
        self.assertTrue(os.path.isfile("foo.h5"))
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(len(images), 0)
        self.assertEqual(images.classes, {})
        self.assertEqual(images.count, 0)
        self.assertEqual(images.images, [])
        self.assertEqual(len(images.labels), 0)
        os.remove('foo.h5')

        # line 57 only works if name=None
        images = Images(None, 'files/empty', config=['store'])
        self.assertTrue(os.path.isfile("files/empty.h5"))
        images = Images()
        images.load('files/empty')
        self.assertEqual(images.name, 'files/empty')
        self.assertEqual(len(images), 0)
        self.assertEqual(images.classes, {})
        self.assertEqual(images.count, 0)
        self.assertEqual(images.images, [])
        self.assertEqual(len(images.labels), 0)
        os.remove('files/empty.h5')

        images = Images(images='files/empty', config=['store'])
        self.assertTrue(os.path.isfile("unnamed.h5"))
        images = Images()
        images.load('unnamed')
        self.assertEqual(images.name, 'unnamed')
        self.assertEqual(len(images), 0)
        self.assertEqual(images.classes, {})
        self.assertEqual(images.count, 0)
        self.assertEqual(images.images, [])
        self.assertEqual(len(images.labels), 0)

        images = Images()
        images.load()
        self.assertEqual(images.name, 'unnamed')
        self.assertEqual(len(images), 0)
        self.assertEqual(images.classes, {})
        self.assertEqual(images.count, 0)
        self.assertEqual(images.images, [])
        self.assertEqual(len(images.labels), 0)
        rmtree('files/empty')

    def test_016(self):
        """ Images - load - bad arguments """
        images = Images()
        with pytest.raises(ValueError):
            images.load(None)
        with pytest.raises(TypeError):
            images.load(1)
        with pytest.raises(TypeError):
            images = Images(_dir=None)
        images = Images()
        images.load(_dir='./')
        self.assertEqual(images.dir, './')
        os.remove('unnamed.h5')

    def test_017(self):
        """ Images - directory - bad images """

        if os.path.isdir('files/bad/tmp1'):
            rmtree('files/bad')
        os.mkdir('files/bad')
        os.mkdir('files/bad/tmp1')
        f = open('files/bad/tmp1/1.jpg', 'w+')
        f.close()
        images = Images('foo', 'files/bad')
        self.assertEqual(images.fail, 1)
        self.assertEqual(images.classes, {'tmp1': 0})
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(len(images.labels[0]), 0)
        self.assertEqual(len(images), 1)
        self.assertEqual(len(images[0]), 0)
        self.assertEqual(images.count, 0)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.errors), 1)
        rmtree('files/bad')

    def test_018(self):
        """ Images - attributes """
        if os.path.isdir('files/empty'):
            rmtree('files/empty')
        os.mkdir('files/empty')
        images = Images('foo', 'files/empty',
                        config=['store', 'author=andy', 'license=2.0', 'desc=any', 'src=mysrc'])
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.author, 'andy')
        self.assertEqual(images.license, '2.0')
        self.assertEqual(images.desc, 'any')
        self.assertEqual(images.src, 'mysrc')
        self.assertEqual(images.time, 0)
        os.rmdir('files/empty')

    def test_019(self):
        """ Images - directory - single class """
        if os.path.isdir('files/root'):
            rmtree('files/root')
        os.mkdir('files/root')
        os.mkdir('files/root/tmp1')
        copy('files/1.jpg', 'files/root/tmp1')
        copy('files/2.jpg', 'files/root/tmp1')
        images = Images('foo', 'files/root', config=['store'])
        self.assertEqual(images.classes, {'tmp1': 0})
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[0][1], 0)
        self.assertEqual(images.images[0].shape, (2, 128, 128, 3))
        self.assertTrue(images.time > 0)

        # load, store
        images = Images()
        images.load('foo')
        self.assertEqual(images.classes, {'tmp1': 0})
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[0][1], 0)
        self.assertEqual(images.images[0].shape, (2, 128, 128, 3))
        self.assertTrue(images.time > 0)

        # stream
        images = Images('foo', 'files/root', config=['stream'])
        self.assertEqual(images.classes, {'tmp1': 0})
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.count, 2)
        self.assertTrue(images.time > 0)

        # load, stream
        images = Images()
        images.load('foo')
        self.assertEqual(images.classes, {'tmp1': 0})
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[0][1], 0)
        self.assertEqual(images.images[0].shape, (2, 128, 128, 3))
        self.assertTrue(images.time > 0)

        # error
        f = open('files/root/tmp1/bad.jpg', 'w+')
        f.close()
        images = Images('foo', 'files/root', config=['store'])
        self.assertEqual(images.classes, {'tmp1': 0})
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[0][1], 0)
        self.assertEqual(images.images[0].shape, (2, 128, 128, 3))
        self.assertTrue(images.time > 0)

        # load, error, store
        images = Images()
        images.load('foo')
        self.assertEqual(images.classes, {'tmp1': 0})
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[0][1], 0)
        self.assertEqual(images.images[0].shape, (2, 128, 128, 3))
        self.assertTrue(images.time > 0)

        # error, stream
        images = Images('foo', 'files/root', config=['stream'])
        self.assertEqual(images.classes, {'tmp1': 0})
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.count, 2)
        self.assertTrue(images.time > 0)

        # load, error, stream
        images = Images()
        images.load('foo')
        self.assertEqual(images.classes, {'tmp1': 0})
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[0][1], 0)
        self.assertEqual(images.images[0].shape, (2, 128, 128, 3))
        self.assertTrue(images.time > 0)

        # stream height != width
        images = Images('foo', 'files/root', config=['stream', 'resize=(50,40)'])
        self.assertEqual(images.classes, {'tmp1': 0})
        self.assertEqual(images.fail, 1)
        self.assertEqual(images.shape, (50, 40))
        self.assertEqual(images.count, 2)
        self.assertTrue(images.time > 0)

        # load, stream, height != width
        images = Images()
        images.load('foo')
        self.assertEqual(images.classes, {'tmp1': 0})
        self.assertEqual(images.fail, 1)
        self.assertEqual(images.shape, (50, 40))
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[0][1], 0)
        self.assertEqual(images.images[0].shape, (2, 50, 40, 3))
        self.assertTrue(images.time > 0)

        rmtree('files/root')
        os.remove('foo.h5')

    def test_020(self):
        """ Images - directory - shape on flatten and resize """
        if os.path.isdir('files/root'):
            rmtree('files/root')
        os.mkdir('files/root')
        os.mkdir('files/root/tmp1')
        copy('files/1.jpg', 'files/root/tmp1')
        copy('files/2.jpg', 'files/root/tmp1')
        images = Images('foo', 'files/root', config=['store', 'resize=(50,50)', 'flatten'])
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(images.images[0][0].shape, (7500,))
        images = Images('foo', 'files/root', config=['store', 'resize=(30,50)', 'flatten'])
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.images[0][0].shape, (4500,))
        images = Images('foo', 'files/root', config=['store', 'resize=(30,50)', 'flatten', 'gray'])
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.images[0][0].shape, (1500,))

        rmtree('files/root')
        os.remove('foo.h5')

    def test_021(self):
        """ Images - directory - multi class """
        if os.path.isdir('files/root'):
            rmtree('files/root')
        os.mkdir('files/root')
        os.mkdir('files/root/tmp1')
        os.mkdir('files/root/tmp2')
        copy('files/1.jpg', 'files/root/tmp1')
        copy('files/2.jpg', 'files/root/tmp1')
        copy('files/3.jpg', 'files/root/tmp2')

        images = Images('foo', 'files/root', config=['store', 'resize=(50,50)'])
        self.assertEqual(images.classes, {'tmp1': 0, 'tmp2': 1})
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(images.count, 3)
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[0][1], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (2, 50, 50, 3))
        self.assertEqual(images.images[1].shape, (1, 50, 50, 3))

        images = Images('foo', 'files/root', config=['store', 'resize=(50,50)', 'gray', 'flat'])
        self.assertEqual(images.classes, {'tmp1': 0, 'tmp2': 1})
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(images.count, 3)
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[0][1], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (2, 2500))
        self.assertEqual(images.images[1].shape, (1, 2500))

        rmtree('files/root')
        os.remove('foo.h5')

    def test_022(self):
        """ Images - bad dir, free h5 file """
        with pytest.raises(OSError):
            images = Images('foo', 'files/nodir',
                            config=['store', 'resize=(50,50)', 'gray', 'flat'])
        os.remove('foo.h5')

    def test_023(self):
        """ Images - memory - no images """
        memory = np.asarray([])
        images = Images('foo', memory, 1)
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 0)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 0)
        self.assertEqual(len(images.labels), 0)
        self.assertEqual(images.classes, {'1': 0})

    def test_024(self):
        """ Images - memory - 1D - no store """

        # single image - same size as resize
        memory = np.asarray([[1]])
        images = Images('foo', memory, 1, config=['resize=(1,1)'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (1, 1))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(images.labels[0][0], 0)

        # flattened to unflattened
        a = cv2.imread('files/1.jpg', cv2.IMREAD_GRAYSCALE)
        b = cv2.resize(a, (50, 50), interpolation=cv2.INTER_AREA).flatten()
        images = Images('foo', [b], 1, config=['resize=(50,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(images.labels[0][0], 0)

        # multiple flattened images
        c = np.asarray([b, b, b])
        images = Images('foo', c, 1, config=['resize=(50,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.classes, {'1': 0})

        d = np.asarray([b for _ in range(1000)])
        images = Images('foo', d, 1, config=['resize=(50,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1000)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(len(images.images[0]), 1000)
        self.assertEqual(len(images.labels[0]), 1000)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.classes, {'1': 0})
        self.assertTrue(images.time > 0)
        self.assertEqual(images.images[0][0].shape, (50, 50))

        # multiple flattened images - flatten
        c = np.asarray([b, b, b])
        images = Images('foo', c, 1, config=['resize=(30,50)', 'flat'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(images.images[0][0].shape, (1500,))

        # multiple flattened images - list is one value
        c = np.asarray([b, b, b])
        images = Images('foo', c, [0, 0, 0], config=['resize=(30,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(images.images[0][0].shape, (30, 50))

        # multiple flattened images - list different values
        c = np.asarray([b, b, b, b, b])
        images = Images('foo', c, [0, 1, 0, 1, 1], config=['resize=(30,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(len(images.images[1]), 3)
        self.assertEqual(len(images.labels[1]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(images.images[0][0].shape, (30, 50))

        # gray
        c = np.asarray([a, a, a, a, a])
        images = Images('foo', c, [0, 1, 0, 1, 1], config=['resize=(30,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))
        images = Images('foo', c, [0, 1, 0, 1, 1], config=['resize=(30,50)', 'flat'])
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(images.images[0][0].shape, (4500,))
        images = Images('foo', c, [0, 1, 0, 1, 1], config=['resize=(30,50)', 'flat', 'gray'])
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(images.images[0][0].shape, (1500,))

    def test_025(self):
        """ Images - memory - 2D - no store """

        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)
        c = np.asarray([a, a, a, a, a])
        images = Images('foo', c, [0, 1, 0, 1, 1], config=['resize=(30,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))

        images = Images('foo', c, [0, 1, 0, 1, 1], config=['resize=(30,50)', 'flat'])
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(images.images[0][0].shape, (4500,))

        images = Images('foo', c, [0, 1, 0, 1, 1], config=['resize=(30,50)', 'flat', 'gray'])
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(images.images[0][0].shape, (1500,))

    def test_026(self):
        """ Images - memory - 3D - no store """

        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)
        b = np.asarray([['a']])

        memory = np.asarray([a])
        images = Images('foo', memory, 1, config=['resize=(30,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))

        # load bad image from memory
        memory = np.asarray([b])
        images = Images('foo', memory, 1, config=['resize=(30,50)'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 0)
        self.assertEqual(len(images.images), 0)
        self.assertEqual(len(images.labels), 0)
        self.assertEqual(images.classes, {'1': 0})

        # load None image from memory
        memory = np.asarray([None])
        images = Images('foo', memory, 1, config=['resize=(30,50)'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 0)
        self.assertEqual(len(images.images), 0)
        self.assertEqual(len(images.labels), 0)
        self.assertEqual(images.classes, {'1': 0})

        # load several images two with errors
        memory = np.asarray([None, a, b])
        images = Images('foo', memory, 1, config=['resize=(30,50)'])
        self.assertEqual(images.fail, 2)
        self.assertEqual(len(images.errors), 2)
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))

    def test_027(self):
        """ Images - memory - store """

        # one class
        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)
        c = np.asarray([a, a, a, a, a])
        images = Images('foo', c, 0, config=['resize=(30,50)', 'store'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))
        self.assertEqual(images.labels[0][0], 0)

        # one class, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))
        self.assertEqual(images.labels[0][0], 0)

        # multi-class
        images = Images('foo', c, [0, 1, 0, 1, 1], config=['resize=(30,50)', 'store'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)

        # multi class, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)

        os.remove('foo.h5')

    def test_028(self):
        """ Images - memory - stream """

        # one class, stream
        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)
        c = np.asarray([a])
        images = Images('foo', c, 0, config=['resize=(30,50)', 'stream'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'0': 0})

        # one class, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))
        self.assertEqual(images.labels[0][0], 0)

        # one class, gray, stream
        a = cv2.imread('files/1.jpg', cv2.IMREAD_GRAYSCALE)
        c = np.asarray([a, a, a, a])
        images = Images('foo', c, 0, config=['resize=(30,50)', 'stream', 'gray'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'0': 0})

        # one class, gray, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(images.images[0][0].shape, (30, 50))
        self.assertEqual(images.labels[0][0], 0)

        # multi class, stream
        c = np.asarray([a, a, a, a, a])
        images = Images('foo', c, [0, 1, 1, 0, 1], config=['resize=(30,50)', 'stream'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'0': 0, '1': 1})

        # multi class, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.images[1]), 3)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(len(images.labels[1]), 3)
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)

        os.remove('foo.h5')

    def test_029(self):
        """ Images - memory - labels as strings """

        # single class, label is string
        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)
        c = np.asarray([a])
        images = Images('foo', c, 'cat', config=['resize=(30,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)

        # single class, label is string, store
        c = np.asarray([a, a, a, a])
        images = Images('foo', c, 'cat', config=['resize=(30,50)', 'store'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)

        # single class, label is string, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)

        # single class, label is string, stream
        c = np.asarray([a, a, a, a])
        images = Images('foo', c, 'cat', config=['resize=(30,50)', 'stream'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)

        # single class, label is string, load, stream
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)

        os.remove('foo.h5')

    def test_030(self):
        """ Images - memory - labels as [strings] """

        # single class, label is [string]
        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)
        c = np.asarray([a])
        images = Images('foo', c, ['cat'], config=['resize=(30,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)

        # single class, label is [string], store
        c = np.asarray([a])
        images = Images('foo', c, ['cat'], config=['resize=(30,50)', 'store'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)

        # single class, label is [string], load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)

        # single class, label is [string], stream
        c = np.asarray([a])
        images = Images('foo', c, ['cat'], config=['resize=(30,50)', 'stream'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)

        # single class, label is [string], load, stream
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.labels[0][0], 0)

        # multi class, label is [string]
        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)
        c = np.asarray([a, a, a, a, a])
        images = Images('foo', c, ['cat', 'dog', 'cat', 'cat', 'dog'], config=['resize=(30,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.images[0]), 3)
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.images[1]), 3)
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.images[1]), 3)
            else:
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.images[0]), 3)

        # multi class, label is [string], store
        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)
        c = np.asarray([a, a, a, a, a])
        images = Images('foo', c, ['cat', 'dog', 'cat', 'cat', 'dog'],
                        config=['resize=(30,50)', 'store'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.images[1]), 3)
                self.assertEqual(len(images.images[0]), 2)
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.images[1]), 3)
            else:
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.images[0]), 3)

        # multi class, label is [string], store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.images[1]), 3)
                self.assertEqual(len(images.images[0]), 2)
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.images[1]), 3)
            else:
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.images[0]), 3)

        # multi class, label is [string], stream
        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)
        c = np.asarray([a, a, a, a, a])
        images = Images('foo', c, ['cat', 'dog', 'cat', 'cat', 'dog'],
                        config=['resize=(30,50)', 'stream'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.labels[0]), 2)
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
            else:
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.labels[0]), 3)

        # multi class, label is [string], stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.images[1]), 3)
                self.assertEqual(len(images.images[0]), 2)
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.images[1]), 3)
            else:
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.images[0]), 3)
            self.assertEqual(len(images.labels[0]), 3)

        os.remove('foo.h5')

    def test_031(self):
        """ Images - list - no images """
        images = Images('foo', [], 1)
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 0)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 0)
        self.assertEqual(len(images.labels), 0)
        self.assertEqual(images.time, 0)
        self.assertEqual(images.classes, [])

    def test_032(self):
        """ Images - list - nonexist-image """

        # single non-exist local image
        images = Images('foo', ['noimage.jpg'], 1)
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 0)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 0)
        self.assertEqual(len(images.labels), 0)
        self.assertEqual(images.classes, {'1': 0})

        # one good, one bad image
        images = Images('foo', ['files/1.jpg', 'noimage.jpg'], 1)
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(images.labels[0][0], 0)

        # single non-exist remote image
        images = Images('foo', ['http://foobar17.com/noimage.jpg'], 1)
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 0)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 0)
        self.assertEqual(len(images.labels), 0)
        self.assertEqual(images.classes, {'1': 0})

    def test_033(self):
        """ Images - list local files """

        # single class
        images = Images('foo', ['files/1.jpg', 'files/2.jpg'], 1)
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 128, 128, 3))

        # single class, store
        images = Images('foo', ['files/1.jpg', 'files/2.jpg'], 1,
                        config=['store', 'resize=40,50'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50, 3))

        # single class, store, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50, 3))

        # single class, stream
        images = Images('foo', ['files/1.jpg', 'files/2.jpg'], 1,
                        config=['stream', 'resize=40,50', 'gray'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # single class, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50))

        # multi-class
        images = Images('foo', ['files/1.jpg', 'files/2.jpg', 'files/3.jpg'], [0, 1, 1])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
        self.assertEqual(images.images[1].shape, (2, 128, 128, 3))

        # multi-class, store
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 1, 1],
                        config=['store'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
        self.assertEqual(images.images[1].shape, (2, 128, 128, 3))

        # multi-class, store, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
        self.assertEqual(images.images[1].shape, (2, 128, 128, 3))

        # multi-class, stream
        images = Images('foo', ['files/1.jpg', 'files/2.jpg', 'files/3.jpg'], [0, 1, 1],
                        config=['stream', 'flat', 'resize=50,50'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)

        # multi-class, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 7500))
        self.assertEqual(images.images[1].shape, (2, 7500))

    def test_034(self):
        """ Images - remote files """
        IMAGE1 = 'https://assets.pernod-ricard.com/uk/media_images/test.jpg'
        IMAGE2 = 'https://www.accesshq.com/workspace/images/articles/test-your-technology.jpg'

        # single class
        images = Images('foo', [IMAGE1, IMAGE2], 1)
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 128, 128, 3))

        # single class, store
        images = Images('foo', [IMAGE1, IMAGE2], 1, config=['store', 'resize=40,50'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50, 3))

        # single class, store, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50, 3))

        # single class, stream
        images = Images('foo', [IMAGE1, IMAGE2], 1, config=['stream', 'resize=40,50', 'gray'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # single class, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50))

        # multi-class
        images = Images('foo', [IMAGE1, IMAGE2, IMAGE1], [0, 1, 1])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
        self.assertEqual(images.images[1].shape, (2, 128, 128, 3))

        # multi-class, store
        images = Images('foo', [IMAGE1, IMAGE2, IMAGE1], [0, 1, 1], config=['store'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
        self.assertEqual(images.images[1].shape, (2, 128, 128, 3))

        # multi-class, store, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
        self.assertEqual(images.images[1].shape, (2, 128, 128, 3))

        # multi-class, stream
        images = Images('foo', [IMAGE1, IMAGE2, IMAGE1], [0, 1, 1],
                        config=['stream', 'flat', 'resize=50,50'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)

        # multi-class, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 7500))
        self.assertEqual(images.images[1].shape, (2, 7500))

    def test_035(self):
        """ Images - list - memory """
        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)

        # single class
        images = Images('foo', [a, a], 1)
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 128, 128, 3))

        # single class, store
        images = Images('foo', [a, a], 1, config=['store', 'resize=40,50'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50, 3))

        # single class, store, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50, 3))

        # single class, stream
        images = Images('foo', [a, a], 1, config=['stream', 'resize=40,50', 'gray'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # single class, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'1': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50))

        # multi-class
        images = Images('foo', [a, a, a], [0, 1, 1])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
        self.assertEqual(images.images[1].shape, (2, 128, 128, 3))

        # multi-class, store
        images = Images('foo', [a, a, a], [0, 1, 1], config=['store'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
        self.assertEqual(images.images[1].shape, (2, 128, 128, 3))

        # multi-class, store, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
        self.assertEqual(images.images[1].shape, (2, 128, 128, 3))

        # multi-class, stream
        images = Images('foo', [a, a, a], [0, 1, 1], config=['stream', 'flat', 'resize=50,50'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)

        # multi-class, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images.images[0]), 1)
        self.assertEqual(len(images.labels[0]), 1)
        self.assertEqual(len(images.images[1]), 2)
        self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        self.assertEqual(images.images[0].shape, (1, 7500))
        self.assertEqual(images.images[1].shape, (2, 7500))

    def test_036(self):
        """ Images - list - list as [string] """
        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)

        # single class
        images = Images('foo', [a, a], 'cat')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 128, 128, 3))

        # single class, store
        images = Images('foo', [a, a], 'cat', config=['store', 'resize=40,50'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50, 3))

        # single class, store, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50, 3))

        # single class, stream
        images = Images('foo', [a, a], 'cat', config=['stream', 'resize=40,50', 'gray'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # single class, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'cat': 0})
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (2, 40, 50))

        # multi-class
        images = Images('foo', [a, a, a], ['cat', 'dog', 'dog'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            self.assertEqual(len(images.images[0]), 1)
            self.assertEqual(len(images.labels[0]), 1)
            self.assertEqual(len(images.images[1]), 2)
            self.assertEqual(len(images.labels[1]), 2)
            self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
            self.assertEqual(images.images[1].shape, (2, 128, 128, 3))
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            self.assertEqual(len(images.images[0]), 2)
            self.assertEqual(len(images.labels[0]), 2)
            self.assertEqual(len(images.images[1]), 1)
            self.assertEqual(len(images.labels[1]), 1)
            self.assertEqual(images.images[0].shape, (2, 128, 128, 3))
            self.assertEqual(images.images[1].shape, (1, 128, 128, 3))
        self.assertEqual(images.labels[0][0], 0)

        # multi-class, store
        images = Images('foo', [a, a, a], ['cat', 'dog', 'dog'], config=['store'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            self.assertEqual(len(images.images[0]), 1)
            self.assertEqual(len(images.labels[0]), 1)
            self.assertEqual(len(images.images[1]), 2)
            self.assertEqual(len(images.labels[1]), 2)
            self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
            self.assertEqual(images.images[1].shape, (2, 128, 128, 3))
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            self.assertEqual(len(images.images[0]), 2)
            self.assertEqual(len(images.labels[0]), 2)
            self.assertEqual(len(images.images[1]), 1)
            self.assertEqual(len(images.labels[1]), 1)
            self.assertEqual(images.images[0].shape, (2, 128, 128, 3))
            self.assertEqual(images.images[1].shape, (1, 128, 128, 3))
        self.assertEqual(images.labels[0][0], 0)

        # multi-class, store, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
                self.assertEqual(images.images[1].shape, (2, 128, 128, 3))
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
                self.assertEqual(images.images[0].shape, (2, 128, 128, 3))
                self.assertEqual(images.images[1].shape, (1, 128, 128, 3))
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
                self.assertEqual(images.images[0].shape, (2, 128, 128, 3))
                self.assertEqual(images.images[1].shape, (1, 128, 128, 3))
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(images.images[0].shape, (1, 128, 128, 3))
                self.assertEqual(images.images[1].shape, (2, 128, 128, 3))

        # multi-class, stream
        images = Images('foo', [a, a, a], ['cat', 'dog', 'dog'],
                        config=['stream', 'flat', 'resize=50,50'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            self.assertEqual(len(images.labels[0]), 1)
            self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            self.assertEqual(len(images.labels[0]), 2)
            self.assertEqual(len(images.labels[1]), 1)
        self.assertEqual(images.labels[0][0], 0)

        # multi-class, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(images.images[0].shape, (1, 7500))
                self.assertEqual(images.images[1].shape, (2, 7500))
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
                self.assertEqual(images.images[0].shape, (2, 7500))
                self.assertEqual(images.images[1].shape, (1, 7500))
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
                self.assertEqual(images.images[0].shape, (2, 7500))
                self.assertEqual(images.images[1].shape, (1, 7500))
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(images.images[0].shape, (1, 7500))
                self.assertEqual(images.images[1].shape, (2, 7500))

        # [string] are numbers, store
        images = Images('foo', ['files/1.jpg', 'files/2.jpg'],
                        ['0', '1'], config=['store'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['0'] == 0:
            self.assertEqual(images.classes, {'0': 0, '1': 1})
        else:
            self.assertEqual(images.classes, {'0': 1, '1': 0})

        # [string] are numbers, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['0'] == 0:
            self.assertEqual(images.classes, {'0': 0, '1': 1})
        else:
            self.assertEqual(images.classes, {'0': 1, '1': 0})

        os.remove('foo.h5')

    def test_037(self):
        """ Images - list - bad files """

        # bad file in list - single class
        images = Images('foo', ['files/1.jpg', 'bad.jpg', 'files/2.jpg'], 0)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # bad file in list - single class - store
        images = Images('foo', ['files/1.jpg', 'bad.jpg', 'files/2.jpg'], 0, config=['store'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # bad file in list - single class - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # bad file in list - single class - stream
        images = Images('foo', ['files/1.jpg', 'bad.jpg', 'files/2.jpg'], 0, config=['stream'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # bad file in list - single class - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # bad file in list - empty class
        images = Images('foo', ['files/1.jpg', 'bad.jpg', 'files/2.jpg'], [0, 1, 0])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # bad file in list - empty class - store
        images = Images('foo', ['files/1.jpg', 'bad.jpg', 'files/2.jpg'],
                        [0, 1, 0], config=['store'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # bad file in list - empty class - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # bad file in list - empty class - stream
        images = Images('foo', ['files/1.jpg', 'bad.jpg', 'files/2.jpg'],
                        [0, 1, 0], config=['stream'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        # bad file in list - empty class - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(images.labels[0][0], 0)

        os.remove('foo.h5')

    def test_038(self):
        """ Images - CSV - local path """

        # empty, no header
        f = open('files/empty.csv', 'w')
        f.close()
        images = Images('foo', 'files/empty.csv', config=['image_col=0', 'label_col=1'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 0)

        with pytest.raises(EOFError):
            images = Images('foo', 'files/empty.csv',
                            config=['header', 'image_col=0', 'label_col=1'])

        # empty, header
        f = open('files/empty.csv', 'w')
        f.write('header\n')
        f.close()

        images = Images('foo', 'files/empty.csv', config=['header', 'image_col=0', 'label_col=1'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 0)

        # single class
        f = open('files/test.csv', 'w')
        f.write('0,files/1.jpg\n')
        f.write('0,files/2.jpg\n')
        f.write('0,files/3.jpg\n')
        f.close()
        images = Images('foo', 'files/test.csv', config=['label_col=0', 'image_col=1'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (3, 128, 128, 3))

        # single class, store
        images = Images('foo', 'files/test.csv', config=['label_col=0', 'image_col=1', 'store'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (3, 128, 128, 3))

        # single class, store, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (3, 128, 128, 3))

        # single class, stream
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'stream', 'resize=(50,60)'])
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (50, 60))
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(images.labels[0][0], 0)

        # single class, stream, load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (50, 60))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0].shape, (3, 50, 60, 3))

        # multi class
        f = open('files/test.csv', 'w')
        f.write('\'0\',files/1.jpg\n')
        f.write('\'1\',files/2.jpg\n')
        f.write('\'0\',files/3.jpg\n')
        f.close()
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(40,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {"'0'": 1, "'1'": 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)

        # multi class, store
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(40,50)', 'store'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {"'0'": 1, "'1'": 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)

        # multi class, store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {"'0'": 1, "'1'": 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # multi class, stream
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(40,50)', 'stream', 'gray'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {"'0'": 1, "'1'": 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)

        # multi class, stream/load
        images = Images()
        images.load('foo')
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {"'0'": 1, "'1'": 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        os.remove('files/empty.csv')
        os.remove('files/test.csv')

    def test_039(self):
        """ Images - CSV - bad paths """

        # bad file in class
        f = open('files/test.csv', 'w')
        f.write('\'0\',files/1.jpg\n')
        f.write('\'1\',files/2.jpg\n')
        f.write('\'0\',files/3.jpg\n')
        f.write('\'0\',files/bad.jpg\n')
        f.write('\'0\',files/3.jpg\n')
        f.close()
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(40,50)'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 3)
                self.assertEqual(len(images.labels[1]), 3)
        else:
            self.assertEqual(images.classes, {"'0'": 1, "'1'": 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 3)
                self.assertEqual(len(images.labels[1]), 3)
            else:
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # bad file in class, store
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(40,50)', 'store'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 3)
                self.assertEqual(len(images.labels[1]), 3)
        else:
            self.assertEqual(images.classes, {"'0'": 1, "'1'": 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 3)
                self.assertEqual(len(images.labels[1]), 3)
            else:
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # bad file in class, store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 3)
                self.assertEqual(len(images.labels[1]), 3)
        else:
            self.assertEqual(images.classes, {"'0'": 1, "'1'": 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 3)
                self.assertEqual(len(images.labels[1]), 3)
            else:
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # bad file in class, stream
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(40,50)', 'stream'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.labels[1]), 3)
        else:
            self.assertEqual(images.classes, {"'0'": 1, "'1'": 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.labels[1]), 3)
            else:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.labels[1]), 1)

        # bad file in class, stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 1)
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 3)
                self.assertEqual(len(images.labels[1]), 3)
        else:
            self.assertEqual(images.classes, {"'0'": 1, "'1'": 0})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 3)
                self.assertEqual(len(images.labels[1]), 3)
            else:
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # bad file - empty class
        f = open('files/test.csv', 'w')
        f.write('\'0\',files/1.jpg\n')
        f.write('\'1\',files/2.jpg\n')
        f.write('\'0\',files/3.jpg\n')
        f.write('\'2\',files/bad.jpg\n')
        f.write('\'0\',files/3.jpg\n')
        f.close()
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(40,50)'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)

        # bad file- empty class - store
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(40,50)', 'store'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)

        # bad file- empty class - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)

        # bad file- empty class - stream
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(40,50)', 'stream'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 2)

        # bad file- empty class - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 4)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)

        os.remove('files/test.csv')
        os.remove('foo.h5')

    def test_040(self):
        """ Images - CSV - remote path """

        # Images - remote files
        IMAGE1 = 'https://assets.pernod-ricard.com/uk/media_images/test.jpg'
        IMAGE2 = 'https://www.accesshq.com/workspace/images/articles/test-your-technology.jpg'

        # multi class
        f = open('files/test.csv', 'w')
        f.write('\'0\',' + IMAGE1 + '\n')
        f.write('\'1\',' + IMAGE2 + '\n')
        f.write('\'0\',' + IMAGE1 + '\n')
        f.close()
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(50,50)', 'flatten'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {"'1'": 0, "'0'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
        self.assertEqual(images.images[0][0].shape, (7500,))

        # multi class - store
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(50,50)', 'flatten', 'store'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {"'1'": 0, "'0'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
        self.assertEqual(images.images[0][0].shape, (7500,))

        # multi class - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {"'1'": 0, "'0'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)

        # multi class - stream
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(50,50)', 'flatten', 'store'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {"'1'": 0, "'0'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
        self.assertEqual(images.images[0][0].shape, (7500,))

        # multi class - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.count, 3)
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {"'1'": 0, "'0'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)

        # bad remote file
        f = open('files/test.csv', 'w')
        f.write('\'0\',' + IMAGE1 + '\n')
        f.write('\'1\',' + IMAGE2 + '\n')
        f.write('\'0\',' + IMAGE1 + '\n')
        f.write('\'0\',http://foobar17.com/noimage.jpg\n')
        f.close()
        images = Images('foo', 'files/test.csv',
                        config=['label_col=0', 'image_col=1', 'resize=(50,50)', 'flatten'])
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.count, 3)
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes["'0'"] == 0:
            self.assertEqual(images.classes, {"'0'": 0, "'1'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            self.assertEqual(images.classes, {"'1'": 0, "'0'": 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
        self.assertEqual(images.images[0][0].shape, (7500,))

        os.remove('foo.h5')
        os.remove('files/test.csv')

    def test_041(self):
        """ Images - CSV - memory """

        # multi-class - flatten/gray
        images = Images('foo', 'files/array.csv',
                        config=['resize=(40,50)', 'gray', 'flat', 'label_col=0', 'image_col=1'])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['daisy'] == 0:
            self.assertEqual(images.classes, {'daisy': 0, 'cat': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        else:
            self.assertEqual(images.classes, {'daisy': 1, 'cat': 0})
        self.assertEqual(images.images[0][0].shape, (2000,))

        # multi-class - gray
        images = Images('foo', 'files/array.csv',
                        config=['resize=(40,50)', 'gray', 'label_col=0', 'image_col=1'])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['daisy'] == 0:
            self.assertEqual(images.classes, {'daisy': 0, 'cat': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        else:
            self.assertEqual(images.classes, {'daisy': 1, 'cat': 0})
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # multi-class - color
        images = Images('foo', 'files/array.csv',
                        config=['resize=(40,50)', 'label_col=0', 'image_col=1'])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['daisy'] == 0:
            self.assertEqual(images.classes, {'daisy': 0, 'cat': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        else:
            self.assertEqual(images.classes, {'daisy': 1, 'cat': 0})
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # multi-class - store
        images = Images('foo', 'files/array.csv',
                        config=['resize=(40,50)', 'label_col=0', 'image_col=1', 'store'])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['daisy'] == 0:
            self.assertEqual(images.classes, {'daisy': 0, 'cat': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        else:
            self.assertEqual(images.classes, {'daisy': 1, 'cat': 0})
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # multi-class - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['daisy'] == 0:
            self.assertEqual(images.classes, {'daisy': 0, 'cat': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        else:
            self.assertEqual(images.classes, {'daisy': 1, 'cat': 0})
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # multi-class - stream
        images = Images('foo', 'files/array.csv',
                        config=['resize=(40,50)', 'label_col=0', 'image_col=1', 'stream'])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 2)
        if images.classes['daisy'] == 0:
            self.assertEqual(images.classes, {'daisy': 0, 'cat': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
        else:
            self.assertEqual(images.classes, {'daisy': 1, 'cat': 0})

        # multi-class - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['daisy'] == 0:
            self.assertEqual(images.classes, {'daisy': 0, 'cat': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        else:
            self.assertEqual(images.classes, {'daisy': 1, 'cat': 0})
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        os.remove('foo.h5')

    def test_042(self):
        """ Images - CSV - wrong line size """

        # bad file - incomplete embedded data
        f = open('files/test.csv', 'w')
        f.write('0,"[1,2,3,4"\n')
        f.close()
        images = Images('foo', 'files/test.csv',
                        config=['resize=(8,8)', 'label_col=0', 'image_col=1'])
        self.assertEqual(images.count, 0)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)

        # bad file - missing image column
        f = open('files/test.csv', 'w')
        f.write('0,files/1.jpg\n')
        f.write('0\n')
        f.write('0,files/2.jpg\n')
        f.close()
        images = Images('foo', 'files/test.csv',
                        config=['resize=(8,8)', 'label_col=0', 'image_col=1'])
        self.assertEqual(images.count, 2)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)

        # bad file - blank file
        f = open('files/test.csv', 'w')
        f.write('0,files/1.jpg\n')
        f.write('0,\n')
        f.write('0,files/2.jpg\n')
        f.close()
        images = Images('foo', 'files/test.csv',
                        config=['resize=(8,8)', 'label_col=0', 'image_col=1'])
        self.assertEqual(images.count, 2)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)

         # bad file - wrong file type
        f = open('files/test.csv', 'w')
        f.write('0,files/1.jpg\n')
        f.write('0,func_test.py\n')
        f.write('0,files/2.jpg\n')
        f.close()
        images = Images('foo', 'files/test.csv',
                        config=['resize=(8,8)', 'label_col=0', 'image_col=1'])
        self.assertEqual(images.count, 2)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)

        os.remove('files/test.csv')

    def test_043(self):
        """ Images - JSON - local path """

        # single class
        f = open('files/test.json', 'w')
        f.write("[")
        f.write('{"label": 0, "image": "files/1.jpg"},\n')
        f.write('{"label": 0, "image": "files/2.jpg"},\n')
        f.write('{"label": 0, "image": "files/3.jpg"}\n')
        f.write("]")
        f.close()
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)', 'label_key=label', 'image_key=image'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # single class - store
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)', 'label_key=label', 'image_key=image', 'store'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # single class - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # single class - stream
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)', 'label_key=label', 'image_key=image', 'stream'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(len(images.labels[0]), 3)

        # single class - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(images.images[0][0].shape, (40, 50, 3))

        # multi class
        f = open('files/test.json', 'w')
        f.write("[")
        f.write('{"label": "cat", "image": "files/1.jpg"},\n')
        f.write('{"label": "dog", "image": "files/2.jpg"},\n')
        f.write('{"label": "cat", "image": "files/3.jpg"}\n')
        f.write("]")
        f.close()
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)', 'label_key=label', 'image_key=image', 'gray'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # multi class - store
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)',
                                'label_key=label',
                                'image_key=image',
                                'gray',
                                'flat',
                                'store'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.images[0][0].shape, (2000,))

        # multi class - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.images[0][0].shape, (2000,))

        # multi class - stream
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)',
                                'label_key=label',
                                'image_key=image',
                                'gray',
                                'flat',
                                'stream'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)

        # multi class - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        else:
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[1]), 1)
                self.assertEqual(len(images.labels[1]), 1)
            else:
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[1]), 2)
                self.assertEqual(len(images.labels[1]), 2)
        self.assertEqual(images.images[0][0].shape, (2000,))

        os.remove('files/test.json')
        os.remove('foo.h5')

    def test_044(self):
        """ Images - JSON - bad local path """

        # multi-class
        f = open('files/test.json', 'w')
        f.write("[")
        f.write('{"label": "cat", "image": "files/1.jpg"},\n')
        f.write('{"label": "dog", "image": "files/2.jpg"},\n')
        f.write('{"label": "cat", "image": "files/nofile.jpg"},\n')
        f.write('{"label": "cat", "image": "files/3.jpg"}\n')
        f.write("]")
        f.close()
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)', 'label_key=label', 'image_key=image'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)

        # multi-class - store
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)', 'label_key=label', 'image_key=image', 'store'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)

        # multi-class - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)

        # multi-class - stream
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)', 'label_key=label', 'image_key=image', 'stream'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 1)
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 1)

        # multi-class - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes['cat'] == 0:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)
        else:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)

        os.remove('files/test.json')
        os.remove('foo.h5')

    def test_045(self):
        """ Images - JSON - remote path """

        IMAGE1 = 'https://assets.pernod-ricard.com/uk/media_images/test.jpg'
        IMAGE2 = 'https://www.accesshq.com/workspace/images/articles/test-your-technology.jpg'

        # single file
        f = open('files/test.json', 'w')
        f.write("[")
        f.write('{"label": 0, "image": "' + IMAGE1 + '"},\n')
        f.write('{"label": 0, "image": "' + IMAGE2 + '"},\n')
        f.write('{"label": 0, "image": "' + IMAGE1 + '"}\n')
        f.write("]")
        f.close()
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)', 'label_key=label', 'image_key=image', 'gray'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # single file - store
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)',
                                'label_key=label',
                                'image_key=image',
                                'gray',
                                'store'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # single file - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # single file - stream
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)',
                                'label_key=label',
                                'image_key=image',
                                'gray',
                                'stream'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(images.labels[0][0], 0)

        # single file - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(len(images.labels[0]), 3)
        self.assertEqual(len(images.images[0]), 3)
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # multi class
        f = open('files/test.json', 'w')
        f.write("[")
        f.write('{"label": 0, "image": "' + IMAGE1 + '"},\n')
        f.write('{"label": 1, "image": "' + IMAGE2 + '"},\n')
        f.write('{"label": 1, "image": "' + IMAGE1 + '"}\n')
        f.write("]")
        f.close()
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)', 'label_key=label', 'image_key=image', 'gray'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0: 0, 1: 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
                self.assertEqual(len(images.images[1]), 1)
        else:
            self.assertEqual(images.classes, {0: 1, 1: 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
                self.assertEqual(len(images.images[1]), 1)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # multi class - store
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)',
                                'label_key=label',
                                'image_key=image',
                                'gray',
                                'store'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0: 0, 1: 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
                self.assertEqual(len(images.images[1]), 1)
        else:
            self.assertEqual(images.classes, {0: 1, 1: 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
                self.assertEqual(len(images.images[1]), 1)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # multi class - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0: 0, 1: 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
                self.assertEqual(len(images.images[1]), 1)
        else:
            self.assertEqual(images.classes, {0: 1, 1: 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
                self.assertEqual(len(images.images[1]), 1)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # multi class - stream
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)',
                                'label_key=label',
                                'image_key=image',
                                'gray',
                                'stream'])
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 2)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0: 0, 1: 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
        else:
            self.assertEqual(images.classes, {0: 1, 1: 0})

        # multi class - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 3)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0: 0, 1: 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 1)
                self.assertEqual(len(images.images[0]), 1)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 1)
                self.assertEqual(len(images.images[1]), 1)
        else:
            self.assertEqual(images.classes, {0: 1, 1: 0})
        self.assertEqual(images.images[0][0].shape, (40, 50))

        os.remove('files/test.json')
        os.remove('foo.h5')

    def test_046(self):
        """ Images - JSON - bad remote files """

        IMAGE1 = 'https://assets.pernod-ricard.com/uk/media_images/test.jpg'
        IMAGE2 = 'https://www.accesshq.com/workspace/images/articles/test-your-technology.jpg'

        # empty class
        f = open('files/test.json', 'w')
        f.write("[")
        f.write('{"label": 0, "image": "http://badfile.ppt"},\n')
        f.write('{"label": 1, "image": "' + IMAGE2 + '"},\n')
        f.write('{"label": 1, "image": "' + IMAGE1 + '"}\n')
        f.write("]")
        f.close()

        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)',
                                'label_key=label',
                                'image_key=image',
                                'gray'])
        self.assertEqual(images.count, 2)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0:0, 1:1})
        else:
            self.assertEqual(images.classes, {0:1, 1:0})
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # empty class - store
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)',
                                'label_key=label',
                                'image_key=image',
                                'gray',
                                'store'])
        self.assertEqual(images.count, 2)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0:0, 1:1})
        else:
            self.assertEqual(images.classes, {0:1, 1:0})
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # empty class - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 2)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0:0, 1:1})
        else:
            self.assertEqual(images.classes, {0:1, 1:0})
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        # empty class - stream
        images = Images('foo', 'files/test.json',
                        config=['resize=(40,50)',
                                'label_key=label',
                                'image_key=image',
                                'gray',
                                'stream'])
        self.assertEqual(images.count, 2)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.labels), 1)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0:0, 1:1})
        else:
            self.assertEqual(images.classes, {0:1, 1:0})
        self.assertEqual(len(images.labels[0]), 2)

        # empty class - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 2)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)
        self.assertEqual(images.shape, (40, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0:0, 1:1})
        else:
            self.assertEqual(images.classes, {0:1, 1:0})
        self.assertEqual(len(images.labels[0]), 2)
        self.assertEqual(len(images.images[0]), 2)
        self.assertEqual(images.images[0][0].shape, (40, 50))

        os.remove('files/test.json')
        os.remove('foo.h5')

    def test_047(self):
        """ Images - JSON - memory """

        f = open('files/test.json', 'w')
        f.write("[\n")
        f.write('{"label": 0, "image": "[0,1,2,3,4,5,6,7]"},\n')
        f.write('{"label": 0, "image": "[10,11,12,13,14,15,16,17]"},\n')
        f.write('{"label": 0, "image": "[0,1,2,3,4,5,6,7]"},\n')
        f.write('{"label": 1, "image": "[20,21,22,23,24,25,26,27]"},\n')
        f.write('{"label": 1, "image": "[0,1,2,3,4,5,6,7]"}\n')
        f.write("]\n")
        f.close()

        # multi class
        images = Images('foo', 'files/test.json',
                        config=['resize=(8,8)',
                                'label_key=label',
                                'image_key=image',
                                'gray'])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (8, 8))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0: 0, 1: 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        else:
            self.assertEqual(images.classes, {0: 1, 1: 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        self.assertEqual(images.images[0][0].shape, (8, 8))

        # multi class - store
        images = Images('foo', 'files/test.json',
                        config=['resize=(8,8)',
                                'label_key=label',
                                'image_key=image',
                                'gray',
                                'store'])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (8, 8))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0: 0, 1: 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        else:
            self.assertEqual(images.classes, {0: 1, 1: 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        self.assertEqual(images.images[0][0].shape, (8, 8))

        # multi class - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (8, 8))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0: 0, 1: 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        else:
            self.assertEqual(images.classes, {0: 1, 1: 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        self.assertEqual(images.images[0][0].shape, (8, 8))

        # multi class - stream
        images = Images('foo', 'files/test.json',
                        config=['resize=(8,8)',
                                'label_key=label',
                                'image_key=image',
                                'gray',
                                'stream'])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (8, 8))
        self.assertEqual(len(images.labels), 2)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0: 0, 1: 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)

        else:
            self.assertEqual(images.classes, {0: 1, 1: 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)

        # multi class - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.count, 5)
        self.assertEqual(images.fail, 0)
        self.assertEqual(len(images.errors), 0)
        self.assertEqual(images.shape, (8, 8))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        if images.classes[0] == 0:
            self.assertEqual(images.classes, {0: 0, 1: 1})
            if images.labels[0][0] == 0:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        else:
            self.assertEqual(images.classes, {0: 1, 1: 0})
            if images.labels[0][0] == 1:
                self.assertEqual(len(images.labels[0]), 3)
                self.assertEqual(len(images.images[0]), 3)
                self.assertEqual(len(images.labels[1]), 2)
                self.assertEqual(len(images.images[1]), 2)
            else:
                self.assertEqual(len(images.labels[0]), 2)
                self.assertEqual(len(images.images[0]), 2)
                self.assertEqual(len(images.labels[1]), 3)
                self.assertEqual(len(images.images[1]), 3)
        self.assertEqual(images.images[0][0].shape, (8, 8))

        os.remove('foo.h5')
        os.remove('files/test.json')

    def test_048(self):
        """ Images - flatten """

        images = Images()
        with pytest.raises(TypeError):
            images.flatten = 'A'

        if os.path.isdir('files/root'):
            rmtree('files/root')
        os.mkdir('files/root')
        os.mkdir('files/root/tmp1')
        copy('files/1.jpg', 'files/root/tmp1')
        copy('files/2.jpg', 'files/root/tmp1')

        # flatten, color
        images = Images('foo', 'files/root', config=['resize=(50,50)'])
        images.flatten = True
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(images.images[0][0].shape, (7500,))

        # already flatten
        images.flatten = True
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(images.images[0][0].shape, (7500,))

        # unflatten, color
        images.flatten = False
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(images.images[0][0].shape, (50, 50, 3))

        # flatten, gray
        images = Images('foo', 'files/root', config=['resize=(50,50)', 'gray'])
        images.flatten = True
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(images.images[0][0].shape, (2500,))

        # already flatten
        images.flatten = True
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(images.images[0][0].shape, (2500,))

        # unflatten, gray
        images.flatten = False
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(images.images[0][0].shape, (50, 50))

        rmtree('files/root')

    def test_049(self):
        """ Images - resize """

        images = Images()
        with pytest.raises(TypeError):
            images.resize = 'A'
        with pytest.raises(AttributeError):
            images.resize = (1,)
        with pytest.raises(TypeError):
            images.resize = (50, 'A')
        with pytest.raises(ValueError):
            images.resize = (0, 10)
        with pytest.raises(ValueError):
            images.resize = (-10, 10)
        with pytest.raises(ValueError):
            images.resize = (10, 0)
        with pytest.raises(ValueError):
            images.resize = (10, -1)

        if os.path.isdir('files/root'):
            rmtree('files/root')
        os.mkdir('files/root')
        os.mkdir('files/root/tmp1')
        copy('files/1.jpg', 'files/root/tmp1')
        copy('files/2.jpg', 'files/root/tmp1')

        # color
        images = Images('foo', 'files/root', config=['resize=(50,50)'])
        images.resize = (20, 30)
        self.assertEqual(images.images[0][0].shape, (20, 30, 3))
        self.assertEqual(images.shape, (20, 30))

        # flattened, color
        images = Images('foo', 'files/root', config=['resize=(50,50)', 'flat'])
        images.resize = (20, 30)
        self.assertEqual(images.images[0][0].shape, (20, 30, 3))
        self.assertEqual(images.shape, (20, 30))

        # gray
        images = Images('foo', 'files/root', config=['resize=(50,50)', 'gray'])
        images.resize = (20, 30)
        self.assertEqual(images.images[0][0].shape, (20, 30))
        self.assertEqual(images.shape, (20, 30))

        # flattened, gray
        images = Images('foo', 'files/root', config=['resize=(50,50)', 'flat', 'gray'])
        images.resize = (20, 30)
        self.assertEqual(images.images[0][0].shape, (20, 30))
        self.assertEqual(images.shape, (20, 30))

        rmtree('files/root')

    _async_obj = None
    SLEEP = 3
    def done_048(self, obj):
        self._async_obj = obj

    def test_050(self):
        """ Images - async handler """

        # basic
        Images('foo', ['files/1.jpg', 'files/2.jpg'], [0, 1], ehandler=self.done_048)
        time.sleep(self.SLEEP)
        self.assertTrue(self._async_obj is not None)
        self.assertEqual(self._async_obj.name, 'foo')
        self.assertEqual(self._async_obj.count, 2)
        self.assertEqual(self._async_obj.shape, (128, 128))
        self.assertEqual(len(self._async_obj.images), 2)
        self.assertEqual(len(self._async_obj.labels), 2)
        self._async_obj = None

        # basic - store
        Images('foo', ['files/1.jpg', 'files/2.jpg'], [0, 1],
               ehandler=self.done_048, config=['store', 'resize=50,50'])
        time.sleep(self.SLEEP)
        self.assertTrue(self._async_obj is not None)
        self.assertEqual(self._async_obj.name, 'foo')
        self.assertEqual(self._async_obj.count, 2)
        self.assertEqual(self._async_obj.shape, (50, 50))
        self.assertEqual(len(self._async_obj.images), 2)
        self.assertEqual(len(self._async_obj.labels), 2)
        self._async_obj = None

        # basic - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)

        # basic - stream
        Images('foo', ['files/1.jpg', 'files/2.jpg'], [0, 1],
               ehandler=self.done_048, config=['stream', 'resize=50,50'])
        time.sleep(self.SLEEP)
        self.assertTrue(self._async_obj is not None)
        self.assertEqual(self._async_obj.name, 'foo')
        self.assertEqual(self._async_obj.count, 2)
        self.assertEqual(self._async_obj.shape, (50, 50))
        self.assertEqual(len(self._async_obj.labels), 2)
        self._async_obj = None

        # basic - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)

        # error
        Images('foo', ['files/1.jpg', 'bad.jpg', 'files/2.jpg'],
               [0, 1, 2], ehandler=self.done_048)
        time.sleep(self.SLEEP)
        self.assertTrue(self._async_obj is not None)
        self.assertEqual(self._async_obj.name, 'foo')
        self.assertEqual(self._async_obj.count, 2)
        self.assertEqual(self._async_obj.shape, (128, 128))
        self.assertEqual(len(self._async_obj.images), 2)
        self.assertEqual(len(self._async_obj.labels), 2)
        self.assertEqual(self._async_obj.fail, 1)
        self.assertEqual(len(self._async_obj.errors), 1)
        self._async_obj = None

        # error - store
        Images('foo', ['files/1.jpg', 'bad.jpg', 'files/2.jpg'],
               [0, 1, 2], ehandler=self.done_048, config=['store'])
        time.sleep(self.SLEEP)
        self.assertTrue(self._async_obj is not None)
        self.assertEqual(self._async_obj.name, 'foo')
        self.assertEqual(self._async_obj.count, 2)
        self.assertEqual(self._async_obj.shape, (128, 128))
        self.assertEqual(len(self._async_obj.images), 2)
        self.assertEqual(len(self._async_obj.labels), 2)
        self.assertEqual(self._async_obj.fail, 1)
        self.assertEqual(len(self._async_obj.errors), 1)
        self._async_obj = None

        # error - store/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)

        # error - stream
        Images('foo', ['files/1.jpg', 'bad.jpg', 'files/2.jpg'],
               [0, 1, 2], ehandler=self.done_048, config=['stream'])
        time.sleep(self.SLEEP)
        self.assertTrue(self._async_obj is not None)
        self.assertEqual(self._async_obj.name, 'foo')
        self.assertEqual(self._async_obj.count, 2)
        self.assertEqual(self._async_obj.shape, (128, 128))
        self.assertEqual(len(self._async_obj.labels), 2)
        self.assertEqual(self._async_obj.fail, 1)
        self.assertEqual(len(self._async_obj.errors), 1)
        self._async_obj = None

        # error - stream/load
        images = Images()
        images.load('foo')
        self.assertEqual(images.name, 'foo')
        self.assertEqual(images.count, 2)
        self.assertEqual(images.shape, (128, 128))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.fail, 1)
        self.assertEqual(len(images.errors), 1)

        os.remove('foo.h5')

    def done_049(self, obj, arg):
        self._async_obj = obj
        self._async_arg = arg

    def test_051(self):
        """ Images - ehandler / args """

        Images('foo', ['files/1.jpg', 'files/2.jpg'], [0, 1], ehandler=(self.done_049, 17))
        time.sleep(self.SLEEP)
        self.assertTrue(self._async_obj is not None)
        self.assertEqual(self._async_obj.name, 'foo')
        self.assertEqual(self._async_obj.count, 2)
        self.assertEqual(self._async_obj.shape, (128, 128))
        self.assertEqual(len(self._async_obj.images), 2)
        self.assertEqual(len(self._async_obj.labels), 2)
        self.assertEqual(self._async_arg, (17,))
        self._async_obj = None

    def test_052(self):
        """ Images - dtype """

        # float64
        images = Images('foo', ['files/1.jpg'], 0,
                        config=['resize=50,50', 'gray', 'flat', 'float64', 'store'])
        self.assertTrue(images.dtype == np.float64)
        self.assertEqual(images.images[0][0].dtype, np.float64)
        images = Images()
        images.load('foo')
        self.assertTrue(images.dtype == np.float64)
        self.assertEqual(images.images[0][0].dtype, np.float64)
        images.flatten = False
        self.assertEqual(images.images[0][0].dtype, np.float64)
        images.flatten = True
        self.assertEqual(images.images[0][0].dtype, np.float64)
        # TODO resize complains about type 13 (32F C3) - must really mean 64?
        #images.resize = (25,25)
        #self.assertEqual(images.images[0][0].dtype, np.float64)

        # float32
        images = Images('foo', ['files/1.jpg'], 0,
                        config=['resize=50,50', 'gray', 'flat', 'float32', 'store'])
        self.assertTrue(images.dtype == np.float32)
        self.assertEqual(images.images[0][0].dtype, np.float32)
        images = Images()
        images.load('foo')
        self.assertTrue(images.dtype == np.float32)
        self.assertEqual(images.images[0][0].dtype, np.float32)
        images.flatten = False
        self.assertEqual(images.images[0][0].dtype, np.float32)
        images.flatten = True
        self.assertEqual(images.images[0][0].dtype, np.float32)
        images.resize = (25, 25)
        self.assertEqual(images.images[0][0].dtype, np.float32)

        # float16
        images = Images('foo', ['files/1.jpg'], 0,
                        config=['resize=50,50', 'gray', 'flat', 'float16', 'store'])
        self.assertEqual(images.dtype, np.float16)
        self.assertEqual(images.images[0][0].dtype, np.float16)
        images = Images()
        images.load('foo')
        self.assertEqual(images.dtype, np.float16)
        self.assertEqual(images.images[0][0].dtype, np.float16)
        images.flatten = False
        self.assertEqual(images.images[0][0].dtype, np.float16)
        images.flatten = True
        self.assertEqual(images.images[0][0].dtype, np.float16)
        # TODO resize complains about type 24 (16F C3)
        # images.resize = (25,25)
        # self.assertEqual(images.images[0][0].dtype, np.float16)

        # uint16
        images = Images('foo', ['files/1.jpg'], 0,
                        config=['resize=50,50', 'gray', 'flat', 'uint16', 'store'])
        self.assertEqual(images.dtype, np.uint16)
        self.assertEqual(images.images[0][0].dtype, np.uint16)
        images = Images()
        images.load('foo')
        self.assertEqual(images.dtype, np.uint16)
        self.assertEqual(images.images[0][0].dtype, np.uint16)
        images.flatten = False
        self.assertEqual(images.images[0][0].dtype, np.uint16)
        images.flatten = True
        self.assertEqual(images.images[0][0].dtype, np.uint16)
        images.resize = (25, 25)
        self.assertEqual(images.images[0][0].dtype, np.uint16)

        # uint8
        images = Images('foo', ['files/1.jpg'], 0,
                        config=['resize=50,50', 'gray', 'flat', 'uint8', 'store'])
        self.assertEqual(images.dtype, np.uint8)
        self.assertEqual(images.images[0][0].dtype, np.uint8)
        images = Images()
        images.load('foo')
        self.assertEqual(images.dtype, np.uint8)
        self.assertEqual(images.images[0][0].dtype, np.uint8)
        images.flatten = False
        self.assertEqual(images.images[0][0].dtype, np.uint8)
        images.flatten = True
        self.assertEqual(images.images[0][0].dtype, np.uint8)
        images.resize = (25, 25)
        self.assertEqual(images.images[0][0].dtype, np.uint8)

        os.remove('foo.h5')

    def test_053(self):
        """ Images - image types """

        # JPG 8bpp Gray
        images = Images('foo', ['files/8gray.jpg'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # JPG 8bpp RGB
        images = Images('foo', ['files/8rgb.jpg'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # J2K 8bpp RGB
        images = Images('foo', ['files/8rgb.jp2'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # PNG 8bpp Gray
        images = Images('foo', ['files/8gray.png'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # PNG 8bpp RGB
        images = Images('foo', ['files/8rgb.png'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # PNG 8bpp RGBA
        images = Images('foo', ['files/8rgba.png'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # PNG 16bit RGB
        images = Images('foo', ['files/16rgb.png'], 0, config=['uint16', '16bpp'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() > 256)

        # PNG 16bit RGBA
        images = Images('foo', ['files/16rgba.png'], 0, config=['uint16', '16bpp'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() > 256)

        # TIF 8bpp Gray
        images = Images('foo', ['files/8gray.tiff'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # TIF 8bpp RGB
        images = Images('foo', ['files/8rgb.tiff'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # TIFF 8bpp RGBA
        images = Images('foo', ['files/8rgba.tiff'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # TIFF 16bpp RGB
        images = Images('foo', ['files/16rgb.tiff'], 0, config=['uint16'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # TIFF 16bpp RGBA
        images = Images('foo', ['files/16rgba.tiff'], 0, config=['uint16'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # GIF 8bpp RGB
        images = Images('foo', ['files/8rgb.gif'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # GIF 8bpp RGBA
        images = Images('foo', ['files/8rgba.gif'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # GIF 16bpp RGB
        images = Images('foo', ['files/16rgb.gif'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # GIF 16bpp RGBA
        images = Images('foo', ['files/16rgba.gif'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # BMP 8bpp Gray
        images = Images('foo', ['files/8gray.bmp'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # BMP 8bpp RGB
        images = Images('foo', ['files/8rgb.bmp'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

        # BMP 16bpp RGB
        images = Images('foo', ['files/16rgb.bmp'], 0, config=['uint8'])
        self.assertEqual(images.count, 1)
        self.assertTrue(images.images[0][0].max() < 256)

    def test_054(self):
        """ Images - image types """

        # gif - grayscale
        images = Images('foo', ['files/8rgb.gif'], 0, config=['grayscale'])
        self.assertEqual(images.count, 1)

        # gay tiff - color
        images = Images('foo', ['files/16gray.tif'], 0, config=['16bpp'])
        self.assertEqual(images.count, 1)

        # 2 channels - color
        images = Images('foo', ['files/16_2ch.bmp'], 0, config=['grayscale', '16bpp'])
        self.assertEqual(images.count, 1)

        # unsopported format
        images = Images('foo', ['files/16mch.raw'], 0)
        self.assertEqual(images.count, 0)

    def test_055(self):
        """ Images - config setting: normalize """

        # 8bpp
        a = cv2.imread('files/1.jpg')
        images = Images('foo', [a], 0, config=['gray', 'flat', 'norm=pos'])
        self.assertEqual("%.3f" % images.images[0][0][0], "0.420")

        images = Images('foo', [a], 0, config=['gray', 'flat', 'norm=zero'])
        self.assertEqual("%.3f" % images.images[0][0][0], "-0.161")

        images = Images('foo', [a], 0, config=['gray', 'flat', 'norm=std'])
        self.assertEqual("%.3f" % images.images[0][0][0], "-0.062")

        # 16bpp
        images = Images('foo', ['files/16rgb.png'], 0,
                        config=['gray', 'flat', 'norm=pos', '16bpp'])
        self.assertEqual("%.3f" % images.images[0][0][0], "0.116")

        images = Images('foo', ['files/16rgb.png'], 0,
                        config=['gray', 'flat', 'norm=zero', '16bpp'])
        self.assertEqual("%.3f" % images.images[0][0][0], "-0.767")

        images = Images('foo', ['files/16rgb.png'], 0, config=['gray', 'flat', 'norm=std', '16bpp'])
        self.assertEqual("%.3f" % images.images[0][0][0], "-2.156")

    def test_056(self):
        """ Images - property gray """
        images = Images('foo', ['files/1.jpg', 'files/2.jpg'], 0,
                        config=['resize=(50,50)'])
        self.assertEqual(images.images[0].shape, (2, 50, 50, 3))
        images.gray = False
        self.assertEqual(images.images[0].shape, (2, 50, 50, 3))
        images.gray = True
        self.assertEqual(images.images[0].shape, (2, 50, 50))
        images.gray = True
        self.assertEqual(images.images[0].shape, (2, 50, 50))

    def test_057(self):
        """ Images - split - setter - bad arguments """
        images = Images()
        with pytest.raises(TypeError):
            images.split = 'A'
        with pytest.raises(ValueError):
            images.split = 12.6
        with pytest.raises(ValueError):
            images.split = 1.0
        with pytest.raises(TypeError):
            images.split = 0.6, 'a'

    def test_058(self):
        """ Images - split - single class - setter """
        images = Images('foo', ['files/1.jpg', 'files/2.jpg'], 0, config=['store'])
        images.split = 0.5, 12
        self.assertEqual(len(images._train), 1)
        self.assertEqual(len(images._test), 1)
        images = Images('foo', ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'], 0)
        images.split = 0.25, 12
        self.assertEqual(len(images._train), 3)
        self.assertEqual(len(images._test), 1)

        # load
        images = Images()
        images.load('foo')
        images.split = 0.5, 12
        self.assertEqual(len(images._train), 1)
        self.assertEqual(len(images._test), 1)
        images = Images('foo', ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'], 0)
        images.split = 0.25, 12
        self.assertEqual(len(images._train), 3)
        self.assertEqual(len(images._test), 1)
        os.remove('foo.h5')

    def test_059(self):
        """ Images - split - multi class - setter """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/3.jpg', 'files/1.jpg', 'files/2.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1], config=['store'])
        images.split = 0.25, 12
        self.assertEqual(len(images._train), 6)
        self.assertEqual(len(images._test), 2)

        # load
        images = Images()
        images.load('foo')
        images.split = 0.25, 12
        self.assertEqual(len(images._train), 6)
        self.assertEqual(len(images._test), 2)
        os.remove('foo.h5')

    def test_060(self):
        """ Images - split - single class - getter """
        images = Images('foo', ['files/1.jpg', 'files/2.jpg'], 0, config=['store'])
        images.split = 0.5, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (1, 128, 128, 3))
        self.assertEqual(X_test.shape, (1, 128, 128, 3))
        self.assertEqual(Y_train[0][0], [1])
        self.assertEqual(Y_test[0][0], [1])
        self.assertEqual(type(Y_train[0][0]), np.uint8)
        self.assertEqual(type(Y_test[0][0]), np.uint8)
        self.assertEqual(images.classes, {'0': 0})

        # load
        images = Images()
        images.load('foo')
        images.split = 0.5, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (1, 128, 128, 3))
        self.assertEqual(X_test.shape, (1, 128, 128, 3))
        self.assertEqual(Y_train[0][0], [1])
        self.assertEqual(Y_test[0][0], [1])
        self.assertEqual(images.classes, {'0': 0})
        os.remove('foo.h5')

    def test_061(self):
        """ Images - split - multi class - getter """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/3.jpg', 'files/1.jpg', 'files/2.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1], config=['store'])
        images.split = 0.25, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (6, 128, 128, 3))
        self.assertEqual(X_test.shape, (2, 128, 128, 3))
        self.assertEqual(Y_train.shape, (6, 2))
        self.assertEqual(Y_test.shape, (2, 2))
        self.assertEqual(Y_train[0][0], 1)
        self.assertEqual(Y_train[0][1], 0)
        self.assertEqual(images.classes, {'0': 0, '1': 1})

        # load
        images = Images()
        images.load('foo')
        images.split = 0.25, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (6, 128, 128, 3))
        self.assertEqual(X_test.shape, (2, 128, 128, 3))
        self.assertEqual(Y_train.shape, (6, 2))
        self.assertEqual(Y_test.shape, (2, 2))
        self.assertEqual(Y_train[0][0], 1)
        self.assertEqual(Y_train[0][1], 0)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        os.remove('foo.h5')

    def test_062(self):
        """ Images - split - single class - getter, class != 0 """
        images = Images('foo', ['files/1.jpg', 'files/2.jpg'], 1, config=['store'])
        images.split = 0.5, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (1, 128, 128, 3))
        self.assertEqual(X_test.shape, (1, 128, 128, 3))
        self.assertEqual(Y_train[0][0], 1)
        self.assertEqual(Y_test[0][0], 1)
        self.assertEqual(images.classes, {'1': 0})

        # load
        images = Images()
        images.load('foo')
        images.split = 0.5, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (1, 128, 128, 3))
        self.assertEqual(X_test.shape, (1, 128, 128, 3))
        self.assertEqual(Y_train[0][0], 1)
        self.assertEqual(Y_test[0][0], 1)
        self.assertEqual(images.classes, {'1': 0})

    def test_063(self):
        """ Images - split - single string class - setter - string """
        images = Images('foo', ['files/1.jpg', 'files/2.jpg'], 'cat', config=['store'])
        images.split = 0.5, 12
        self.assertEqual(len(images._train), 1)
        self.assertEqual(len(images._test), 1)

        # load
        images = Images()
        images.load('foo')
        images.split = 0.5, 12
        self.assertEqual(len(images._train), 1)
        self.assertEqual(len(images._test), 1)

        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        'cat')
        images.split = 0.25, 12
        self.assertEqual(len(images._train), 3)
        self.assertEqual(len(images._test), 1)
        self.assertEqual(images.classes, {'cat': 0})
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        'cat')
        images.split = 0.25, 12
        self.assertEqual(len(images._train), 3)
        self.assertEqual(len(images._test), 1)
        self.assertEqual(images.classes, {'cat': 0})

        # load
        os.remove('foo.h5')

    def test_064(self):
        """ Images - split - multi class - setter - string """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/3.jpg', 'files/1.jpg', 'files/2.jpg'],
                        ['cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog'],
                        config=['store'])
        images.split = 0.25, 12
        self.assertEqual(len(images._train), 6)
        self.assertEqual(len(images._test), 2)

        # load
        images = Images()
        images.load('foo')
        images.split = 0.25, 12
        self.assertEqual(len(images._train), 6)
        self.assertEqual(len(images._test), 2)
        os.remove('foo.h5')

    def test_065(self):
        """ Images - split - single class - getter - string """
        images = Images('foo', ['files/1.jpg', 'files/2.jpg'], 'cat', config=['store'])
        images.split = 0.5, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (1, 128, 128, 3))
        self.assertEqual(X_test.shape, (1, 128, 128, 3))
        self.assertEqual(Y_train[0], [1])
        self.assertEqual(Y_test[0], [1])
        self.assertEqual(images.classes, {'cat': 0})

        # load
        images = Images()
        images.load('foo')
        images.split = 0.5, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (1, 128, 128, 3))
        self.assertEqual(X_test.shape, (1, 128, 128, 3))
        self.assertEqual(Y_train[0], [1])
        self.assertEqual(Y_test[0], [1])
        self.assertEqual(images.classes, {'cat': 0})
        os.remove('foo.h5')

    def test_066(self):
        """ Images - split - multi class - getter - string """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/3.jpg', 'files/1.jpg', 'files/2.jpg'],
                        ['cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog'],
                        config=['store'])
        images.split = 0.25, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (6, 128, 128, 3))
        self.assertEqual(X_test.shape, (2, 128, 128, 3))
        self.assertEqual(Y_train.shape, (6, 2))
        self.assertEqual(Y_test.shape, (2, 2))
        self.assertEqual(Y_train[0][0], 1)
        self.assertEqual(Y_train[0][1], 0)
        if images.classes['cat'] == 1:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
        else:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})

        # load
        images = Images()
        images.load('foo')
        images.split = 0.25, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (6, 128, 128, 3))
        self.assertEqual(X_test.shape, (2, 128, 128, 3))
        self.assertEqual(Y_train.shape, (6, 2))
        self.assertEqual(Y_test.shape, (2, 2))
        # TODO: not sure if error is due to bug if just random shuffle
        self.assertEqual(Y_train[0][0], 1)
        self.assertEqual(Y_train[0][1], 0)
        if images.classes['cat'] == 1:
            self.assertEqual(images.classes, {'cat': 1, 'dog': 0})
        else:
            self.assertEqual(images.classes, {'cat': 0, 'dog': 1})
        os.remove('foo.h5')

    def test_067(self):
        """ Images - split, getter only -- """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg',
                         'files/8rgb.jpg', 'files/8rgb.png'],
                        'cat')
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (4, 128, 128, 3))
        self.assertEqual(X_test.shape, (1, 128, 128, 3))
        self.assertEqual(Y_train[0], [1])
        self.assertEqual(Y_test[0], [1])
        self.assertEqual(images.classes, {'cat': 0})

        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 0, 0, 1, 1])
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (5, 128, 128, 3))
        self.assertEqual(X_test.shape, (3, 128, 128, 3))
        self.assertEqual(Y_train.shape, (5, 2))
        self.assertEqual(Y_test.shape, (3, 2))
        self.assertEqual(Y_train.dtype, np.uint8)
        self.assertEqual(Y_test.dtype, np.uint8)

    def test_068(self):
        """ Images - split 100% training """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg',
                         'files/8rgb.jpg', 'files/8rgb.png'],
                        'cat')
        images.split = 0.0, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (5, 128, 128, 3))
        self.assertEqual(X_test, None)
        self.assertEqual(Y_train.shape, (5, 1))
        self.assertEqual(Y_test, None)

    def test_069(self):
        """ Images - unevent split -- """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 0, 0, 1, 1])
        images.split = 0.8, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.shape, (2, 128, 128, 3))
        self.assertEqual(X_test.shape, (6, 128, 128, 3))
        self.assertEqual(Y_train.shape, (2, 2))
        self.assertEqual(Y_test.shape, (6, 2))

    def test_070(self):
        """ Images - split - uint8, then normalize -- """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 0, 0, 1, 1], config=['uint8'])
        self.assertEqual(images.dtype, np.uint8)
        images.split = 0.5, 12
        self.assertEqual(images.dtype, np.uint8)
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.dtype, np.float32)
        self.assertEqual(X_test.dtype, np.float32)

        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 0, 0, 1, 1], config=['uint16'])
        self.assertEqual(images.dtype, np.uint16)
        images.split = 0.5, 12
        self.assertEqual(images.dtype, np.uint16)
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(X_train.dtype, np.float32)
        self.assertEqual(X_test.dtype, np.float32)

    def test_071(self):
        """ images - split, shuffle order -- """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1])
        images.split = 0.5, 12
        self.assertEqual(len(images._train), 4)
        self.assertEqual(images._train[0], (0, 0))
        self.assertEqual(images._train[1], (1, 0))
        self.assertEqual(images._train[2], (0, 2))
        self.assertEqual(images._train[3], (1, 2))
        self.assertEqual(len(images._test), 4)
        self.assertEqual(images._test[0], (0, 1))
        self.assertEqual(images._test[1], (0, 3))
        self.assertEqual(images._test[2], (1, 3))
        self.assertEqual(images._test[3], (1, 1))

    def test_072(self):
        """ images - split, more shuffle order -- """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 0, 0, 1, 1])
        images.split = 0.25, 12
        X_train, X_test, Y_train, Y_test = images.split
        self.assertEqual(len(images._train), 5)
        self.assertEqual(len(X_train), 5)
        self.assertEqual(len(X_test), 3)
        self.assertEqual(len(Y_train), 5)
        self.assertEqual(len(Y_test), 3)
        for _ in range(len(images._train)):
            # this is the label index
            y = images._train[_][0]
            self.assertEqual(Y_train[_][y], 1)

    def test_073(self):
        """ Images - split / next -- no image data """
        images = Images()
        with pytest.raises(AttributeError):
            images.split = 0.5
        images = Images()
        with pytest.raises(AttributeError):
            x, x, x, x = images.split
        images = Images()
        with pytest.raises(AttributeError):
            next(images)

    def test_074(self):
        """ Images -next() operator """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], config=['float16'])
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)
        self.assertEqual(next(images), (None, None))
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)

        # load
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], config=['float16', 'store'])
        images = Images()
        images.load('foo')
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)
        self.assertEqual(next(images), (None, None))
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)
        os.remove('foo.h5')

    def test_075(self):
        """ Images -next() operator / normalize """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], config=['uint8'])
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float32)
        self.assertEqual(next(images), (None, None))
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float32)

    def test_076(self):
        """ Images -next() operator / implicit split """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1])
        # loop thru list twice
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float32)
        self.assertEqual(next(images), (None, None))
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float32)

    def test_077(self):
        """ Images - next() - shuffle order -- """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1])
        images.split = 0.5, 12
        x, y = next(images)
        self.assertEqual(list(y), [1, 0])
        x, y = next(images)
        self.assertEqual(list(y), [0, 1])
        x, y = next(images)
        self.assertEqual(list(y), [1, 0])
        x, y = next(images)
        self.assertEqual(list(y), [0, 1])
        # None
        x, y = next(images)
        # New order
        x, y = next(images)
        self.assertEqual(list(y), [0, 1])
        x, y = next(images)
        self.assertEqual(list(y), [1, 0])
        x, y = next(images)
        self.assertEqual(list(y), [1, 0])
        x, y = next(images)
        self.assertEqual(list(y), [0, 1])

    def test_078(self):
        """ Images - minibatch - bad args """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1])
        with pytest.raises(TypeError):
            images.minibatch = 'a'
        with pytest.raises(ValueError):
            images.minibatch = 0
        images = Images()
        with pytest.raises(AttributeError):
            images.minibatch = 2

    def test_079(self):
        """ Images - minibatch - setter """
        # explicit split
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 0, 0, 1, 1])
        images.split = 0.5, 12
        images.minibatch = 2
        self.assertEqual(images._minisz, 2)
        self.assertEqual(images._trainsz, 4)

        # implicit split
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 0, 0, 1, 1])
        images.minibatch = 2
        self.assertEqual(images._minisz, 2)
        self.assertEqual(images._trainsz, 5)

    def test_080(self):
        """ Images - minibatch - getter """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1])
        images.split = 0.5, 12
        images.minibatch = 2

        g = images.minibatch
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 128, 128, 3))
                self.assertEqual(y_batch.shape, (2, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                # second batch
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                # next epoch
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [0, 1])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                else:
                    break

    def test_081(self):
        """ stream file csv with urls paths """
        url = 'https://raw.githubusercontent.com/gapml/CV/master/tests/files/fp_urls.csv'
        images = Images('foo', url,
                        config=['label_col=1', 'image_col=0', 'resize=(50,50)', 'header'])
        if images.count == 25:
            self.assertEqual(images.count, 25)
            self.assertEqual(images.fail, 0)
            self.assertEqual(len(images.errors), 0)
            self.assertEqual(len(images.images), 5)
            self.assertEqual(len(images.labels), 5)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(images.classes, {'daisy': 0,
                                          'dandelion': 1,
                                          'roses': 2,
                                          'sunflowers': 3,
                                          'tulips': 4})
        if images.count == 25:
            self.assertEqual(len(images.labels[0]), 5)
            self.assertEqual(len(images.images[0]), 5)
        self.assertEqual(images.images[0][0].shape, (50, 50, 3))
        bad_url = 'https://raw.githubusercontent.com/gapml/CV/master/tests/file/fp_urls.csv'
        with pytest.raises(OSError):
            images = Images('foo', bad_url,
                            config=['label_col=1', 'image_col=0',
                                    'resize=(50,50)', 'header'])

    def test_082(self):
        """ stream file json with urls paths """
        url = 'https://raw.githubusercontent.com/gapml/CV/master/tests/files/fp_urls.json'
        images = Images('foo', url,
                        config=['label_key=image_label', 'image_key=image_key', 'resize=(50,50)'])
        if images.count == 25:
            self.assertEqual(images.count, 25)
            self.assertEqual(images.fail, 0)
            self.assertEqual(len(images.errors), 0)
            self.assertEqual(len(images.images), 5)
            self.assertEqual(len(images.labels), 5)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(images.classes, {'daisy': 0,
                                          'dandelion': 1,
                                          'roses': 2,
                                          'sunflowers': 3,
                                          'tulips': 4})
        if images.count == 25:
            self.assertEqual(len(images.labels[0]), 5)
            self.assertEqual(len(images.images[0]), 5)
        self.assertEqual(images.images[0][0].shape, (50, 50, 3))
        bad_url = 'https://raw.githubusercontent.com/gapml/CV/master/tests/file/fp_urls.json'
        with pytest.raises(OSError):
            images = Images('foo', bad_url,
                            config=['label_key=image_label',
                                    'image_key=image_key',
                                    'resize=(50,50)'])

    def test_083(self):
        """ Images - stratify - bad args """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1])
        with pytest.raises(AttributeError):
            images.stratify = 2, 0.5, 6, 6
        with pytest.raises(AttributeError):
            images.stratify = 'a'
        with pytest.raises(TypeError):
            images.stratify = 2, 'a'
        with pytest.raises(TypeError):
            images.stratify = 2, 0.6, 'a'
        with pytest.raises(ValueError):
            images.stratify = 0
        with pytest.raises(ValueError):
            images.stratify = 2, 1.0
        with pytest.raises(ValueError):
            images.stratify = 1, 0.5
        images = Images()
        with pytest.raises(AttributeError):
            images.stratify = 1, 0.5

    def test_084(self):
        """ Images - stratify - setter - batch size """
        images = Images('foo', ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                                'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1])
        images.stratify = 2
        self.assertEqual(images._minisz, 2)
        images = Images('foo', ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                                'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1])
        images.stratify = 2, 0.5
        self.assertEqual(images._minisz, 2)
        images = Images('foo', ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                                'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1])
        images.stratify = 2, 0.5, 12
        self.assertEqual(images._minisz, 2)

    def test_085(self):
        """ Images - stratify - setter - getter """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1])
        images.stratify = 2, 0.5, 12

        g = images.stratify
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 128, 128, 3))
                self.assertEqual(y_batch.shape, (2, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                else:
                    break

    def test_086(self):
        """ Images - += Images - invalid """
        images1 = Images('foo', ['files/1.jpg', 'files/2.jpg'], 0,
                         config=['resize=(50,50)'])
        with pytest.raises(TypeError):
            images1 += 1
        images1 = Images('foo', ['files/1.jpg', 'files/2.jpg'], 0,
                         config=['resize=(50,50)'])
        images2 = Images('foo', ['files/3.jpg', 'files/8rgb.jpg'], 0,
                         config=['resize=(40,50)'])
        with pytest.raises(AttributeError):
            images1 += images2
        images1 = Images('foo', ['files/1.jpg', 'files/2.jpg'], 0,
                         config=['resize=(50,50)', 'gray'])
        images2 = Images('foo', ['files/3.jpg', 'files/8rgb.jpg'], 0,
                         config=['resize=(50,50)'])
        with pytest.raises(AttributeError):
            images1 += images2
        images1 = Images('foo', ['files/1.jpg', 'files/2.jpg'], 0,
                         config=['resize=(50,50)', 'uint8'])
        images2 = Images('foo', ['files/3.jpg', 'files/8rgb.jpg'], 0,
                         config=['resize=(50,50)'])
        with pytest.raises(AttributeError):
            images1 += images2

    def test_087(self):
        """ Images - += Images - same single class """
        images1 = Images('foo', ['files/1.jpg', 'files/2.jpg'], 0,
                         config=['resize=(50,50)'])
        images2 = Images('foo', ['files/1.jpg', 'files/8rgb.jpg', 'nonexist.jpg'], 0,
                         config=['resize=(50,50)'])
        images1 += images2
        self.assertEqual(images1.count, 4)
        self.assertEqual(images1.fail, 1)
        self.assertEqual(images1.classes, {'0': 0})
        self.assertEqual(len(images1), 1)
        self.assertEqual(images1.images[0].shape, (4, 50, 50, 3))
        self.assertEqual(images1.labels[0].shape, (4, ))

    def test_088(self):
        """ Images - += Images - different class """
        images1 = Images('foo', ['files/1.jpg', 'files/2.jpg'], 0,
                         config=['resize=(50,50)'])
        images2 = Images('foo', ['files/1.jpg', 'files/8rgb.jpg', 'nonexist.jpg'], 1,
                         config=['resize=(50,50)'])
        self.assertEqual(images2.classes, {'1': 0})
        images1 += images2
        self.assertEqual(images1.count, 4)
        self.assertEqual(images1.fail, 1)
        self.assertEqual(images1.classes, {'0': 0, '1': 1})
        self.assertEqual(len(images1), 2)
        self.assertEqual(images1.images[0].shape, (2, 50, 50, 3))
        self.assertEqual(images1.images[1].shape, (2, 50, 50, 3))
        self.assertEqual(images1.labels[0].shape, (2, ))
        self.assertEqual(images1.labels[1].shape, (2, ))
        self.assertEqual(images1.labels[0][0], 0)
        self.assertEqual(images1.labels[1][0], 1)

    def test_089(self):
        """ Images - += Images - different metadata """
        images1 = Images('foo', ['files/1.jpg', 'files/2.jpg'], 0,
                         config=['resize=(50,50)', 'author=sam', 'src=x', 'desc=aa', 'license=c0'])
        images2 = Images('foo', ['files/1.jpg', 'files/8rgb.jpg'], 0,
                         config=['resize=(50,50)', 'author=sue', 'src=y', 'desc=bb', 'license=c1'])
        images1 += images2
        self.assertEqual(images1.author, 'sam,sue')
        self.assertEqual(images1.src, 'x,y')
        self.assertEqual(images1.desc, 'aa,bb')
        self.assertEqual(images1.license, 'c0,c1')

    def test_090(self):
        """ Images Constructor - no images, augment argument """
        images = Images(augment=None)
        images = Images(augment=[])
        images = Images(augment=['edge'])
        images = Images(augment=['denoise'])
        with pytest.raises(TypeError):
            images = Images(augment=7)
        with pytest.raises(TypeError):
            images = Images(augment='7')
        with pytest.raises(AttributeError):
            images = Images(augment=[7])
        with pytest.raises(AttributeError):
            images = Images(augment=['foo'])
        images = Images(augment=['zoom=1.5'])
        with pytest.raises(AttributeError):
            images = Images(augment=['zoom='])
        with pytest.raises(AttributeError):
            images = Images(augment=['zoom=-1'])
        with pytest.raises(AttributeError):
            images = Images(augment=['zoom=abc'])
        images = Images(augment=['flip=horizontal'])
        images = Images(augment=['flip=vertical'])
        images = Images(augment=['flip=both'])
        with pytest.raises(AttributeError):
            images = Images(augment=['flip=a'])
        with pytest.raises(AttributeError):
            images = Images(augment=['flip=1'])
        images = Images(augment=['rotate=-90,90'])
        with pytest.raises(AttributeError):
            images = Images(augment=['rotate='])
        with pytest.raises(AttributeError):
            images = Images(augment=['rotate=1'])
        with pytest.raises(AttributeError):
            images = Images(augment=['rotate=2,a'])
        with pytest.raises(AttributeError):
            images = Images(augment=['rotate=0,360'])
        with pytest.raises(AttributeError):
            images = Images(augment=['rotate=-360,0'])
        images = Images(augment=['contrast=2.0'])
        images = Images(augment=['brightness=50'])
        images = Images(augment=['brightness=50', 'contrast=2.0'])
        images = Images(augment=['contrast=2'])
        images = Images(augment=['brightness=50.5'])
        images = Images(augment=['brightness=50.5', 'contrast=2'])
        with pytest.raises(AttributeError):
            images = Images(augment=['contrast='])
        with pytest.raises(AttributeError):
            images = Images(augment=['contrast=-1.0'])
        with pytest.raises(AttributeError):
            images = Images(augment=['contrast=4.0'])
        with pytest.raises(AttributeError):
            images = Images(augment=['contrast=abc'])
        with pytest.raises(AttributeError):
            images = Images(augment=['brightness='])
        with pytest.raises(AttributeError):
            images = Images(augment=['brightness=-1'])
        with pytest.raises(AttributeError):
            images = Images(augment=['brightness=101'])
        with pytest.raises(AttributeError):
            images = Images(augment=['brightness=abc'])

    def test_091(self):
        """ Images -next() operator with augmentation """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], augment=['flip=horizontal'])
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))
        # next epoch
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))

        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], augment=['flip=vertical'])
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))
        # next epoch
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))

        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], augment=['flip=both'])
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))
        # next epoch
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))

        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], augment=['zoom=1.5'])
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))
        # next epoch
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))

        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], augment=['rotate=-30,60'])
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))
        # next epoch
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))

        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], augment=['rotate=-30,60', 'flip=vertical', 'zoom=2'])
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))
        # next epoch
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))

    def test_092(self):
        """ Images - augmentation - minibatch """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1], augment=['flip=vertical'])
        images.split = 0.5, 12

        images.minibatch = 2
        g = images.minibatch
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 128, 128, 3))
                self.assertEqual(y_batch.shape, (2, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                # second batch
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [0, 1])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                # next epoch
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                else:
                    break

    def test_093(self):
        """ Images - stratify - augmentation """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1], augment=['flip=both'])
        images.stratify = 2, 0.5, 12

        g = images.stratify
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (4, 128, 128, 3))
                self.assertEqual(y_batch.shape, (4, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                # second batch
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                # next epoch
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                else:
                    break

    def test_094(self):
        """ Images - augmentation - minibatch - uint8 """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1], config=['uint8'], augment=['flip=vertical'])
        images.split = 0.5, 12
        images.minibatch = 2

        g = images.minibatch
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 128, 128, 3))
                self.assertEqual(y_batch.shape, (2, 2))
                self.assertEqual(x_batch.dtype, np.float32)
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                # second batch
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [0, 1])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                # next epoch
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                else:
                    break

    def test_095(self):
        """ Images - stratify - augmentation - uint16 """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1], augment=['flip=both'], config=['uint16'])
        images.stratify = 2, 0.5, 12

        g = images.stratify
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (4, 128, 128, 3))
                self.assertEqual(y_batch.shape, (4, 2))
                self.assertEqual(x_batch.dtype, np.float32)
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                # second batch
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                # next epoch
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                else:
                    break

    def test_096(self):
        """ Images - augmentation - minibatch - uint8 """
        images = Images('foo', 'files/fp_urls.csv',
                        config=['uint8',
                                'resize=(50,50)',
                                'label_col=1',
                                'image_col=0',
                                'header'],
                        augment=['flip=both',
                                 'edge',
                                 'zoom=0.5',
                                 'rotate=-30,60',
                                 'denoise',
                                 'brightness=0',
                                 'contrast=1.0'
                                 ])
        images.split = 0.2, 12
        images.minibatch = 2

        g = images.minibatch
        if True:
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 50, 50, 3))
                self.assertEqual(y_batch.shape, (2, 5))
                self.assertEqual(x_batch.dtype, np.float32)
                break

    def test_097(self):
        """ Images - stratify - augmentation - uint8 """
        images = Images('foo', 'files/fp_urls.csv',
                        config=['uint8',
                                'resize=(50,50)',
                                'label_col=1',
                                'image_col=0',
                                'header'],
                        augment=['flip=both',
                                 'edge',
                                 'zoom=0.5',
                                 'rotate=-30,60',
                                 'denoise',
                                 'brightness=0',
                                 'contrast=1.0'
                                 ])
        images.stratify = 5, 0.2, 12

        g = images.stratify
        if True:
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (10, 50, 50, 3))
                self.assertEqual(y_batch.shape, (10, 5))
                self.assertEqual(x_batch.dtype, np.float32)
                break

    def test_098(self):
        """ Images - augmentation - minibatch - float32 """
        images = Images('foo', 'files/fp_urls.csv',
                        config=['float32',
                                'resize=(50,50)',
                                'label_col=1',
                                'image_col=0',
                                'header'],
                        augment=['flip=both',
                                 'edge',
                                 'zoom=0.5',
                                 'rotate=-30,60',
                                 'denoise',
                                 'brightness=0',
                                 'contrast=1.0'
                                 ])
        images.split = 0.2, 12
        images.minibatch = 2

        g = images.minibatch
        if True:
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 50, 50, 3))
                self.assertEqual(y_batch.shape, (2, 5))
                self.assertEqual(x_batch.dtype, np.float32)
                break

    def test_099(self):
        """ Images - stratify - augmentation - float32 """
        images = Images('foo', 'files/fp_urls.csv',
                        config=['float32',
                                'resize=(50,50)',
                                'label_col=1',
                                'image_col=0',
                                'header'],
                        augment=['flip=both',
                                 'edge',
                                 'zoom=0.5',
                                 'rotate=-30,60',
                                 'denoise',
                                 'brightness=0',
                                 'contrast=1.0'
                                 ])
        images.stratify = 5, 0.2, 12
        # First batch
        g = images.stratify
        if True:
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (10, 50, 50, 3))
                self.assertEqual(y_batch.shape, (10, 5))
                self.assertEqual(x_batch.dtype, np.float32)
                break

    def test_100(self):
        """ Images - augmentation - minibatch - float16 """
        images = Images('foo', 'files/fp_urls.csv',
                        config=['float16',
                                'resize=(50,50)',
                                'label_col=1',
                                'image_col=0',
                                'header'],
                        augment=['flip=both',
                                 'edge',
                                 'zoom=0.5',
                                 'rotate=-30,60',
                                 'denoise',
                                 'brightness=0',
                                 'contrast=1.0'
                                 ])
        images.split = 0.2, 12
        images.minibatch = 2

        g = images.minibatch
        if True:
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 50, 50, 3))
                self.assertEqual(y_batch.shape, (2, 5))
                self.assertEqual(x_batch.dtype, np.float16)
                break

    def test_101(self):
        """ Images - stratify - augmentation - float16 """
        images = Images('foo', 'files/fp_urls.csv',
                        config=['float16',
                                'resize=(50,50)',
                                'label_col=1',
                                'image_col=0',
                                'header'],
                        augment=['flip=both',
                                 'edge',
                                 'zoom=0.5',
                                 'rotate=-30,60',
                                 'denoise',
                                 'brightness=0',
                                 'contrast=1.0'
                                 ])
        images.stratify = 5, 0.2, 12
        # First batch
        g = images.stratify
        if True:
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (10, 50, 50, 3))
                self.assertEqual(y_batch.shape, (10, 5))
                self.assertEqual(x_batch.dtype, np.float16)
                break

    def test_102(self):
        """ Images - memory - load/stream """

        # one class
        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)
        c = np.asarray([a, a, a, a, a])
        images = Images('foo', c, 0, config=['resize=(30,50)', 'store'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))
        self.assertEqual(images.labels[0][0], 0)

        # one class, load
        images = Images(config=['stream'])
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 0)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {'0': 0})
        self.assertEqual(images.labels[0][0], 0)
        # temp
        images._hf.close()

        # multi-class
        images = Images('foo', c, [0, 1, 0, 1, 1], config=['resize=(30,50)', 'store'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 2)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(images.images[0][0].shape, (30, 50, 3))
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)

        # multi class, load
        images = Images(config=['stream'])
        images.load('foo')
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 5)
        self.assertEqual(images.shape, (30, 50))
        self.assertEqual(len(images.images), 0)
        self.assertEqual(len(images.labels), 2)
        self.assertEqual(images.classes, {'0': 0, '1': 1})
        self.assertEqual(images.labels[0][0], 0)
        self.assertEqual(images.labels[1][0], 1)
        # temp
        images._hf.close()

        os.remove('foo.h5')

    def test_103(self):
        """ Images - split - stream """
        a = cv2.imread('files/1.jpg', cv2.IMREAD_COLOR)
        c = np.asarray([a, a, a, a, a])
        images = Images('foo', c, [0, 1, 0, 1, 1], config=['resize=(30,50)', 'stream'])
        images.split = 0.5
        with pytest.raises(AttributeError):
            x_train, x_test, y_train, y_test = images.split

        images = Images(config=['stream'])
        images.load('foo')
        with pytest.raises(AttributeError):
            x_train, x_test, y_train, y_test = images.split
        # temp
        images._hf.close()

        os.remove('foo.h5')

    def test_104(self):
        """ Images - minibatch - stream """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1], config=['stream'])
        # during store
        images.split = 0.5, 12
        images.minibatch = 2
        if True:
            step = 0
            g = images.minibatch
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 128, 128, 3))
                self.assertEqual(y_batch.shape, (2, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                # second batch
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                # next epoch
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [0, 1])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                else:
                    break

        # temp
        images._hf.close()

        # during load
        images = Images(config=['stream'])
        images.load('foo')
        images.split = 0.5, 12
        images.minibatch = 2
        if True:
            step = 0
            g = images.minibatch
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 128, 128, 3))
                self.assertEqual(y_batch.shape, (2, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                # second batch
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                # next epoch
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [0, 1])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                else:
                    break

        # temp
        images._hf.close()

        os.remove('foo.h5')

    def test_105(self):
        """ Images - augmentation - minibatch - stream """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1], augment=['flip=vertical'], config=['stream'])
        images.split = 0.5, 12
        images.minibatch = 2
        g = images.minibatch
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 128, 128, 3))
                self.assertEqual(y_batch.shape, (2, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                # second batch
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [0, 1])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                # next epoch
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                else:
                    break

        # temp
        images._hf.close()

        # during load
        images = Images(config=['stream'], augment=['flip=vertical'])
        images.load('foo')
        images.split = 0.5, 12
        images.minibatch = 2
        g = images.minibatch
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 128, 128, 3))
                self.assertEqual(y_batch.shape, (2, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                # second batch
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [0, 1])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                # next epoch
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                else:
                    break

        # temp
        images._hf.close()

        os.remove('foo.h5')

    def test_106(self):
        """ Images - stratify - stream """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1], config=['stream'])
        images.stratify = 2, 0.5, 12

        g = images.stratify
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 128, 128, 3))
                self.assertEqual(y_batch.shape, (2, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                else:
                    break

        # temp
        images._hf.close()

        # during load
        images = Images(config=['stream'])
        images.load('foo')
        images.stratify = 2, 0.5, 12

        g = images.stratify
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (2, 128, 128, 3))
                self.assertEqual(y_batch.shape, (2, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [0, 1])
                else:
                    break

        # temp
        images._hf.close()

        os.remove('foo.h5')

    def test_107(self):
        """ Images - stratify - augmentation - stream """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg',
                         'files/8rgb.png', 'files/1.jpg', 'files/2.jpg', 'files/3.jpg'],
                        [0, 0, 0, 0, 1, 1, 1, 1], augment=['flip=both'], config=['stream'])
        images.stratify = 2, 0.5, 12

        g = images.stratify
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (4, 128, 128, 3))
                self.assertEqual(y_batch.shape, (4, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                else:
                    break

        # temp
        images._hf.close()

        # during load
        images = Images(config=['stream'], augment=['flip=both'])
        images.load('foo')
        images.stratify = 2, 0.5, 12

        g = images.stratify
        if True:
            step = 0
            for x_batch, y_batch in g:
                self.assertEqual(x_batch.shape, (4, 128, 128, 3))
                self.assertEqual(y_batch.shape, (4, 2))
                step += 1
                # first batch
                if step == 1:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                elif step == 2:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                elif step == 3:
                    self.assertEqual(list(y_batch[0]), [1, 0])
                    self.assertEqual(list(y_batch[1]), [1, 0])
                    self.assertEqual(list(y_batch[2]), [0, 1])
                    self.assertEqual(list(y_batch[3]), [0, 1])
                else:
                    break

        # temp
        images._hf.close()

        os.remove('foo.h5')

    def test_108(self):
        """ Images -next() operator - stream """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], config=['float16', 'stream'])
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)
        self.assertEqual(next(images), (None, None))
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)

        # temp
        images._hf.close()

        # load
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], config=['float16', 'store'])
        images = Images()
        images.load('foo')
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)
        self.assertEqual(next(images), (None, None))
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)

        # temp
        images._hf.close()

        # during load
        images = Images(config=['stream'])
        images.load('foo')
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)
        self.assertEqual(next(images), (None, None))
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)

        # temp
        images._hf.close()

        # load
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], config=['float16', 'store'])
        images = Images()
        images.load('foo')
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)
        self.assertEqual(next(images), (None, None))
        for _ in range(2):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
            self.assertEqual(x.dtype, np.float16)

        # temp
        images._hf.close()

        os.remove('foo.h5')

    def test_109(self):
        """ Images -next() operator with augmentation - stream """
        images = Images('foo',
                        ['files/1.jpg', 'files/2.jpg', 'files/3.jpg', 'files/8rgb.jpg'],
                        [0, 0, 1, 1], augment=['flip=horizontal'], config=['stream'])
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))
        # next epoch
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))

        # temp
        images._hf.close()

        # during load
        images = Images(config=['stream'], augment=['flip=horizontal'])
        images.load('foo')
        images.split = 0.5, 12
        # loop thru list twice
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))
        # next epoch
        for _ in range(4):
            x, y = next(images)
            self.assertEqual(x.shape, (128, 128, 3))
            self.assertEqual(y.shape, (2,))
        self.assertEqual(next(images), (None, None))

        # temp
        images._hf.close()

        os.remove('foo.h5')

    def test_110(self):
        """ data type when streaming """
        images = Images('foo', ['files/1.jpg'], 0, config=['store', 'uint8'])
        x1 = images._data[0][0][0][0][0]
        images = Images('foo', ['files/1.jpg'], 0, config=['stream', 'uint8'])
        images = Images()
        images.load('foo')
        x2 = images._data[0][0][0][0][0]
        self.assertEqual(x1, x2)

        images = Images('foo', ['files/1.jpg'], 0, config=['store', 'uint16'])
        x1 = images._data[0][0][0][0][0]
        images = Images('foo', ['files/1.jpg'], 0, config=['stream', 'uint16'])
        images = Images()
        images.load('foo')
        x2 = images._data[0][0][0][0][0]
        self.assertEqual(x1, x2)

        images = Images('foo', ['files/1.jpg'], 0, config=['store', 'float16'])
        x1 = images._data[0][0][0][0][0]
        images = Images('foo', ['files/1.jpg'], 0, config=['stream', 'float16'])
        images = Images()
        images.load('foo')
        x2 = images._data[0][0][0][0][0]
        self.assertEqual(x1, x2)

        images = Images('foo', ['files/1.jpg'], 0, config=['store', 'float32'])
        x1 = images._data[0][0][0][0][0]
        images = Images('foo', ['files/1.jpg'], 0, config=['stream', 'float32'])
        images = Images()
        images.load('foo')
        x2 = images._data[0][0][0][0][0]
        self.assertEqual(x1, x2)

        images = Images('foo', ['files/1.jpg'], 0, config=['store', 'float64'])
        x1 = images._data[0][0][0][0][0]
        images = Images('foo', ['files/1.jpg'], 0, config=['stream', 'float64'])
        images = Images()
        images.load('foo')
        x2 = images._data[0][0][0][0][0]
        self.assertEqual(x1, x2)
        os.remove('foo.h5')

    def test_111(self):
        ''' memory labels - empty '''
        a = cv2.imread('files/1.jpg', cv2.IMREAD_GRAYSCALE)

        # list
        i = np.asarray([a])
        with pytest.raises(AttributeError):
            images = Images('foo', i, [], config=['resize=(50,50)'])

        # numpy array
        l = np.asarray([])
        with pytest.raises(AttributeError):
            images = Images('foo', i, l, config=['resize=(50,50)'])

    def test_112(self):
        ''' memory numpy - different int types for labels '''

        # uint8
        a = cv2.imread('files/1.jpg', cv2.IMREAD_GRAYSCALE)
        i = np.asarray([a])
        l = np.asarray([0]).astype(np.uint8)
        images = Images('foo', i, l, config=['resize=(50,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(images.labels[0][0], 0)

        # uint16
        a = cv2.imread('files/1.jpg', cv2.IMREAD_GRAYSCALE)
        i = np.asarray([a])
        l = np.asarray([0]).astype(np.uint16)
        images = Images('foo', i, l, config=['resize=(50,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(images.labels[0][0], 0)

        # uint32
        a = cv2.imread('files/1.jpg', cv2.IMREAD_GRAYSCALE)
        i = np.asarray([a])
        l = np.asarray([0]).astype(np.uint32)
        images = Images('foo', i, l, config=['resize=(50,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(images.labels[0][0], 0)

        # int8
        a = cv2.imread('files/1.jpg', cv2.IMREAD_GRAYSCALE)
        i = np.asarray([a])
        l = np.asarray([0]).astype(np.int8)
        images = Images('foo', i, l, config=['resize=(50,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(images.labels[0][0], 0)

        # int16
        a = cv2.imread('files/1.jpg', cv2.IMREAD_GRAYSCALE)
        i = np.asarray([a])
        l = np.asarray([0]).astype(np.uint16)
        images = Images('foo', i, l, config=['resize=(50,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(images.labels[0][0], 0)

        # int32
        a = cv2.imread('files/1.jpg', cv2.IMREAD_GRAYSCALE)
        i = np.asarray([a])
        l = np.asarray([0]).astype(np.int32)
        images = Images('foo', i, l, config=['resize=(50,50)'])
        self.assertEqual(images.fail, 0)
        self.assertEqual(images.errors, [])
        self.assertEqual(images.count, 1)
        self.assertEqual(images.shape, (50, 50))
        self.assertEqual(len(images.images), 1)
        self.assertEqual(len(images.labels), 1)
        self.assertEqual(images.classes, {0: 0})
        self.assertEqual(images.labels[0][0], 0)

    def test_113(self):
        ''' resize= None '''
        pass
