#!/usr/bin/env python

from distutils.core import setup

setup(name = 'gemi',
      version = '0.1',
      description = 'EMI geometries',
      author = 'Miroslav Kuchta',
      author_email = 'miroslav.kuchta@gmail.com',
      url = 'https://github.com/mirok/gemi.git',
      packages = ['gemi'],
      package_dir = {'gemi': 'gemi'}
)
