from __future__ import absolute_import
from setuptools import setup, find_packages
from os import path

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir, 'denoiseg', 'version.py')) as f:
    exec(f.read())

with open(path.join(_dir, 'README.md')) as f:
    long_description = f.read()

setup(name='denoiseg',
      version=__version__,
      description='DenoiSeg',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/juglab/DenoiSeg/',
      author='Tim-Oliver Buchholz, Mangal Prakash, Alexander Krull, Florian Jug',
      author_email='tim-oliver.buchholz@fmi.ch, prakash@mpi-cbg.de, a.f.f.krull@bham.ac.uk, florian.jug@fht.org',
      license='BSD 3-Clause License',
      packages=find_packages(),

      project_urls={
          'Repository': 'https://github.com/juglab/DenoiSeg/',
      },

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
      ],

      install_requires=[
          "n2v>=0.3.2",
          "wrapt<=1.12.1",
          "pathlib2;python_version<'3'",
          "backports.tempfile;python_version<'3.4'",
          "csbdeep>=0.7.2,<0.8.0",
          "numpy",
          "numba",
          "scikit-learn",
          "scikit-image"
      ],
      )
