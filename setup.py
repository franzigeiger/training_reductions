#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "xarray<=0.12",
    "brain-score @ git+https://github.com/brain-score/brain-score",
    "model-tools @ git+https://github.com/brain-score/model-tools",
    "numpy",
    "result_caching @ git+https://github.com/mschrimpf/result_caching",
    "networkx",
    "tqdm",
    "gitpython",
    "scikit-image",
    "seaborn",
    "matplotlib",
    "sklearn",
    "torch",
    "torchvision",
    "cornet @ git+https://github.com/dicarlolab/CORnet",
    "bagnets @ git+https://github.com/mschrimpf/bag-of-local-features-models.git",
    "texture_vs_shape @ git+https://github.com/mschrimpf/texture-vs-shape.git",
    "Fixing-the-train-test-resolution-discrepancy-scripts @ git+https://github.com/mschrimpf/FixRes.git",
    "dcgan @ git+https://github.com/franzigeiger/dcgan.git"
]

setup(
    name='weight_initialization',
    version='0.1.0',
    description="A project to evaluate the impact of weight initialization with no or very little training on test accuracy",
    long_description=readme,
    author="Franziska Geiger",
    author_email='fgeiger@mit.edu',
    url='https://github.com/franzigeiger/',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='candidate-models brain-score',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
)
