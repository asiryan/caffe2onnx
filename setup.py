import setuptools
import os

# package metadata
NAME = 'caffe2onnx'
VERSION = '1.0.0.1'
DESCRIPTION = 'Convert Caffe models to ONNX.'
LICENSE = 'MIT'
GIT = 'https://github.com/asiryan/caffe-onnx'
PYTHON = '>=3.5'

# directory
this = os.path.dirname(__file__)

# readme
with open(os.path.join(this, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# requirements
with open(os.path.join(this, 'requirements.txt'), "r") as f:
    requirements = [_ for _ in [_.strip("\r\n ")
                                for _ in f.readlines()] if _ is not None]
# setup tools
setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    license=LICENSE,
    packages=setuptools.find_packages(),
    python_requires=PYTHON,
    author='Valery Asiryan',
    author_email='dmc5mod@yandex.ru',
    url=GIT,
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Topic :: Software Development :: Libraries',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)