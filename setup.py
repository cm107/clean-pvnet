from setuptools import setup, find_packages
import clean_pvnet

packages = find_packages(
        where='.',
        include=['clean_pvnet*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='clean_pvnet',
    version=clean_pvnet.__version__,
    description='Fork of clean-pvnet',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cm107/clean-pvnet",
    author='Clayton Mork',
    author_email='mork.clayton3@gmail.com',
    license='MIT License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch',
        'torchvision',
        'Cython',
        'yacs',
        'numpy',
        'opencv-python',
        'tqdm',
        'pycocotools',
        'matplotlib',
        'plyfile==0.6',
        'scikit-image==0.14.2',
        'PyOpenGL==3.1.1a1',
        'ipdb==0.11',
        'cyglfw3==3.1.0.2',
        'progressbar==2.5',
        'tensorboardX==1.2',
        'cffi==1.11.5',
        'transforms3d',
        'glumpy',
        'tensorboard',
        'pandas',
        'pylint==2.4.4',
        'pyclay-common_utils @ https://github.com/cm107/common_utils/archive/0.2.4+py36.zip',
        'pyclay-annotation_utils @ https://github.com/cm107/annotation_utils/archive/0.3.7+py36.zip'
    ],
    python_requires='==3.6.*'
)

# Build csrc
import os
for target in ['ransac_voting']:
    os.system(f'pushd .; cd clean_pvnet/csrc/{target}; python setup.py build_ext --inplace; popd')