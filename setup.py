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
        'pylint==2.4.4',
    ],
    python_requires='>=3.7'
)

# Build csrc
import os
for target in ['ransac_voting']:
    os.system(f'pushd .; cd clean_pvnet/csrc/{target}; python setup.py build_ext --inplace; popd')