#!/usr/bin/env python

from setuptools import setup, find_packages

from src._version import __version__

setup(
    name="genome_opts",
    version=__version__,
    description="Estimate oxygen, pH, temp, and salinity preferences from genomes",
    url="https://github.com/cultivarium/Genome-OPTS",
    author="Tyler Barnum",
    author_email="tyler@cultivarium.org",
    license="MIT License",
    package_data={"genome_opts": []},
    packages=find_packages(exclude=["tests"]),
    scripts=["src/"],
    python_requires=">=3.8.16",
    install_requires=[
        "biopython>=1.81",
        "hmmlearn==0.3.0",
        "scikit-learn==1.2.2",
        "numpy>=1.23.5",
        "pandas>=1.5.3",
        "bacdive>=0.2",
    ],
    zip_safe=False,
)
