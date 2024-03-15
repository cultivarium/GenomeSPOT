#!/usr/bin/env python

from genome_spot._version import __version__
from setuptools import (
    find_packages,
    setup,
)


setup(
    name="genome_spot",
    version=__version__,
    description="Predict oxygen, temp, salinity, and pH preferences of bacteria and archaea from a genome",
    url="https://github.com/cultivarium/GenomeSPOT",
    author="Tyler Barnum",
    author_email="tyler@cultivarium.org",
    license="MIT License",
    package_data={"genome_spot": []},
    packages=find_packages(exclude=["tests"]),
    scripts=["genome_spot/"],
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
