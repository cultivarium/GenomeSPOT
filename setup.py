#!/usr/bin/env python

from setuptools import setup, find_packages

from src._version import __version__

setup(
    name="Genome-OPTS",
    version=__version__,
    description="Estimate oxygen, pH, temp, and salinity preferences from genomes",
    url="https://github.com/cultivarium/Genome-OPTS",
    author="Tyler Barnum",
    author_email="tyler@cultivarium.org",
    license="MIT License",
    package_data={"Genome-OPTS": []},
    packages=find_packages(exclude=["tests"]),
    scripts=["src/"],
    python_requires=">=3.8.16",
    install_requires=[],
    zip_safe=False,
)
