dfrom setuptools import setup, find_packages

description = "Utilities for D2S sea level prediction"
version = "0.1"

setup(
    name="utils",
    url="https://github.com/andrewbrettin/zos_predictability_aies",
    description=description,
    author="@andrewbrettin",
    packages=find_packages()
)