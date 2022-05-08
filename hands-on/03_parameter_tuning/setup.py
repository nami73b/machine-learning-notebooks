from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'cloudml-hypertune'
]

setup(
    name='mfashion_keras',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
)
