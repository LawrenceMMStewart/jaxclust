"""Setup script for JAXClust."""

import os
from setuptools import find_packages
from setuptools import setup


folder = os.path.dirname(__file__)
version_path = os.path.join(folder, "jaxclust", "version.py")

__version__ = None
with open(version_path) as f:
  exec(f.read(), globals())

req_path = os.path.join(folder, "requirements.txt")
install_requires = []
if os.path.exists(req_path):
  with open(req_path) as fp:
    install_requires = [line.strip() for line in fp]

readme_path = os.path.join(folder, "README.md")
readme_contents = ""
if os.path.exists(readme_path):
  with open(readme_path) as fp:
    readme_contents = fp.read().strip()

setup(
    name="jaxclust",
    version=__version__,
    description="Hardware accelerated, differentiable clustering in JAX.",
    author="Lawrence Stewart",
    author_email="no-contact@pm.com",
    url="https://github.com/LawrenceMMStewart/jaxclust",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    packages=find_packages(),
    package_data={},
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="clustering, differentiable, deep learning, jax",
    requires_python=">=3.8",
)