# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

from jaxclust.version import __version__

project = 'JAXClust'
copyright = '2023, Lawrence Stewart'
author = 'Lawrence Stewart'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = ['myst_parser',
              'sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'nbsphinx',
              'sphinx_rtd_theme',
              'sphinxcontrib.bibtex'
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# force sphinx to use order seen in .py files
autodoc_member_order = 'bysource'


# bibtext vars
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrt'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


