# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import aerosandbox as asb
import sys, os
from pathlib import Path

# sys.path.insert(0,str(asb._asb_root.absolute()))
for x in os.walk(str(asb._asb_root.absolute())):
  sys.path.insert(0, x[0])

sys.path.insert(0, str(
    Path(__file__).parent.parent.absolute()
))

project = 'AeroSandbox'
copyright = '2023, Peter Sharpe'
author = 'Peter Sharpe'
release = asb.__version__

master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.duration',
    'autoapi.extension',
]
# autosummary_generate = True
autoapi_type = 'python'
autoapi_dirs = [str(
    (asb._asb_root).absolute()
)]
autoapi_generate_api_docs = True
autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = []

autoapi_ignore = ["*/test_*.py", "*/ignore/*"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
