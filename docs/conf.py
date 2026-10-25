# Sphinx configuration (https://www.sphinx-doc.org/en/master/usage/configuration.html)
from importlib.metadata import version as package_version

project = 'ReactEA'
author = 'João Correia'
copyright = '2023, João Correia, Vítor Pereira and Miguel Rocha'
release = package_version('reactea')
version = '.'.join(release.split('.')[:2])

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# pages are written in Markdown (MyST)
source_suffix = {'.md': 'markdown'}
myst_enable_extensions = ['colon_fence']
myst_heading_anchors = 3
exclude_patterns = ['_build']

# API reference from the (NumPy style) docstrings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {'members': True, 'show-inheritance': True}
napoleon_google_docstring = False
# optional dependencies of some evaluation functions
autodoc_mock_imports = ['dockstring', 'deepsweet_models', 'ensemble']

html_theme = 'furo'
html_title = f'ReactEA {release}'