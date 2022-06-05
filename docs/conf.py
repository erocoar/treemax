"""Sphinx configuration."""
project = "treemax"
author = "Frederik Tiedemann"
copyright = "2022, Frederik Tiedemann"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
