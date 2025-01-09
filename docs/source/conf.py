# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pore2Chip'
copyright = '2025, Aramy Truong, Maruti Mudunuru, Md Lal Mamud'
author = 'Aramy Truong, Maruti Mudunuru, Md Lal Mamud'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'conestack'
html_static_path = ['_static']
html_theme_options ={
    'logo_url': '_static/p2c_logo.svg',
    'logo_title': 'Pore2Chip',
    'logo_width': '70px',
    'logo_height': '70px'
}