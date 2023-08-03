"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
import codecs

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


### Set up tools to get version
def read(rel_path):
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


### Do the setup
setup(
    name='AeroSandbox',
    author='Peter Sharpe',
    version=get_version("aerosandbox/__init__.py"),
    description='AeroSandbox is a Python package for design optimization of engineered systems such as aircraft.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://peterdsharpe.github.io/AeroSandbox/',
    author_email='pds@mit.edu',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='aerodynamics airplane cfd mdo mdao aircraft design aerospace optimization automatic differentiation structures propulsion',
    packages=find_packages(exclude=['docs', 'media', 'examples', 'studies']),
    python_requires='>=3.8',
    install_requires=[
        'numpy >= 1.20.0, <1.25a0',
        'scipy >= 1.7.0',
        'casadi ~= 3.6.0',
        'pandas >= 1',
        'matplotlib >= 3.7.0',
        'seaborn >= 0.11',
        'tqdm >= 4',
        'sortedcontainers >= 2',
        'neuralfoil >= 0.1.9'
    ],
    extras_require={
        "full": [
            'plotly >= 5',
            'pyvista >= 0.31',
            'ipyvtklink >= 0.2',
            'trimesh >= 3',
            'sympy >= 1',
            'cadquery >= 2; python_version >="3.8"',
            'shapely >= 2',
        ],
        "test": [
            'pytest',
            'nbval'
        ],
        "docs": [
            'sphinx',
            'furo',
            'sphinx-autoapi',
        ],
    },
    include_package_data=True,
    package_data={
        'Airfoil database': ['*.dat'],  # include all airfoil *.dat files
    },
    project_urls={  # Optional
        'Source'     : 'https://github.com/peterdsharpe/AeroSandbox',
        'Bug Reports': 'https://github.com/peterdsharpe/AeroSandbox/issues',
    },
)
