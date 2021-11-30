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
    description='A Python 3 package for playing around with aerodynamics ideas related to vortex lattice methods, coupled viscous/inviscid methods, automatic differentiation for gradient computation, aircraft design optimization, and the like. Work in progress!',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://peterdsharpe.github.io/AeroSandbox/',
    author_email='pds@mit.edu',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
    ],
    keywords='aerodynamics airplane cfd mdo mdao aircraft design aerospace optimization automatic differentiation',
    packages=find_packages(exclude=['docs', 'media', 'examples', 'studies']),
    python_requires='>=3.6',
    install_requires=[
        'numpy >= 1',
        'scipy >= 1',
        'casadi >= 3.5.5',
        'plotly >= 4',
        'pandas >= 1',
        'matplotlib >= 3',
        'seaborn >= 0.10',
    ],
    include_package_data=True,
    package_data={
        'Airfoil database': ['*.dat'],  # include all airfoil *.dat files
    },
    project_urls={  # Optional
        'Source'     : 'https://github.com/peterdsharpe/AeroSandbox',
        'Bug Reports': 'https://github.com/peterdsharpe/AeroSandbox/issues',
    },
)
