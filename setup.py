"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from pathlib import Path

# Get the long description from the README file
README_path = Path(__file__).parent / "README.md"
with open(README_path, encoding='utf-8') as f:
    long_description = f.read()

### Get the version number dynamically
init_path = Path(__file__).parent / "aerosandbox" / "__init__.py"

with open(init_path) as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        raise RuntimeError("Unable to find version string.")

### Do the setup
setup(
    name='AeroSandbox',
    author='Peter Sharpe',
    version=version,
    description='AeroSandbox is a Python package that helps you design and optimize aircraft and other engineered systems.',
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
    packages=find_packages(exclude=['docs', 'media', 'studies', 'tutorial']),
    python_requires='>=3.8',
    install_requires=[
        'numpy >= 1.20.0, <2.0a0',
        'scipy >= 1.7.0',
        'casadi >= 3.6.4',
        'pandas >= 2',
        'matplotlib >= 3.7.0',
        'seaborn >= 0.11',
        'tqdm >= 4',
        'sortedcontainers >= 2',
        'dill >= 0.3',
        'neuralfoil >= 0.2.1, <0.3.0'
    ],
    extras_require={
        "full": [
            'plotly >= 5',
            'pyvista >= 0.31',
            'trimesh >= 3',
            'sympy >= 1',
            'shapely >= 2',
        ],
        "test": [
            'pytest<=8.0.2',
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
