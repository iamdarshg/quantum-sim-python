"""
Setup script for Enhanced Quantum Lattice Nuclear Collision Simulator v2.0
"""

from setuptools import setup, Extension
import numpy
import os

# C extension module for high-performance lattice operations
lattice_extension = Extension(
    'lattice_c_extensions',
    sources=['lattice_c_extensions.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-fopenmp', '-O3', '-march=native', '-std=c99'],
    extra_link_args=['-fopenmp'],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
)

setup(
    name='quantum_lattice_simulator',
    version='2.0.0',
    description='Enhanced Quantum Lattice Nuclear Collision Simulator with Systematic Accuracy Improvements',
    long_description="""
Advanced quantum field theory simulator for nuclear collisions implementing:
- QED with Feynman loop corrections
- QCD with Wilson improved fermions and HMC updates  
- Electroweak theory with Higgs mechanism
- Multi-scale lattice analysis for systematic error control
- Full nuclear database with realistic structure
- High-performance multithreaded computing
- Real-time systematic error analysis
    """,
    author='Advanced Physics Simulation Team',
    author_email='physics@quantum-lattice.org',
    url='https://github.com/quantum-lattice/simulator',

    py_modules=[
        'enhanced_quantum_simulator',
        'nuclear_database', 
        'systematic_analysis'
    ],

    ext_modules=[lattice_extension],

    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
        'psutil>=5.8.0'
    ],

    extras_require={
        'gui': ['tkinter'],
        'gpu': ['cupy-cuda12x'], 
        'mpi': ['mpi4py'],
        'hdf5': ['h5py']
    },

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GPL v3.0',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9', 
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C',
    ],

    python_requires='>=3.8',
)
