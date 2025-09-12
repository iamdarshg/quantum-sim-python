"""
Setup script for Quantum Lattice Nuclear Collision Simulator v2.0
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except:
        return "Advanced quantum field theory simulator for nuclear collisions"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except:
        return ["numpy>=1.20.0", "scipy>=1.7.0", "matplotlib>=3.3.0"]

setup(
    name="quantum-lattice-simulator",
    version="2.0.0",
    author="Advanced Physics Simulation Team", 
    author_email="physics@quantum-lattice.org",
    description="Advanced quantum field theory simulator for nuclear collisions",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/quantum-lattice/simulator",
    
    packages=find_packages(),
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    python_requires=">=3.8",
    
    install_requires=read_requirements(),
    
    extras_require={
        "gui": ["tkinter"],
        "visualization": ["plotly>=5.0.0", "matplotlib>=3.3.0"],
        "performance": ["numba>=0.55.0"],
        "all": ["plotly>=5.0.0", "matplotlib>=3.3.0", "numba>=0.55.0"]
    },
    
    entry_points={
        "console_scripts": [
            "quantum-lattice=quantum_lattice:launch_gui",
        ],
    },
    
    include_package_data=True,
    
    keywords="physics simulation nuclear collision quantum lattice QCD QED",
    project_urls={
        "Bug Reports": "https://github.com/quantum-lattice/simulator/issues",
        "Source": "https://github.com/quantum-lattice/simulator",
        "Documentation": "https://quantum-lattice.readthedocs.io",
    },
)